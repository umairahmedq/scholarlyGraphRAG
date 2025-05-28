import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BartTokenizerFast,
    BartForSequenceClassification,
    BartConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import torch

# Load CSV dataset (ensure your CSV file has columns "query" and "label")
df = pd.read_csv("/home/jovyan/trainingmodel/querykg/code/deepseek/sparqlquery_ftdata.csv")

# Map labels to integers
label_list = sorted(df["label"].unique())
label_to_id = {label: idx for idx, label in enumerate(label_list)}
print()
df["label_id"] = df["label"].map(label_to_id)

# Stratified split using the pandas DataFrame
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Convert DataFrames into Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict with train and test splits
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Load the BART tokenizer for MNLI
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-mnli")

# Tokenize the queries with padding and truncation
def preprocess_function(examples):
    return tokenizer(examples["query"], truncation=True, padding="max_length", max_length=32)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Remove extra columns and rename the label column to "labels"
encoded_dataset = encoded_dataset.remove_columns(["query", "label"])
encoded_dataset = encoded_dataset.rename_column("label_id", "labels")

# Fine-tuning BART-MNLI without ignore_mismatched_sizes:
# 1. Load a temporary model from the checkpoint (with original num_labels, i.e. 3)
temp_model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
# 2. Create a new configuration with your desired number of labels
num_labels = len(label_list)
config = BartConfig.from_pretrained("facebook/bart-large-mnli")
config.num_labels = num_labels
# 3. Load a new model with this configuration (this initializes a new classification head)
model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli", config=config)
# 4. Manually copy (pad) the classification head weights from temp_model
with torch.no_grad():
    # Get original weights and biases (shape: [3, hidden_size])
    old_weight = temp_model.classification_head.out_proj.weight.data
    old_bias = temp_model.classification_head.out_proj.bias.data
    # Get new weights and biases (shape: [num_labels, hidden_size])
    new_weight = model.classification_head.out_proj.weight.data
    new_bias = model.classification_head.out_proj.bias.data
    # Copy over the pretrained weights for the first 3 classes
    new_weight[:old_weight.size(0)] = old_weight
    new_bias[:old_bias.size(0)] = old_bias

# Set up training arguments
training_args = TrainingArguments(
    output_dir="/home/jovyan/shared/Umair/models/finetuned/bartmnli/scholarlyqueryclassft",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
    max_grad_norm=1.0,
    fp16=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the accuracy metric using the evaluate library
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1).flatten()
    labels = np.array(labels).flatten()
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Optionally, evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
