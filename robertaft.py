import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np

# Load CSV dataset (ensure your CSV file has columns "query" and "label")
df = pd.read_csv("/home/jovyan/trainingmodel/querykg/code/deepseek/sparqlquery_ftdatalab.csv")  # adjust path as needed

# Filter dataframe to only include queries for "authors", "conferences", "organizations", "papers"
keep_labels = {"authors", "conferences", "organizations", "papers"}
df = df[df["label"].isin(keep_labels)].reset_index(drop=True)

# Map labels to integers based on the filtered labels
label_list = sorted(df["label"].unique())
label_to_id = {label: idx for idx, label in enumerate(label_list)}
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

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

# Tokenize the queries with padding and truncation
def preprocess_function(examples):
    return tokenizer(examples["query"], truncation=True, padding="max_length", max_length=16)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Remove extra columns and rename the label column to "labels"
encoded_dataset = encoded_dataset.remove_columns(["query", "label"])
encoded_dataset = encoded_dataset.rename_column("label_id", "labels")

# Load the RoBERTa model for sequence classification
num_labels = len(label_list)
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=num_labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="/home/jovyan/shared/Umair/models/finetuned/roberta_large/scholarlyqueryclassft",
    evaluation_strategy="epoch",
    learning_rate=8e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=9,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    max_grad_norm=1.0,
    fp16=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the accuracy metric using the evaluate library
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert predictions to indices
    predictions = np.argmax(logits, axis=-1).tolist()
    # Flatten the labels in case they're nested
    labels = np.array(labels).flatten().tolist()
    return accuracy_metric.compute(predictions=predictions, references=labels)

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
