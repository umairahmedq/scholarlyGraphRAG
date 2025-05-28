import os
# Set this as early as possible, before any CUDA allocations:

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# Optionally, adjust tokenizers parallelism as needed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load the entire JSONL file into a Pandas DataFrame.
# The file should have one JSON object per line, each with at least a "text" field.
df = pd.read_json("/home/jovyan/trainingmodel/querykg/data/scholarlydata_dataset.jsonl", lines=True)
assert "text" in df.columns, "The JSONL file must contain a 'text' column."

print(f"Loaded {len(df)} examples for unsupervised training.")

# Define your DeepSeek model path
model_path = "/home/jovyan/shared/Umair/models/DeepSeek-R1/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"

# Load tokenizer and model (use trust_remote_code=True if required)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()
# Set pad token to eos_token and resize token embeddings if necessary
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False  # Disable cache if using gradient checkpointing

# Preprocessing function: wrap text with special tokens and tokenize.
def preprocess_function(examples):
    # Wrap the raw text with delimiters
    texts = [f"<|begin_of_text|>{txt}<|end_of_text|>" for txt in examples["text"]]
    model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
    # For causal LM training, use the same tokens as labels.
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Convert the entire DataFrame to a Hugging Face Dataset (using only the "text" field)
dataset = Dataset.from_pandas(df[["text"]])

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Define LoRA configuration for causal language modeling
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Specify that this is a causal LM task
    r=8,
    lora_alpha=16,
    target_modules=["gate_proj", "up_proj", "down_proj"],  # Adjust these based on DeepSeek architecture
    lora_dropout=0.1,
    bias="none"
)

# Apply PEFT with LoRA to the model
peft_model = get_peft_model(model, lora_config)

# Optionally, print trainable parameters for verification
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
    else:
        print(f"Frozen: {name}")

# Set up training arguments.
training_args = TrainingArguments(
    output_dir='/home/jovyan/shared/Umair/models/finetuned/deepseek_skg/results',
    learning_rate=4e-5,
    per_device_train_batch_size=1,         # Use a small batch size to reduce memory usage
    gradient_accumulation_steps=8,           # Simulate a larger effective batch size
    num_train_epochs=3,
    logging_steps=500,
    save_steps=500,
    fp16=True if torch.cuda.is_available() else False,
)

# Initialize the Trainer with the entire dataset as training data.
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Start fine-tuning.
trainer.train()

# Save the final fine-tuned model and tokenizer.
peft_model.save_pretrained('/home/jovyan/shared/Umair/models/finetuned/deepseek_skg')
tokenizer.save_pretrained('/home/jovyan/shared/Umair/models/finetuned/deepseek_skg')
