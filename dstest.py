import json
from transformers import AutoTokenizer

# Path to your model and JSONL file
model_name_or_path = "/home/jovyan/shared/Umair/models/DeepSeek-R1/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
file_path = "/home/jovyan/trainingmodel/querykg/data/scholarlydata_dataset.jsonl"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

max_token_length = 0
max_text = None

with open(file_path, "r") as f:
    for line in f:
        example = json.loads(line)
        tokens = tokenizer(example["text"], truncation=False)["input_ids"]
        token_len = len(tokens)
        if token_len > max_token_length:
            max_token_length = token_len
            max_text = example["text"]

print("Maximum token length in dataset:", max_token_length)
print("\nExample text for maximum token length:\n")
print(max_text)