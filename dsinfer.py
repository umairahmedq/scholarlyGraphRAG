# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "if i finetune you the large language model a particular knowledge graph will it be able to answer questions from it?"},
]
pipe = pipeline("text-generation", model="/home/jovyan/shared/Umair/models/DeepSeek-R1/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60", trust_remote_code=True, max_new_tokens=512, device = "cuda")
temp = pipe(messages)
print(temp)