import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Path to your fine-tuned model directory (checkpoint fine-tuned on 4 labels)
model_path = "/home/jovyan/shared/Umair/models/finetuned/roberta_large/scholarlyqueryclassft/checkpoint-180"  # adjust checkpoint as needed

# Load the tokenizer and model from the saved directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a text classification pipeline using your fine-tuned model
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Mapping of label IDs to actual labels for the 4-label classification (sorted alphabetically)
label_mapping = {
    "LABEL_0": "authors",
    "LABEL_1": "conferences",
    "LABEL_2": "organizations",
    "LABEL_3": "papers"
}

def classify_query(query: str):
    results = classifier(query)
    # results is a list containing one list of dicts with 'label' and 'score'
    scores = results[0]
    best = max(scores, key=lambda x: x["score"])
    label = label_mapping.get(best["label"], best["label"])
    return label, best["score"]

# Load evaluation examples from CSV file
csv_path = "/home/jovyan/trainingmodel/querykg/code/deepseek/sparqlquery_evaldata.csv"
df = pd.read_csv(csv_path)

# Filter evaluation examples to only include queries for the 4 desired labels
keep_labels = {"authors", "conferences", "organizations", "papers"}
df = df[df["label"].isin(keep_labels)].reset_index(drop=True)

predicted_labels = []
confidences = []

print("Running inference on evaluation examples...\n")
for index, row in df.iterrows():
    query = row["query"]
    expected_label = row["label"]
    pred_label, confidence = classify_query(query)
    predicted_labels.append(pred_label)
    confidences.append(confidence)
    print(f"Query: {query}")
    print(f"Expected: {expected_label}, Predicted: {pred_label}, Confidence: {confidence:.3f}\n")

# Append predictions and confidences to the DataFrame
df["predicted_label"] = predicted_labels
df["confidence"] = confidences

# Calculate overall accuracy
overall_accuracy = (df["label"] == df["predicted_label"]).mean()
print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

# Optionally, save the predictions to a new CSV file
output_csv_path = "/home/jovyan/trainingmodel/querykg/code/deepseek/evaluation_results_scibert.csv"
df.to_csv(output_csv_path, index=False)
print(f"Evaluation results saved to: {output_csv_path}")
