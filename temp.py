import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --------------------- Configuration ---------------------
input_file = "/home/jovyan/trainingmodel/querykg/data/graphdb_results.jsonl"  # Input JSON Lines file with structured paper data
structured_output_file = "/home/jovyan/trainingmodel/querykg/data/structured_papers.jsonl"  # Optional: file to save structured output
top_k = 5  # Number of papers to retrieve for a query

# --------------------- Step 1: Load the Data ---------------------
# Read the JSON Lines file into a DataFrame.
df = pd.read_json(input_file, lines=True)

# Expected columns:
expected_columns = ["paper", "paperTitle", "paperAbstract", "allKeywords", "parentDocs", "authorNames"]

# Ensure all expected columns exist; if a column is missing, add it as an empty string.
for col in expected_columns:
    if col not in df.columns:
        df[col] = ""
        
# Replace any null/None values with empty strings.
df.fillna("", inplace=True)

print(f"Loaded {len(df)} records from the file.")

# --------------------- Step 2: Create a Structured DataFrame with Value Extraction ---------------------
def extract_cell(cell):
    """Extract the 'value' if cell is a dict, otherwise return the cell as a stripped string."""
    if isinstance(cell, dict):
        return cell.get("value", "").strip()
    else:
        return str(cell).strip()

def structure_record(row: pd.Series) -> dict:
    return {
        "paper": extract_cell(row["paper"]),
        "paperTitle": extract_cell(row["paperTitle"]),
        "paperAbstract": extract_cell(row["paperAbstract"]),
        "allKeywords": extract_cell(row["allKeywords"]),
        "parentDocs": extract_cell(row["parentDocs"]),
        "authorNames": extract_cell(row["authorNames"])
    }

structured_data = df.apply(structure_record, axis=1).tolist()
structured_df = pd.DataFrame(structured_data)

# Optionally, save the structured data to a new JSONL file.
structured_df.to_json(structured_output_file, orient="records", lines=True)
print(f"Structured paper data saved to {structured_output_file}")

# --------------------- Step 3: Create a Concatenated Text Field ---------------------
def create_concatenated_text(row: pd.Series) -> str:
    """
    Concatenate available fields into a single text string.
    Only include non-empty fields.
    """
    parts = []
    if row.get("paperTitle"):
        parts.append(row["paperTitle"])
    if row.get("paperAbstract"):
        parts.append(row["paperAbstract"])
    if row.get("allKeywords"):
        parts.append(row["allKeywords"])
    if row.get("parentDocs"):
        parts.append(row["parentDocs"])
    if row.get("authorNames"):
        parts.append(row["authorNames"])
    if row.get("paper"):
        parts.append(row["paper"])
    concatenated = ". ".join(parts)
    return concatenated

structured_df["concat_text"] = structured_df.apply(create_concatenated_text, axis=1)

# --------------------- Step 4: Build an Embedding Index ---------------------
# Initialize a sentence embedding model.
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the concatenated texts.
concatenated_texts = structured_df["concat_text"].tolist()
embeddings = embed_model.encode(concatenated_texts, convert_to_numpy=True)

# Build a FAISS index (using L2 distance).
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embeddings)
print(f"FAISS index built with {faiss_index.ntotal} documents.")

# --------------------- Step 5: Query Interface ---------------------
def query_papers(query: str, top_k: int = 5):
    """
    Given a query, encode it and retrieve the top_k matching papers.
    Then print the value of each column for each retrieved paper.
    """
    # Encode the query.
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    results = structured_df.iloc[indices[0]]
    
    print("\nRetrieved Papers:")
    for idx, row in results.iterrows():
        print("\n---------------------------")
        for col in expected_columns:
            print(f"{col}: {row[col]}")
        print("---------------------------\n")

# --------------------- Step 6: Main ---------------------
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query to search for papers (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        query_papers(user_query, top_k=top_k)
