import os
import re
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import pipeline
from openai import OpenAI

# --------------------- Configuration ---------------------
# File paths for structured paper data (JSONL)
input_file = "/home/jovyan/trainingmodel/querykg/data/graphdb_results.jsonl"  # Your JSONL file with structured paper data
structured_output_file = "/home/jovyan/trainingmodel/querykg/data/structured_papers.jsonl"  # Optional output file
top_k = 5  # Number of papers to retrieve for embedding queries

# SPARQL endpoint URL for Apache Jena (update as needed, e.g., "http://localhost:3030/dataset/sparql")
endpoint_url = "http://localhost:3030/scholarly/sparql"

# --------------------- OpenAI Client Setup ---------------------
client = OpenAI(api_key="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")

# --------------------- Decision Using Zero-Shot Classification ---------------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def decide_method(query: str) -> str:
    """
    Use zero-shot classification with few-shot examples to decide whether the query
    should be answered using embedding-based retrieval ("embedding") or via a SPARQL query ("sparql").
    """
    prompt = (
        "Example 1: Query: 'List all conferences in 2020.' → Answer: sparql. "
        "Example 2: Query: 'list all the papers.' → Answer: sparql. "
        "Example 3: Query: 'Find me papers that are focused on traffic management' → Answer: embedding. "
        "Example 4: Query: 'show me articles that have used machine learning in it' → Answer: embedding. "
        "Now, classify the following query as either 'embedding' or 'sparql'.\n"
        f"Query: {query}\n"
        "Answer:"
    )
    candidate_labels = ["embedding", "sparql"]
    result = classifier(prompt, candidate_labels, multi_label=False)
    chosen = result["labels"][0].lower()
    return chosen

# --------------------- Function: Generate SPARQL Query Using GPT-4o-mini ---------------------
def generate_sparql(query: str) -> str:
    """
    Generate a SPARQL query from a natural language query.
    The prompt includes ontology prefixes and instructs GPT-4o-mini to output only the SPARQL query.
    """
    ontology_prefixes = (
        "PREFIX cowl: <https://w3id.org/scholarlydata/ontology/conference-ontology.owl#>\n"
       # "PREFIX dc: <http://purl.org/dc/elements/1.1/>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    )
    prompt = (
        f"{ontology_prefixes}\n"
        "Generate a SPARQL query to retrieve whatever extensive records are available from a scholarly knowledge graph based on the following query. Keep all other variables as optional except the main one\n "
        f"Query: {query}\n"
        "SPARQL Query (output only the SPARQL query):"
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in generating SPARQL queries for scholarly data. Output only the SPARQL query without any extra commentary."},
                {"role": "user", "content": prompt}
            ]
        )
        sparql_query = ""
        for choice in completion.choices:
            sparql_query += choice.message.content
        sparql_query = sparql_query.strip()
        import re

        # Assume sparql_query is a string containing your SPARQL query.
        sparql_query = sparql_query.strip()
        
        # Remove all occurrences of triple backticks
        sparql_query = sparql_query.replace("```", "")
        
        # Remove all occurrences of the word "sparql" (case-insensitive)
        sparql_query = re.sub(r"(?i)sparql", "", sparql_query)
        
        # Optionally, strip any additional whitespace that may have resulted
        sparql_query = sparql_query.strip()
        # Ensure ontology prefixes are present
        if "prefix" not in sparql_query:
            sparql_query = ontology_prefixes + "\n" + sparql_query
        return sparql_query
    except Exception as e:
        print("Error generating SPARQL query via GPT-4o-mini:", e)
        return ""

# --------------------- Function: Run SPARQL Query ---------------------
def run_sparql_query(sparql_query: str):
    """
    Execute the SPARQL query against the Apache Jena endpoint and return results.
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

# --------------------- Embedding-Based Retrieval Setup ---------------------
# Load the JSONL file into a DataFrame.
df = pd.read_json(input_file, lines=True)

# Expected columns (as in your file)
expected_columns = ["paper", "paperTitle", "paperAbstract", "allKeywords", "parentDocs", "authorNames"]

# Ensure all expected columns exist.
for col in expected_columns:
    if col not in df.columns:
        df[col] = ""
df.fillna("", inplace=True)
print(f"Loaded {len(df)} records from the file.")

def extract_cell(cell):
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
structured_df.to_json(structured_output_file, orient="records", lines=True)
print(f"Structured paper data saved to {structured_output_file}")

def create_concatenated_text(row: pd.Series) -> str:
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
    return ". ".join(parts)

structured_df["concat_text"] = structured_df.apply(create_concatenated_text, axis=1)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
concatenated_texts = structured_df["concat_text"].tolist()
embeddings = embed_model.encode(concatenated_texts, convert_to_numpy=True)
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embeddings)
print(f"FAISS index built with {faiss_index.ntotal} documents.")

def query_papers_embedding(query: str, top_k: int = 5):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = structured_df.iloc[indices[0]]
    print("\n[Embedding-Based Retrieval] Retrieved Papers:")
    for idx, row in results.iterrows():
        print("\n---------------------------")
        for col in expected_columns:
            print(f"{col}: {row[col]}")
        print("---------------------------\n")

# --------------------- Main Query Processing ---------------------
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        
        method = decide_method(user_query)
        print(f"Classifier decision: {method}")
        
        if method == "embedding":
            query_papers_embedding(user_query, top_k=top_k)
        elif method == "sparql":
            sparql_query = generate_sparql(user_query)
            print("\n[SPARQL Branch] Generated SPARQL Query:")
            print(sparql_query)
            try:
                results = run_sparql_query(sparql_query)
                print("\nSPARQL Query Results:")
                print(results)
                # for binding in results.get("results", {}).get("bindings", []):
                #     for var in expected_columns:
                #         val = binding.get(var, {}).get("value", "")
                #         print(f"{var}: {val}")
                    #print("---------------------------")
            except Exception as e:
                print("Error executing SPARQL query:", e)
        else:
            print("Unknown decision. Defaulting to embedding retrieval.")
            query_papers_embedding(user_query, top_k=top_k)
