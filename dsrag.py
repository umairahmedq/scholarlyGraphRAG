import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ==================== Step 1: Load the Knowledge Graph Data ====================
kg_file = "/home/jovyan/trainingmodel/querykg/data/scholarlydata_dataset.jsonl"
df = pd.read_json(kg_file, lines=True)
assert "text" in df.columns, "The JSONL file must contain a 'text' column."
print(f"Loaded {len(df)} examples from the knowledge graph.")

# ==================== Step 2: Prepare SPARQL-like and Embedding Retrieval ====================
def run_sparql_query(query: str) -> str:
    """
    Simulate a structured query by filtering the DataFrame.
    For example, if the query mentions 'conference' and a year, we filter accordingly.
    """
    ql = query.lower()
    if "conference" in ql:
        results = df[df["text"].str.lower().str.contains("conference")]
    elif "paper" in ql or "article" in ql:
        results = df[df["text"].str.lower().str.contains("paper|article", regex=True)]
    else:
        results = df.copy()
    
    # Filter by year if a 4-digit year is mentioned.
    year_match = re.search(r'\b(20\d{2})\b', ql)
    if year_match:
        year = year_match.group(1)
        results = results[results["text"].str.contains(year)]
    
    # Combine the top 5 matching entries as context (adjust as needed)
    context = " ".join(results["text"].head(5).tolist())
    return context

# For embedding-based retrieval, prepare document list and build FAISS index.
documents = df["text"].tolist()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
embedding_dim = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(doc_embeddings)
print(f"FAISS index built with {faiss_index.ntotal} documents.")

def run_embedding_query(query: str, top_k: int = 5) -> str:
    """
    Retrieve top_k similar documents via embedding search and return their concatenation.
    """
    q_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(q_embedding, top_k)
    retrieved = [documents[i] for i in indices[0]]
    context = " ".join(retrieved)
    return context

# ==================== Step 3: Load the LLM for Decision and Generation ====================
# We use Flan-T5 as our LLM for both deciding retrieval method and generating the final answer.
llm_model_path = "/home/jovyan/shared/Umair/models/flant5large/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
tokenizer = T5Tokenizer.from_pretrained(llm_model_path)
llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_path)

def llm_decide(query: str) -> str:
    """
    Ask the LLM to decide which retrieval method to use for a given query.
    Prompt the model to output either "sparql" or "embedding".
    """
    prompt = (
        f"Determine the best retrieval method for the following question.\n"
        f"Question: {query}\n"
        "If the question is best answered by structured filtering (e.g., listing conferences, papers by year), reply with 'sparql'. "
        "Otherwise, reply with 'embedding'."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = llm_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=10,
        num_beams=3,
        early_stopping=True
    )
    decision = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    # Basic check: if not exactly "sparql" or "embedding", default to embedding.
    if decision not in ["sparql", "embedding"]:
        decision = "embedding"
    return decision

def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using the LLM given retrieved context and the question.
    """
    prompt = f"Based on the following context: {context}\nAnswer the question: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=150,
        num_beams=5,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ==================== Step 4: Main Function to Process a Query ====================
def process_query(query: str):
    # Use the LLM to decide which retrieval method to use.
    decision = llm_decide(query)
    print(f"LLM decision for retrieval method: {decision}")
    
    if decision == "sparql":
        context = run_sparql_query(query)
        print("\n[Structured Retrieval] Context:")
        print(context)
    else:
        context = run_embedding_query(query, top_k=5)
        print("\n[Embedding-Based Retrieval] Context:")
        print(context)
    
    print("\nGenerating answer...")
    answer = generate_answer(context, query)
    print("\nFinal Answer:")
    print(answer)

# ==================== Step 5: Run the System ====================
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    process_query(user_query)
