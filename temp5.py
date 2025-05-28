import os
import json
import re
import pandas as pd
import numpy as np
import faiss
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

############################################
# 1) Classification Model (If Needed)
############################################
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Path to your fine-tuned model directory (checkpoint fine-tuned on 4 labels)
model_path = "/home/jovyan/shared/Umair/models/finetuned/roberta_large/scholarlyqueryclassft/checkpoint-180"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

label_mapping = {
    "LABEL_0": "authors",
    "LABEL_1": "conferences",
    "LABEL_2": "organizations",
    "LABEL_3": "papers"
}

def classify_query(query: str) -> str:
    """
    Use the fine-tuned 4-label model to categorize the query into:
       "authors", "conferences", "organizations", or "papers".
    If the top score is below 0.5, default to "papers".
    """
    results = classifier(query)
    scores = results[0]
    best = max(scores, key=lambda x: x["score"])
    label = label_mapping.get(best["label"], best["label"])
    if best["score"] < 0.5:
        return "papers"
    return label

############################################
# 2) SPARQL Endpoint Config
############################################
endpoint_url = "http://localhost:3030/scholarly/sparql"  # Adjust if needed
top_k = 10  # We'll produce 10 results at the end

############################################
# 3) Ontology Prefixes
############################################
ontology_prefixes = (
    "PREFIX cowl: <https://w3id.org/scholarlydata/ontology/conference-ontology.owl#>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX dcterms: <http://purl.org/dc/terms/>\n"
    "PREFIX dc: <http://purl.org/dc/elements/1.1/>\n"
    "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
    "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)

############################################
# 4) Run SPARQL Query
############################################
def run_sparql_query(sparql_query: str):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(120)
    try:
        results = sparql.query().convert()
    except Exception as e:
        print("SPARQL query failed:", e)
        return None
    return results

############################################
# 5) Build DataFrames & FAISS Indices
############################################
def fetch_all_papers():
    query = (
        ontology_prefixes +
        """
        SELECT ?paper
               (SAMPLE(COALESCE(?cowlTitle, ?dcTitle, ?dctermsTitle, ?rdfsLabel)) AS ?paperTitle)
               (SAMPLE(COALESCE(?cowlAbstract, ?dctermsAbstract, ?dcDescription, ?rdfsComment)) AS ?paperAbstract)
               (GROUP_CONCAT(DISTINCT COALESCE(?cowlKeyword, ?dctermsSubject, ?dcSubject); separator=", ") AS ?allKeywords)
               (GROUP_CONCAT(DISTINCT ?partOf; separator=", ") AS ?conference)
               (GROUP_CONCAT(DISTINCT COALESCE(?creatorStr); separator=", ") AS ?authorNames)
               (GROUP_CONCAT(DISTINCT ?orgName; separator=", ") AS ?institutes)
        WHERE {
          ?paper a cowl:InProceedings .
          OPTIONAL { ?paper cowl:title ?cowlTitle. }
          OPTIONAL { ?paper dc:title ?dcTitle. }
          OPTIONAL { ?paper dcterms:title ?dctermsTitle. }
          OPTIONAL { ?paper rdfs:label ?rdfsLabel. }
          
          OPTIONAL { ?paper cowl:abstract ?cowlAbstract. }
          OPTIONAL { ?paper dcterms:abstract ?dctermsAbstract. }
          OPTIONAL { ?paper dc:description ?dcDescription. }
          OPTIONAL { ?paper rdfs:comment ?rdfsComment. }
          
          OPTIONAL { ?paper cowl:keyword ?cowlKeyword. }
          OPTIONAL { ?paper dcterms:subject ?dctermsSubject. }
          OPTIONAL { ?paper dc:subject ?dcSubject. }
          
          OPTIONAL { ?paper cowl:isPartOf ?partOf. }
          
          OPTIONAL { 
            ?paper dc:creator ?creator .
            OPTIONAL { ?creator cowl:name ?creatorName. }
            OPTIONAL { ?creator rdfs:label ?creatorLabel. }
            OPTIONAL { ?creator cowl:givenName ?creatorGiven. }
            BIND( COALESCE(?creatorName, ?creatorLabel, ?creatorGiven) AS ?authorBase )
            OPTIONAL {
              ?creator cowl:hasAffiliation ?aff.
              OPTIONAL { ?aff cowl:withOrganisation ?org.
                         OPTIONAL { ?org cowl:name ?orgName. } }
            }
            BIND( ?authorBase AS ?creatorStr )
          }
        }
        GROUP BY ?paper
        """
    )
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    records = []
    for result in results.get("results", {}).get("bindings", []):
        record = {
            "paper": result.get("paper", {}).get("value", ""),
            "paperTitle": result.get("paperTitle", {}).get("value", ""),
            "paperAbstract": result.get("paperAbstract", {}).get("value", ""),
            "allKeywords": result.get("allKeywords", {}).get("value", ""),
            "conference": result.get("conference", {}).get("value", ""),
            "authorNames": result.get("authorNames", {}).get("value", ""),
            "institutes": result.get("institutes", {}).get("value", "")
        }
        records.append(record)
    return pd.DataFrame(records)

def fetch_all_conferences():
    query = (
        ontology_prefixes +
        """
        SELECT ?conference
               (SAMPLE(?acronym) AS ?acronymVal)
               (SAMPLE(?title) AS ?titleVal)
               (SAMPLE(?descr) AS ?descrVal)
               (SAMPLE(?start) AS ?startDateVal)
               (SAMPLE(?end) AS ?endDateVal)
               (SAMPLE(?location) AS ?locationVal)
        WHERE {
          ?conference a cowl:Conference .
          OPTIONAL { ?conference cowl:acronym ?acronym. }
          OPTIONAL { ?conference rdfs:label ?title. }
          OPTIONAL { ?conference cowl:description ?descr. }
          OPTIONAL { ?conference cowl:startDate ?start. }
          OPTIONAL { ?conference cowl:endDate ?end. }
          OPTIONAL { ?conference cowl:location ?location. }
        }
        GROUP BY ?conference
        """
    )
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    records = []
    for result in results.get("results", {}).get("bindings", []):
        record = {
            "conference": result.get("conference", {}).get("value", ""),
            "acronymVal": result.get("acronymVal", {}).get("value", ""),
            "titleVal": result.get("titleVal", {}).get("value", ""),
            "descrVal": result.get("descrVal", {}).get("value", ""),
            "startDateVal": result.get("startDateVal", {}).get("value", ""),
            "endDateVal": result.get("endDateVal", {}).get("value", ""),
            "locationVal": result.get("locationVal", {}).get("value", "")
        }
        records.append(record)
    return pd.DataFrame(records)

def fetch_all_authors():
    query = (
        ontology_prefixes +
        """
        SELECT ?person
               (SAMPLE(COALESCE(?givenName, ?familyName, ?name)) AS ?authorNameVal)
               (GROUP_CONCAT(DISTINCT ?hasAffiliation; separator=", ") AS ?affiliations)
        WHERE {
          ?person a cowl:Person .
          OPTIONAL { ?person cowl:givenName ?givenName. }
          OPTIONAL { ?person cowl:familyName ?familyName. }
          OPTIONAL { ?person cowl:name ?name. }
          OPTIONAL { ?person cowl:hasAffiliation ?hasAffiliation. }
        }
        GROUP BY ?person
        """
    )
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    records = []
    for result in results.get("results", {}).get("bindings", []):
        record = {
            "person": result.get("person", {}).get("value", ""),
            "authorNameVal": result.get("authorNameVal", {}).get("value", ""),
            "affiliations": result.get("affiliations", {}).get("value", "")
        }
        records.append(record)
    return pd.DataFrame(records)

def fetch_all_organizations():
    query = (
        ontology_prefixes +
        """
        SELECT ?org
               (SAMPLE(?orgName) AS ?orgNameVal)
               (SAMPLE(?desc) AS ?orgDesc)
        WHERE {
          ?org a cowl:Organisation .
          OPTIONAL { ?org cowl:name ?orgName. }
          OPTIONAL { ?org cowl:description ?desc. }
        }
        GROUP BY ?org
        """
    )
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    records = []
    for result in results.get("results", {}).get("bindings", []):
        record = {
            "org": result.get("org", {}).get("value", ""),
            "orgNameVal": result.get("orgNameVal", {}).get("value", ""),
            "orgDesc": result.get("orgDesc", {}).get("value", "")
        }
        records.append(record)
    return pd.DataFrame(records)

print("Fetching all papers from GraphDB...")
papers_df = fetch_all_papers()
print(f"Fetched {len(papers_df)} papers.")

print("Fetching all conferences from GraphDB...")
conferences_df = fetch_all_conferences()
print(f"Fetched {len(conferences_df)} conferences.")

print("Fetching all authors from GraphDB...")
authors_df = fetch_all_authors()
print(f"Fetched {len(authors_df)} authors.")

print("Fetching all organizations from GraphDB...")
organizations_df = fetch_all_organizations()
print(f"Fetched {len(organizations_df)} organizations.")

paper_columns = ["paper", "paperTitle", "paperAbstract", "allKeywords", "conference", "authorNames", "institutes"]
conference_columns = ["conference", "acronymVal", "titleVal", "descrVal", "startDateVal", "endDateVal", "locationVal"]
author_columns = ["person", "authorNameVal", "affiliations"]
org_columns = ["org", "orgNameVal", "orgDesc"]

# Fill missing columns
for col in paper_columns:
    if col not in papers_df.columns:
        papers_df[col] = ""
papers_df.fillna("", inplace=True)

for col in conference_columns:
    if col not in conferences_df.columns:
        conferences_df[col] = ""
conferences_df.fillna("", inplace=True)

for col in author_columns:
    if col not in authors_df.columns:
        authors_df[col] = ""
authors_df.fillna("", inplace=True)

for col in org_columns:
    if col not in organizations_df.columns:
        organizations_df[col] = ""
organizations_df.fillna("", inplace=True)

############################################
# Create 'concat_text' for each row
############################################
def create_concat_text(entity, df_row, columns):
    parts = []
    for field in columns:
        if df_row.get(field, ""):
            parts.append(df_row[field])
    parts.append(df_row.get(entity, ""))  # add the URI
    return ". ".join(parts)

papers_df["concat_text"] = papers_df.apply(
    lambda row: create_concat_text("paper", row,
                                   ["paperTitle", "paperAbstract", "allKeywords", "conference", "authorNames", "institutes"]), 
    axis=1
)
conferences_df["concat_text"] = conferences_df.apply(
    lambda row: create_concat_text("conference", row,
                                   ["acronymVal", "titleVal", "descrVal", "startDateVal", "endDateVal", "locationVal"]),
    axis=1
)
authors_df["concat_text"] = authors_df.apply(
    lambda row: create_concat_text("person", row, ["authorNameVal", "affiliations"]),
    axis=1
)
organizations_df["concat_text"] = organizations_df.apply(
    lambda row: create_concat_text("org", row, ["orgNameVal", "orgDesc"]),
    axis=1
)

############################################
# Embedding + FAISS
############################################
print("Precomputing embeddings for each dataset...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

paper_texts = papers_df["concat_text"].tolist()
paper_embeddings_np = embed_model.encode(paper_texts, convert_to_numpy=True)

conference_texts = conferences_df["concat_text"].tolist()
conference_embeddings_np = embed_model.encode(conference_texts, convert_to_numpy=True)

author_texts = authors_df["concat_text"].tolist()
author_embeddings_np = embed_model.encode(author_texts, convert_to_numpy=True)

org_texts = organizations_df["concat_text"].tolist()
org_embeddings_np = embed_model.encode(org_texts, convert_to_numpy=True)

paper_embeddings = {row["paper"]: paper_embeddings_np[idx] for idx, row in papers_df.iterrows()}
conference_embeddings = {row["conference"]: conference_embeddings_np[idx] for idx, row in conferences_df.iterrows()}
author_embeddings = {row["person"]: author_embeddings_np[idx] for idx, row in authors_df.iterrows()}
org_embeddings = {row["org"]: org_embeddings_np[idx] for idx, row in organizations_df.iterrows()}

embedding_dim = paper_embeddings_np.shape[1]

paper_faiss = faiss.IndexFlatL2(embedding_dim)
paper_faiss.add(paper_embeddings_np)

conference_faiss = faiss.IndexFlatL2(embedding_dim)
conference_faiss.add(conference_embeddings_np)

author_faiss = faiss.IndexFlatL2(embedding_dim)
author_faiss.add(author_embeddings_np)

org_faiss = faiss.IndexFlatL2(embedding_dim)
org_faiss.add(org_embeddings_np)

print(f"FAISS indices built: {paper_faiss.ntotal} papers, {conference_faiss.ntotal} conferences, {author_faiss.ntotal} authors, {org_faiss.ntotal} organizations.")

############################################
# Reranking Functions
############################################
def rerank_candidates(df: pd.DataFrame, embeddings_dict: dict, query: str, id_field: str, expected_cols: list, top_k: int = 5):
    query_tensor = embed_model.encode(query, convert_to_tensor=True)
    candidate_similarities = []
    for idx, row in df.iterrows():
        ent_id = row[id_field]
        emb = embeddings_dict.get(ent_id)
        if emb is None:
            continue
        emb_tensor = torch.tensor(emb, device=query_tensor.device)
        similarity = util.cos_sim(query_tensor, emb_tensor).item()
        candidate_similarities.append((similarity, row))
    candidate_similarities.sort(key=lambda x: x[0], reverse=True)
    return candidate_similarities[:top_k]

def semantic_search_all(query: str, faiss_index, df: pd.DataFrame, embeddings_dict: dict, id_field: str, expected_cols: list, top_k: int = 5):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = df.iloc[indices[0]]
    top_candidates = []
    query_tensor = embed_model.encode(query, convert_to_tensor=True)
    for idx, row in results.iterrows():
        ent_id = row[id_field]
        emb = embeddings_dict.get(ent_id)
        if emb is None:
            continue
        emb_tensor = torch.tensor(emb, device=query_tensor.device)
        similarity = util.cos_sim(query_tensor, emb_tensor).item()
        top_candidates.append((similarity, row))
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    return top_candidates

############################################
# Adjusted Logic for Going to Global Embeddings
############################################
def process_query_and_store_code(query_obj: dict):
    """
    For each query:
      1) Run the stored SPARQL.
      2) If qtype=papers and <5 SPARQL results => fallback to global indexing.
         If qtype in others and <1 SPARQL results => fallback to global indexing.
      3) Combine & re-sort. Keep top10
      4) Save them in query_results_code.
    """
    # Extract the stored SPARQL
    sparql_query_str = query_obj.get("query_sparql", "")
    if not sparql_query_str:
        return

    # Run the SPARQL
    results = run_sparql_query(sparql_query_str)
    if results is None:
        return

    # Identify data
    qtype = query_obj.get("query_type", "papers")
    if qtype == "conferences":
        id_field = "conference"
        df = conferences_df
        expected_cols = conference_columns
        faiss_idx = conference_faiss
        embeddings_dict = conference_embeddings
        min_threshold = 1  # conferences fallback if <1 from SPARQL
    elif qtype == "authors":
        id_field = "person"
        df = authors_df
        expected_cols = author_columns
        faiss_idx = author_faiss
        embeddings_dict = author_embeddings
        min_threshold = 1  # authors fallback if <1 from SPARQL
    elif qtype == "organizations":
        id_field = "org"
        df = organizations_df
        expected_cols = org_columns
        faiss_idx = org_faiss
        embeddings_dict = org_embeddings
        min_threshold = 1  # organizations fallback if <1 from SPARQL
    else:
        # default => "papers"
        id_field = "paper"
        df = papers_df
        expected_cols = paper_columns
        faiss_idx = paper_faiss
        embeddings_dict = paper_embeddings
        min_threshold = 5  # papers fallback if <5 from SPARQL

    # Collect candidate IDs
    head_vars = results.get("head", {}).get("vars", [])
    if not head_vars:
        return
    main_var = head_vars[0]
    candidate_ids = set()
    for binding in results.get("results", {}).get("bindings", []):
        ent_id = binding.get(main_var, {}).get("value", "")
        if ent_id:
            candidate_ids.add(ent_id)

    print(f"SPARQL returned {len(candidate_ids)} {qtype} record(s) for query: {query_obj.get('query_value')}")

    # Build initial candidates from SPARQL results
    if candidate_ids:
        candidates_df = df[df[id_field].isin(candidate_ids)]
        # We'll gather top 10 from them, but let's do a smaller rerank first
        initial_candidates = rerank_candidates(
            candidates_df, embeddings_dict,
            query_obj.get("query_value", ""), id_field, expected_cols, top_k=top_k
        )
    else:
        initial_candidates = []

    # Fallback if fewer than min_threshold
    if len(initial_candidates) < min_threshold:
        # We do a global semantic search
        global_candidates = semantic_search_all(
            query_obj.get("query_value", ""), faiss_idx, df,
            embeddings_dict, id_field, expected_cols, top_k=top_k
        )
        # Combine
        existing_ids = set([row[id_field] for _, row in initial_candidates])
        for sim, row in global_candidates:
            if row[id_field] not in existing_ids:
                initial_candidates.append((sim, row))
                existing_ids.add(row[id_field])
    # Now re-sort them all by similarity, keep top 10
    initial_candidates.sort(key=lambda x: x[0], reverse=True)
    final_candidates = initial_candidates[:top_k]

    # Extract the final URIs
    candidate_uris = [row[id_field] for sim, row in final_candidates]
    results_code_str = ", ".join(candidate_uris)
    query_obj["query_results_code"] = results_code_str
    print(f"Updated query_results_code with top {len(candidate_uris)} URIs for query: {query_obj.get('query_value')}")

############################################
# Main function to load JSON, process up to 10 queries
############################################
def update_query_results_code(queries_json: str, max_to_process: int = 10):
    with open(queries_json, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    processed_count = 0
    for query_obj in queries_data:
        # Only process if "query_results_code" is empty
        if not query_obj.get("query_results_code", "").strip():
            process_query_and_store_code(query_obj)
            processed_count += 1
            if processed_count >= max_to_process:
                break
    
    # Save back
    with open(queries_json, "w", encoding="utf-8") as f:
        json.dump(queries_data, f, indent=2, ensure_ascii=False)
    print(f"Processed and updated {processed_count} queries in {queries_json}.")

############################################
# MAIN
############################################
if __name__ == "__main__":
    # Name of your JSON file with queries:
    EVAL_QUERIES_JSON = "/home/jovyan/trainingmodel/querykg/code/deepseek/evaluation_queries.json"
    
    # Update code-based results for up to 10 queries that have empty "query_results_code"
    update_query_results_code(EVAL_QUERIES_JSON, max_to_process=20)
