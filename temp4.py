import os
import re
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import torch

from sentence_transformers import SentenceTransformer, util
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# --------------------------------------------------------------------------
# 1) REPLACE THE OLD CLASSIFICATION FUNCTION WITH YOUR NEW 4-LABEL MODEL
# --------------------------------------------------------------------------

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Path to your fine-tuned model directory (checkpoint fine-tuned on 4 labels)
model_path = "/home/jovyan/shared/Umair/models/finetuned/roberta_large/scholarlyqueryclassft/checkpoint-180"  # adjust path if needed

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a text classification pipeline using your fine-tuned model
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Mapping of label IDs to actual labels for your 4 desired classes
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
    results = classifier(query)  # returns a list of [ {'label': 'LABEL_...', 'score': ...}, ... ]
    scores = results[0]
    best = max(scores, key=lambda x: x["score"])
    label = label_mapping.get(best["label"], best["label"])
    if best["score"] < 0.5:
        return "papers"
    return label

# --------------------------------------------------------------------------
# 2) KEEP OPENAI FOR SPARQL GENERATION
# --------------------------------------------------------------------------

from openai import OpenAI
client = OpenAI(api_key="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")

# Endpoint config, top_k, etc.
endpoint_url = "http://localhost:3030/scholarly/sparql"  # GraphDB endpoint URL
top_k = 5
min_candidates_threshold = 3

# --------------------------------------------------------------------------
# 3) ONTOLOGY PREFIXES
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# 4) FEW-SHOT SPARQL GENERATION (UNCHANGED)
# --------------------------------------------------------------------------

def generate_sparql(query: str, query_type: str) -> str:
    few_shot_prompt = (
        "Consider a scholarly knowledge graph with information about conferences (e.g., ESWC, ACL), "
        "papers (including titles, abstracts, keywords), authors (with names and affiliations), and "
        "organizations (such as universities or research institutions).\n"
        "Examples:\n"
        "1. conferences: 'List all conferences in 2020', 'Show me conferences held in Europe.'\n"
        "2. papers: 'Give me all research papers on deep learning', 'List papers published in 2019.'\n"
        "3. authors: 'Show me all authors in the dataset', 'List authors who have published multiple papers.'\n"
        "4. organizations: 'Which organizations are present?', 'List all organizations involved in the conferences.'\n"
        "5. papers_by_conference: 'Show me papers from the ISWC 2020 conference', 'List all papers presented at ACL 2019.'\n"
        "6. papers_by_author: 'Give me papers by Aldo Gangemi', 'List papers authored by John Doe.'\n"
        "7. papers_by_organization: 'Show me papers published by Oxford University', 'List research articles from MIT.'\n"
        "8. authors_by_organization: 'List authors affiliated with Harvard University', 'Which authors are from Stanford?'\n"
        "9. authors_by_conference: 'Who presented at ISWC 2020?', 'List authors who published in the ACL conference.'\n"
    )
    
    few_shot_examples = (
        "Example Query 1 (Conferences):\n"
        "Query:\n"
        "  SELECT ?conference (SAMPLE(?acronym) AS ?acronymVal) (SAMPLE(?title) AS ?titleVal) (SAMPLE(?start) AS ?startDateVal) (SAMPLE(?end) AS ?endDateVal) (SAMPLE(?location) AS ?locationVal)\n"
        "  WHERE {\n"
        "    ?conference a cowl:Conference .\n"
        "    OPTIONAL { ?conference cowl:acronym ?acronym. }\n"
        "    OPTIONAL { ?conference rdfs:label ?title. }\n"
        "    OPTIONAL { ?conference cowl:startDate ?start. }\n"
        "    OPTIONAL { ?conference cowl:endDate ?end. }\n"
        "    OPTIONAL { ?conference cowl:location ?location. }\n"
        "    # Apply absolute filters only (use CONTAINS with lower-case matching even for dates).\n"
        "  }\n"
        "  GROUP BY ?conference\n"
        "Sample Result Row:\n"
        "  ?conference = https://w3id.org/scholarlydata/conference/eswc2024\n"
        "  ?acronymVal = \"eswc2024\"\n"
        "  ?titleVal = \"eswc 2024\"\n"
        "  ?startDateVal = \"2016-05-26t09:00:00\"\n"
        "  ?endDateVal = \"2016-05-30t18:00:00\"\n"
        "  ?locationVal = \"hersonissos, crete, greece\"\n\n"
        
        "Example Query 2 (Papers):\n"
        "Query:\n"
        "  SELECT ?paper (SAMPLE(?title) AS ?paperTitle) (SAMPLE(?abstract) AS ?paperAbstract) \n"
        "         (GROUP_CONCAT(DISTINCT COALESCE(?cowlKeyword, ?dctermsSubject, ?dcSubject); separator=\", \") AS ?allKeywords) \n"
        "         (GROUP_CONCAT(DISTINCT ?partOf; separator=\", \") AS ?conference) \n"
        "         (GROUP_CONCAT(DISTINCT COALESCE(?creatorName, ?creatorLabel, ?creatorGiven); separator=\", \") AS ?authorNames) \n"
        "         (GROUP_CONCAT(DISTINCT ?orgName; separator=\", \") AS ?institutes)\n"
        "  WHERE {\n"
        "    ?paper a cowl:InProceedings .\n"
        "    OPTIONAL { ?paper cowl:title ?title . }\n"
        "    OPTIONAL { ?paper cowl:abstract ?abstract . }\n"
        "    OPTIONAL { ?paper cowl:keyword ?cowlKeyword . }\n"
        "    OPTIONAL { ?paper dcterms:subject ?dctermsSubject . }\n"
        "    OPTIONAL { ?paper dc:subject ?dcSubject . }\n"
        "    OPTIONAL { ?paper cowl:isPartOf ?partOf . }\n"
        "    OPTIONAL { \n"
        "       ?paper dc:creator ?creator .\n"
        "       OPTIONAL { ?creator cowl:name ?creatorName. }\n"
        "       OPTIONAL { ?creator rdfs:label ?creatorLabel. }\n"
        "       OPTIONAL { ?creator cowl:givenName ?creatorGiven. }\n"
        "       OPTIONAL {\n"
        "         ?creator cowl:hasAffiliation ?aff.\n"
        "         OPTIONAL { ?aff cowl:withOrganisation ?org. OPTIONAL { ?org cowl:name ?orgName. } }\n"
        "       }\n"
        "    }\n"
        "  }\n"
        "  GROUP BY ?paper\n"
        "Sample Result Row:\n"
        "  ?paper = https://w3id.org/scholarlydata/inproceedings/www2012/poster/167\n"
        "  ?paperTitle = \"multiple spreaders affect the indirect influence on twitter\"\n"
        "  ?paperAbstract = \"most studies on social influence have focused on direct influence, while another ...\"\n"
        "  ?allKeywords = \"indirect influence, twitter, spreaders\"\n"
        "  ?conference = \"https://w3id.org/scholarlydata/conference/www/2012/proceedings\"\n"
        "  ?authorNames = \"john doe, jane smith\"\n"
        "  ?institutes = \"oxford, imperial\"\n\n"
        
        "Example Query 3 (Authors):\n"
        "Query:\n"
        "  SELECT ?person (SAMPLE(COALESCE(?givenName, ?familyName, ?name)) AS ?authorNameVal) \n"
        "         (GROUP_CONCAT(DISTINCT ?hasAffiliation; separator=\", \") AS ?affiliations)\n"
        "  WHERE {\n"
        "    ?person a cowl:Person .\n"
        "    OPTIONAL { ?person cowl:givenName ?givenName. }\n"
        "    OPTIONAL { ?person cowl:familyName ?familyName. }\n"
        "    OPTIONAL { ?person cowl:name ?name. }\n"
        "    OPTIONAL { ?person cowl:hasAffiliation ?hasAffiliation. }\n"
        "  }\n"
        "  GROUP BY ?person\n"
        "Sample Result Row:\n"
        "  ?person = https://w3id.org/scholarlydata/person/aldo-gangemi\n"
        "  ?authorNameVal = \"aldo gangemi\"\n"
        "  ?affiliations = \"oxford\"\n\n"
        
        "Example Query 4 (Organizations):\n"
        "Query:\n"
        "  SELECT ?org (SAMPLE(?orgName) AS ?orgNameVal) (SAMPLE(?desc) AS ?orgDesc)\n"
        "  WHERE {\n"
        "    ?org a cowl:Organisation .\n"
        "    OPTIONAL { ?org cowl:name ?orgName. }\n"
        "    OPTIONAL { ?org cowl:description ?desc. }\n"
        "  }\n"
        "  GROUP BY ?org\n"
        "Sample Result Row:\n"
        "  ?org = https://w3id.org/scholarlydata/organisation/university-of-oxford\n"
        "  ?orgNameVal = \"oxford university\"\n"
        "  ?orgDesc = \"\"\n\n"
    )

    extra_instruction = (
        "All filters should use CONTAINS and lower-case matching, even for dates if matching as string, usually dates are in the format '2017-05-31T15:30:00'.\n"
        "Always use base names for organizations (e.g.: 'oxford university' => 'oxford').\n"
        "Always check names in both organizations and authors.\n"
        "Make sure to have all filters as mandatory except the semantic ones (for example applying on title or abstract).\n"
        "If the query mentions 'papers by X', also check authors & org fields. If authors by org => filter on affiliation.\n"
        "If authors by conference => filter on that conference.\n"
        "Do not filter on the basis of semantic query elements; that would be done later on via embeddings. For example if it asks for papers on some particular topic, do not put filter on it with title or abstract. It will be done later with embedding matching.\n"
    )

    full_prompt = (
        few_shot_prompt
        + "\n"
        + few_shot_examples
        + "\n"
        + extra_instruction
        + f"\nQuery Type: {query_type}\n"
        "Generate a SPARQL query based on the following natural language query. "
        "Do not add semantic text filters on title, abstract, or topic fields; use only absolute filters with CONTAINS in lower-case.\n"
        f"Natural Language Query: {query}\n\n"
        "SPARQL Query (output only the SPARQL query):"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4", or any other
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in generating SPARQL queries for scholarly data. Output only the SPARQL query without extra commentary."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
        )
        sparql_query = ""
        for choice in completion.choices:
            sparql_query += choice.message.content
        # Clean up the content if it has triple backticks, etc.
        sparql_query = sparql_query.strip().replace("```", "")
        sparql_query = re.sub(r"(?i)sparql", "", sparql_query).strip()
        if "prefix" not in sparql_query.lower():
            sparql_query = ontology_prefixes + "\n" + sparql_query
        return sparql_query
    except Exception as e:
        print("Error generating SPARQL query via OpenAI:", e)
        return ""

# --------------------------------------------------------------------------
# 5) RUN SPARQL QUERY
# --------------------------------------------------------------------------

def run_sparql_query(sparql_query: str):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(120)
    try:
        results = sparql.query().convert()
    except Exception as e:
        print("SPARQL query failed:", e)
        raise e
    return results

# --------------------------------------------------------------------------
# 6) DATA RETRIEVAL & EMBEDDING PRECOMPUTATION
# --------------------------------------------------------------------------

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

def create_concat_text(entity, df_row, columns):
    parts = []
    for field in columns:
        if df_row.get(field, ""):
            parts.append(df_row[field])
    # Include the URI as well:
    parts.append(df_row.get(entity, ""))
    return ". ".join(parts)

# Build a single "concat_text" for embeddings
papers_df["concat_text"] = papers_df.apply(
    lambda row: create_concat_text("paper", row, ["paperTitle", "paperAbstract", "allKeywords", "conference", "authorNames", "institutes"]), 
    axis=1
)
conferences_df["concat_text"] = conferences_df.apply(
    lambda row: create_concat_text("conference", row, ["acronymVal", "titleVal", "descrVal", "startDateVal", "endDateVal", "locationVal"]),
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

# --------------------------------------------------------------------------
# 7) RERANKING FUNCTIONS (UNCHANGED)
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# 8) EXTRA STEP: SUBMIT THE TOP-K RESULTS BACK TO GPT4O-MINI FOR ANALYSIS
# --------------------------------------------------------------------------

def summarize_results(user_query: str, qtype: str, top_results: list) -> str:
    """
    Takes the user query, the query type (papers/authors/etc.), and the top reranked results.
    Submits them to GPT-4o-mini (or whichever model) to produce a natural language summary/analysis.
    """
    # Build a short textual representation of the top results
    # Example: For papers, show title, authors, etc.
    # For authors, show name, affiliation, etc. You can adjust as needed.
    result_lines = []
    for sim, row in top_results:
        if qtype == "papers":
            title = row.get("paperTitle", "")
            authors = row.get("authorNames", "")
            result_lines.append(f"Title: {title}\nAuthors: {authors}\nSimilarity: {sim:.3f}\n---\n")
        elif qtype == "authors":
            name = row.get("authorNameVal", "")
            aff = row.get("affiliations", "")
            result_lines.append(f"Author: {name}\nAffiliations: {aff}\nSimilarity: {sim:.3f}\n---\n")
        elif qtype == "organizations":
            orgName = row.get("orgNameVal", "")
            desc = row.get("orgDesc", "")
            result_lines.append(f"Organization: {orgName}\nDesc: {desc}\nSimilarity: {sim:.3f}\n---\n")
        elif qtype == "conferences":
            acr = row.get("acronymVal", "")
            title = row.get("titleVal", "")
            loc = row.get("locationVal", "")
            dateStart = row.get("startDateVal", "")
            dateEnd = row.get("endDateVal", "")
            result_lines.append(f"Acronym: {acr}\nTitle: {title}\nLocation: {loc}\nDateStart: {dateStart}\nDateEnd: {dateEnd}\nSimilarity: {sim:.3f}\n---\n")
        else:
            # default: treat as "papers"
            title = row.get("paperTitle", "")
            result_lines.append(f"Title: {title}\nSimilarity: {sim:.3f}\n---\n")

    # Combine into a single string
    results_text = "\n".join(result_lines)

    # Build the prompt for GPT
    prompt_text = f"""
User Query: {user_query}
Query Type: {qtype}
Below are the top results retrieved by our system:

{results_text}

Please analyze these results and provide a concise answer or summary for the user in natural language. Consider the user query above.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant that helps interpret scholarly data queries. Summarize or answer in plain English."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
        )
        final_answer = completion.choices[0].message.content.strip()
        return final_answer
    except Exception as e:
        print("Error generating summary via GPT:", e)
        return "I'm sorry, I couldn't generate a summary at this time."

# --------------------------------------------------------------------------
# 9) COMBINED PIPELINE (MODIFIED TO CALL summarize_results)
# --------------------------------------------------------------------------

def combined_query(user_query: str):
    """
    End-to-end pipeline:
      1) Classify the user query (papers, authors, organizations, conferences).
      2) Generate SPARQL via OpenAI for that category.
      3) Run SPARQL, collect results.
      4) Rerank them with embeddings; fallback to global search if needed.
      5) Submit the final top results to GPT for analysis.
      6) Print GPT's answer for the user.
    """
    qtype = classify_query(user_query)
    print(f"Classified query type: {qtype}")
    
    if qtype == "conferences":
        df = conferences_df
        id_field = "conference"
        expected_cols = conference_columns
        faiss_idx = conference_faiss
        embeddings_dict = conference_embeddings
    elif qtype == "authors":
        df = authors_df
        id_field = "person"
        expected_cols = author_columns
        faiss_idx = author_faiss
        embeddings_dict = author_embeddings
    elif qtype == "organizations":
        df = organizations_df
        id_field = "org"
        expected_cols = org_columns
        faiss_idx = org_faiss
        embeddings_dict = org_embeddings
    else:
        # default => "papers"
        df = papers_df
        id_field = "paper"
        expected_cols = paper_columns
        faiss_idx = paper_faiss
        embeddings_dict = paper_embeddings
    
    sparql_query = generate_sparql(user_query, qtype)
    print("\n[SPARQL Branch] Generated SPARQL Query:")
    print(sparql_query)
    
    try:
        sparql_results = run_sparql_query(sparql_query)
    except Exception as e:
        print("SPARQL query failed (timeout or error). Falling back to global semantic search.")
        initial_candidates = []
    else:
        candidate_ids = set()
        for result in sparql_results.get("results", {}).get("bindings", []):
            ent_id = result.get(id_field, {}).get("value", "")
            if ent_id:
                candidate_ids.add(ent_id)
        print(f"SPARQL returned {len(candidate_ids)} candidate {qtype} record(s).")
        
        if candidate_ids:
            candidates_df = df[df[id_field].isin(candidate_ids)]
            initial_candidates = rerank_candidates(candidates_df, embeddings_dict, user_query, id_field, expected_cols, top_k)
        else:
            initial_candidates = []
    
    if len(initial_candidates) < top_k:
        print("Not enough SPARQL candidates; performing global semantic search for additional results.")
        global_candidates = semantic_search_all(user_query, faiss_idx, df, embeddings_dict, id_field, expected_cols, top_k)
        existing_ids = set([row[id_field] for _, row in initial_candidates])
        for sim, row in global_candidates:
            if row[id_field] not in existing_ids:
                initial_candidates.append((sim, row))
                existing_ids.add(row[id_field])
                if len(initial_candidates) >= top_k:
                    break
    
    # Sort final results by similarity descending
    initial_candidates.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- Top Results (internal) ---")
    for sim, row in initial_candidates:
        print("---------------------------")
        for col in expected_cols:
            print(f"{col}: {row[col]}")
        print(f"Similarity Score: {sim:.3f}")
        print("---------------------------\n")

    # ---------------- NEW: Summarize the top results with GPT-4o-mini ----------------
    summary = summarize_results(user_query, qtype, initial_candidates)
    print("\n--- GPT-4o-mini's Answer/Analysis ---")
    print(summary)
    print("---------------------------------------------------\n")

    # Return the summary if you want to do something else with it
    return summary

# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        combined_query(user_query)
