import os
import re
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
from openai import OpenAI
from transformers import pipeline

### CONFIGURATION ###
endpoint_url = "http://localhost:3030/scholarly/sparql"  # GraphDB endpoint URL
top_k = 5
min_candidates_threshold = 3
client = OpenAI(api_key="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")

### SET UP ZERO-SHOT CLASSIFIER ###
def classify_query(query: str) -> str:
    """
    Use zero-shot classification with few-shot examples to determine the query type.
    Candidate labels include:
      - "conferences"                (for queries solely about conferences)
      - "papers"                     (for queries solely about papers)
      - "authors"                    (for queries solely about authors)
      - "organizations"              (for queries solely about organizations)
      - "papers_by_conference"       (for queries asking for papers within a specific conference)
      - "papers_by_author"           (for queries asking for papers by a particular author)
      - "papers_by_organization"     (for queries asking for papers from a specific organization)
      - "authors_by_organization"    (for queries asking for authors affiliated with a specific organization)
      - "authors_by_conference"      (for queries asking for authors presenting at a specific conference)
    If the classifier's top score is low, default to "papers".
    """
    candidate_labels = [
        "conferences", "papers", "authors", "organizations",
        "papers_by_conference", "papers_by_author", "papers_by_organization",
        "authors_by_organization", "authors_by_conference"
    ]
    
    few_shot_prompt = (
        "Consider a scholarly knowledge graph with information about conferences (e.g., ESWC, ACL), papers (including titles, abstracts, keywords), "
        "authors (with names and affiliations), and organizations (such as universities).\n"
        "Examples:\n"
        "1. conferences: 'List all conferences in 2020', 'Show me conferences held in Europe.'\n"
        "2. papers: 'Give me all research papers on deep learning', 'List papers published in 2019.'\n"
        "3. authors: 'Show me all authors in the dataset', 'List authors who have published multiple papers.'\n"
        "4. organizations: 'Which organizations (universities) are present in the data?', 'List all organizations involved in the conferences.'\n"
        "5. papers_by_conference: 'Show me papers from the ISWC 2020 conference', 'List all papers presented at ACL 2019.'\n"
        "6. papers_by_author: 'Give me papers by Aldo Gangemi', 'List papers authored by John Doe.'\n"
        "7. papers_by_organization: 'Show me papers published by Oxford University', 'List research articles from MIT.'\n"
        "8. authors_by_organization: 'List authors affiliated with Harvard University', 'Which authors are from Stanford?'\n"
        "9. authors_by_conference: 'Who presented at ISWC 2020?', 'List authors who published in the ACL conference.'\n"
    )
    
    full_prompt = few_shot_prompt + "\nClassify the following query: " + query
    from transformers import pipeline
    classifier_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier_pipe(full_prompt, candidate_labels, multi_label=False)
    best_label = result["labels"][0].lower()
    print("best_label: ")
    print(best_label)
    if result["scores"][0] < 0.5:
        return "papers"
    return best_label


### ONTOLOGY PREFIXES ###
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

### FEW-SHOT SPARQL GENERATION ###
def generate_sparql(query: str, query_type: str) -> str:
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
        "       # Do not concatenate them here; they will be aggregated in separate columns.\n"
        "    }\n"
        "    # Use absolute filters with CONTAINS in lower-case for any filter needed.\n"
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
        "    # When filtering on organization, if the query includes an extended name (e.g., 'oxford university'),\n"
        "    # the filter should only check the base name (e.g., 'oxford').\n"
        "  }\n"
        "  GROUP BY ?org\n"
        "Sample Result Row:\n"
        "  ?org = https://w3id.org/scholarlydata/organisation/university-of-oxford\n"
        "  ?orgNameVal = \"oxford university\"\n"
        "  ?orgDesc = \"\"\n\n"
    )
    
    extra_instruction = ""
    # For dates and text, always use CONTAINS (substring) and lower-case comparisons.
    extra_instruction += ("All filters should use CONTAINS and lower-case matching, even for dates.\n")
    extra_instruction += ("Always use base names for organizations (e.g.: oxford university to oxford).\n")
    extra_instruction += ("Always check names in both organizations and authors.\n")
    extra_instruction += ("make sure to have all the filters as optional and placed at the end of query for less strict filtering.\n")
    if query_type == "papers_by_author":
        extra_instruction += ("If the query asks for papers by a person, use an absolute CONTAINS filter on the author field (in lower-case).\n")
    if query_type == "papers_by_organization":
        extra_instruction += ("If the query asks for papers by an organisation, extract the base name (e.g., from 'oxford university', use 'oxford') and use only the base name and an absolute CONTAINS filter on the organisation field (in lower-case).\n")
    if query_type == "authors_by_organization":
        extra_instruction += ("If the query asks for authors by an organisation, extract the base name and filter on the affiliation field using CONTAINS (in lower-case).\n")
    if query_type == "authors_by_conference":
        extra_instruction += ("If the query asks for authors by conference, use an absolute CONTAINS filter on the conference field (in lower-case) related to their papers.\n")
    # If query mentions papers by xxx, check both author and organization
    extra_instruction += ("If the query asks for papers by 'xxx', check both author and organisation fields using CONTAINS (in lower-case).\n")
    
    full_prompt = (
        f"{ontology_prefixes}\n\n"
        "Below are a few example SPARQL queries along with one sample result row for each to illustrate the expected format and values:\n\n"
        f"{few_shot_examples}\n"
        f"{extra_instruction}\n"
        f"Query Type: {query_type}\n"
        "Generate a SPARQL query based on the following natural language query. "
        "Do not add semantic text filters on title, abstract, or topic fields; use only absolute filters with CONTAINS in lower-case.\n"
        f"Natural Language Query: {query}\n\n"
        "SPARQL Query (output only the SPARQL query):"
    )
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in generating SPARQL queries for scholarly data. Output only the SPARQL query without extra commentary."},
                {"role": "user", "content": full_prompt}
            ]
        )
        sparql_query = ""
        for choice in completion.choices:
            sparql_query += choice.message.content
        sparql_query = sparql_query.strip()
        sparql_query = sparql_query.replace("```", "")
        sparql_query = re.sub(r"(?i)sparql", "", sparql_query).strip()
        if "prefix" not in sparql_query.lower():
            sparql_query = ontology_prefixes + "\n" + sparql_query
        return sparql_query
    except Exception as e:
        print("Error generating SPARQL query via GPT-4o-mini:", e)
        return ""

### FUNCTION: Run SPARQL Query ###
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

### DATA RETRIEVAL & EMBEDDING PRECOMPUTATION ###
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
    parts.append(df_row.get(entity, ""))
    return ". ".join(parts)

# For papers, we now include separate columns for authorNames and institutes.
papers_df["concat_text"] = papers_df.apply(lambda row: create_concat_text("paper", row, ["paperTitle", "paperAbstract", "allKeywords", "conference", "authorNames", "institutes"]), axis=1)
conferences_df["concat_text"] = conferences_df.apply(lambda row: create_concat_text("conference", row, ["acronymVal", "titleVal", "descrVal", "startDateVal", "endDateVal", "locationVal"]), axis=1)
authors_df["concat_text"] = authors_df.apply(lambda row: create_concat_text("person", row, ["authorNameVal", "affiliations"]), axis=1)
organizations_df["concat_text"] = organizations_df.apply(lambda row: create_concat_text("org", row, ["orgNameVal", "orgDesc"]), axis=1)

print("Precomputing embeddings for papers...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
paper_texts = papers_df["concat_text"].tolist()
paper_embeddings_np = embed_model.encode(paper_texts, convert_to_numpy=True)
print("Precomputing embeddings for conferences...")
conference_texts = conferences_df["concat_text"].tolist()
conference_embeddings_np = embed_model.encode(conference_texts, convert_to_numpy=True)
print("Precomputing embeddings for authors...")
author_texts = authors_df["concat_text"].tolist()
author_embeddings_np = embed_model.encode(author_texts, convert_to_numpy=True)
print("Precomputing embeddings for organizations...")
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

### RERANKING FUNCTIONS ###
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

### COMBINED PIPELINE ###
def combined_query(user_query: str):
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
        print("SPARQL query failed (possible timeout). Falling back to global semantic search.")
        initial_candidates = []
    else:
        initial_candidates = []
        candidate_ids = set()
        for result in sparql_results.get("results", {}).get("bindings", []):
            ent_id = result.get(id_field, {}).get("value", "")
            if ent_id:
                candidate_ids.add(ent_id)
        print(f"SPARQL returned {len(candidate_ids)} candidate {qtype} records.")
        if candidate_ids:
            candidates_df = df[df[id_field].isin(candidate_ids)]
            initial_candidates = rerank_candidates(candidates_df, embeddings_dict, user_query, id_field, expected_cols, top_k)
    
    # If the number of SPARQL candidates is less than top_k, perform global semantic search and merge
    if len(initial_candidates) < top_k:
        print("Not enough SPARQL candidates; performing global semantic search for additional results.")
        global_candidates = semantic_search_all(user_query, faiss_idx, df, embeddings_dict, id_field, expected_cols, top_k)
        # Merge without duplicates (based on id_field)
        existing_ids = set([row[id_field] for _, row in initial_candidates])
        for sim, row in global_candidates:
            if row[id_field] not in existing_ids:
                initial_candidates.append((sim, row))
                existing_ids.add(row[id_field])
                if len(initial_candidates) >= top_k:
                    break
        print("\n[Combined Approach] Merged SPARQL and Global Semantic Search Results:")
    
    initial_candidates.sort(key=lambda x: x[0], reverse=True)
    for sim, row in initial_candidates:
        print("\n---------------------------")
        for col in expected_cols:
            print(f"{col}: {row[col]}")
        print(f"Similarity Score: {sim:.3f}")
        print("---------------------------\n")
    return initial_candidates

### MAIN LOOP ###
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        combined_query(user_query)
