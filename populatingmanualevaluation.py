import json
from SPARQLWrapper import SPARQLWrapper, JSON

# Set your endpoint URL
endpoint_url = "http://localhost:3030/scholarly/sparql"
# Path to your JSON file with queries (adjust filename as needed)
filename = "/home/jovyan/trainingmodel/querykg/code/deepseek/evaluation_queries.json"

def run_sparql_query(sparql_query: str):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except Exception as e:
        print("SPARQL query failed:", e)
        return None
    return results

def update_queries_with_manual_results(filename: str):
    # Load the existing JSON file
    with open(filename, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    # Iterate over each query in the JSON file
    for query_obj in queries:
        query_sparql = query_obj.get("query_sparql", "")
        if not query_sparql:
            continue

        results = run_sparql_query(query_sparql)
        print('results done')
        if results is None:
            print('results not done')
            continue

        # Assume that the first variable in the query is the main identifier.
        head_vars = results.get("head", {}).get("vars", [])
        if not head_vars:
            continue
        main_var = head_vars[0]
        bindings = results.get("results", {}).get("bindings", [])
        values = []
        for binding in bindings:
            value = binding.get(main_var, {}).get("value", "")
            if value:
                values.append(value)
        # Update the field with comma separated values
        query_obj["query_results_manual"] = ", ".join(values)
    
    # Save the updated queries back to the JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    print(f"Updated 'query_results_manual' for {len(queries)} queries in {filename}.")

if __name__ == "__main__":
    update_queries_with_manual_results(filename)
