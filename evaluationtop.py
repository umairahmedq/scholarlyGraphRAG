import json

def compare_and_update_match_percentage(filename: str):
    """
    Reads the JSON file, compares 'query_results_manual' vs 'query_results_code',
    calculates how many URIs from the smaller set appear in the larger set (as %),
    and saves that in a new 'match_percentage' field in each query object.
    Only do this for objects where both fields are non-empty.
    """

    # Load the JSON data
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    for query_obj in data:
        # Get the manual and code results
        manual_str = query_obj.get("query_results_manual", "").strip()
        code_str = query_obj.get("query_results_code", "").strip()

        # We'll only compute match if both are non-empty
        if not manual_str or not code_str:
            continue  # skip if either is empty

        # Parse them as comma-separated URIs
        # and strip whitespace around each item
        manual_list = [s.strip() for s in manual_str.split(",") if s.strip()]
        code_list   = [s.strip() for s in code_str.split(",")   if s.strip()]

        # Convert to sets for easier intersection
        manual_set = set(manual_list)
        code_set   = set(code_list)

        # Identify the smaller set and the larger set
        if len(manual_set) <= len(code_set):
            smaller = manual_set
            bigger  = code_set
        else:
            smaller = code_set
            bigger  = manual_set

        # Intersection
        intersection = smaller.intersection(bigger)
        match_count  = len(intersection)
        smaller_size = len(smaller)

        if smaller_size == 0:
            # avoid division by zero, though we already skip if one is empty
            match_percentage = 0.0
        else:
            match_percentage = (match_count / smaller_size) * 100.0

        # Add or update the field in the JSON object
        query_obj["match_percentage"] = round(match_percentage, 2)

    # Write the updated data back to the file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated 'match_percentage' for queries in {filename}.")

if __name__ == "__main__":
    # Adjust the filename to match your evaluation queries JSON
    EVAL_QUERIES_JSON = "/home/jovyan/trainingmodel/querykg/code/deepseek/evaluation_queries.json"
    compare_and_update_match_percentage(EVAL_QUERIES_JSON)
