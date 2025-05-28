import json

def mark_matched_over_70(filename: str):
    """
    Reads the JSON file, looks for the 'match_percentage' field in each object,
    and if it's over 70, sets 'is_matched' = True. Otherwise, False.
    Then saves the file back with this new field.
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    for query_obj in data:
        # Get the current match percentage (default to 0 if missing)
        match_percent = query_obj.get("match_percentage", 0)
        
        # Mark as matched if >70
        if match_percent > 90.0:
            query_obj["is_matched"] = True
        else:
            query_obj["is_matched"] = False

    # Write the updated data back to the file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated 'is_matched' field in {filename} where match_percentage > 70.")

if __name__ == "__main__":
    # Adjust the filename to match your evaluation queries JSON
    EVAL_QUERIES_JSON = "/home/jovyan/trainingmodel/querykg/code/deepseek/evaluation_queries.json"
    mark_matched_over_70(EVAL_QUERIES_JSON)
