import json
from collections import defaultdict

def summarize_evaluation_queries(filename: str):
    """
    Reads a JSON file of evaluation queries (each containing 'query_results_manual',
    'query_results_code', 'match_percentage', and 'query_type').
    
    Produces:
      1. Overall summary of how many queries have both fields non-empty,
         plus how many are >70% matched.
      2. Per query type summary with distribution across match-percentage ranges.
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) Track overall stats
    total_non_empty = 0        # number of queries with both manual+code present
    total_matched_over_70 = 0  # how many have match_percentage > 70

    # 2) Also want stats by query_type
    # We'll keep a structure keyed by query_type:
    # {
    #   "authors": {
    #       "count_non_empty": ...,
    #       "count_matched_over_70": ...,
    #       "range_bins": { "0-25": X, "25-50": Y, "50-75": Z, "75-100": W }
    #   },
    #   "papers": ...
    #   ... etc ...
    # }
    stats_by_type = defaultdict(lambda: {
        "count_non_empty": 0,
        "count_matched_over_70": 0,
        # For distribution bins:
        "range_bins": {
            "0-25": 0,
            "25-50": 0,
            "50-75": 0,
            "75-100": 0
        }
    })

    for query_obj in data:
        query_type = query_obj.get("query_type", "unknown")
        manual_str = query_obj.get("query_results_manual", "").strip()
        code_str   = query_obj.get("query_results_code", "").strip()
        match_perc = query_obj.get("match_percentage", 0.0)

        # Only consider queries with both fields non-empty
        if manual_str and code_str:
            total_non_empty += 1
            # Update type-level stats
            stats_by_type[query_type]["count_non_empty"] += 1

            # Check if >70
            if match_perc > 70:
                total_matched_over_70 += 1
                stats_by_type[query_type]["count_matched_over_70"] += 1

            # Compute distribution bin
            # We'll place each query in exactly one bin based on match_perc
            if match_perc < 25:
                stats_by_type[query_type]["range_bins"]["0-25"] += 1
            elif match_perc < 50:
                stats_by_type[query_type]["range_bins"]["25-50"] += 1
            elif match_perc < 75:
                stats_by_type[query_type]["range_bins"]["50-75"] += 1
            else:
                # 75-100
                stats_by_type[query_type]["range_bins"]["75-100"] += 1

    # PRINT RESULTS

    # Overall:
    print("=== Overall Summary ===")
    print(f"Total queries with non-empty manual+code: {total_non_empty}")
    print(f"Number with match_percentage > 70%: {total_matched_over_70}")
    if total_non_empty > 0:
        overall_pct = (total_matched_over_70 / total_non_empty) * 100.0
        print(f"That's {overall_pct:.2f}% of the queries with both fields.")
    print()

    # By type:
    print("=== Per Query Type Summary ===")
    for qtype, stats in stats_by_type.items():
        c_non_empty = stats["count_non_empty"]
        c_matched   = stats["count_matched_over_70"]
        print(f"\n--- {qtype.upper()} ---")
        print(f"  Queries with non-empty fields: {c_non_empty}")
        print(f"  Matched >70%: {c_matched}")
        if c_non_empty > 0:
            pct_type = (c_matched / c_non_empty) * 100.0
            print(f"  => {pct_type:.2f}% matched over 70% in {qtype}.")
        # Distribution bins
        bins = stats["range_bins"]
        print("  Distribution (match_percentage ranges):")
        print(f"     0-25  : {bins['0-25']}")
        print(f"     25-50 : {bins['25-50']}")
        print(f"     50-75 : {bins['50-75']}")
        print(f"     75-100: {bins['75-100']}")

if __name__ == "__main__":
    # Adjust to your file's name
    EVAL_QUERIES_JSON = "/home/jovyan/trainingmodel/querykg/code/deepseek/evaluation_queries.json"
    summarize_evaluation_queries(EVAL_QUERIES_JSON)
