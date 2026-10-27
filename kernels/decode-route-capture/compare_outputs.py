"""Exact token comparison and a separately labelled observed baseline distance."""


def output_distance(reference, actual):
    a = [r["output_token_ids"] for r in reference["requests"]]
    b = [r["output_token_ids"] for r in actual["requests"]]
    compatible = (reference["selected_prompts"] == actual["selected_prompts"] and len(a) == len(b))
    if not compatible:
        return {"prompts_match": False, "exact_match": False, "token_distance": None}
    differences = []
    for request, (left, right) in enumerate(zip(a, b)):
        indices = [i for i, (x, y) in enumerate(zip(left, right)) if x != y]
        if len(left) != len(right):
            indices.extend(range(min(len(left), len(right)), max(len(left), len(right))))
        differences.append({"request": request, "lengths": [len(left), len(right)],
                            "different_positions": len(indices),
                            "first_difference": indices[0] if indices else None})
    distance = sum(r["different_positions"] for r in differences)
    return {"prompts_match": True, "exact_match": distance == 0,
            "token_distance": distance, "requests": differences}


def compare_outputs(reference, actual, baselines=()):
    result = output_distance(reference, actual)
    distances = [output_distance(reference, baseline) for baseline in baselines]
    if any(not d["prompts_match"] for d in distances):
        raise ValueError("baseline output uses different selected prompts or request count")
    maximum = max((d["token_distance"] for d in distances), default=None)
    result["baseline_variation"] = {
        "comparisons_to_reference": distances, "max_observed_token_distance": maximum,
        "within_observed_token_distance": (
            result["token_distance"] <= maximum
            if maximum is not None and result["token_distance"] is not None else None),
        "interpretation": "Positional token-distance envelope of supplied controls only; "
                          "not equivalence, a statistical tolerance, or causal attribution. "
                          "An exact mismatch remains a failed comparison."}
    return result
