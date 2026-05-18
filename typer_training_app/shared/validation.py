def validate_preference_dataset(records):
    if not records:
        raise ValueError("Dataset empty")

    required = {"prompt", "chosen", "rejected"}

    cleaned = []

    for i, r in enumerate(records):

        if not required.issubset(r.keys()):
            raise ValueError(f"Missing columns at row {i}: {required}")

        prompt = r["prompt"]

        chosen = r["chosen"] if "chosen" in r else r.get("response_a")
        rejected = r["rejected"] if "rejected" in r else r.get("response_b")

        if not chosen or not rejected:
            raise ValueError(f"Invalid row {i}: missing chosen/rejected")

        cleaned.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    return cleaned