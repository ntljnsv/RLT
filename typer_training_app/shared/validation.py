def validate_preference_dataset(records):
    if not records:
        raise ValueError("Dataset empty")

    cleaned = []

    for i, r in enumerate(records):

        prompt = r.get("prompt")
        chosen = r.get("chosen") or r.get("response_a")
        rejected = r.get("rejected") or r.get("response_b")

        if not prompt:
            raise ValueError(f"Row {i}: missing prompt")

        if not chosen or not rejected:
            raise ValueError(f"Row {i}: missing chosen/rejected (or response_a/b)")

        cleaned.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    return cleaned
