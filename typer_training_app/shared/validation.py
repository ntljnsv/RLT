def validate_preference_dataset(records):
    if not records:
        raise ValueError("Dataset empty")

    cleaned = []
    skipped_equal = 0

    for i, r in enumerate(records):
        prompt = (
            r.get("prompt")
            or r.get("user_question")
        )
        chosen = (
            r.get("chosen")
            or r.get("response_a")
            or r.get("preferred_answer")
        )
        rejected = (
            r.get("rejected")
            or r.get("response_b")
            or r.get("non_preferred_answer")
        )

        if not prompt or not str(prompt).strip():
            raise ValueError(f"Row {i}: missing prompt")

        if not chosen or not str(chosen).strip():
            raise ValueError(f"Row {i}: missing chosen/preferred answer")

        if not rejected or not str(rejected).strip():
            raise ValueError(f"Row {i}: missing rejected/non-preferred answer")

        chosen = str(chosen).strip()
        rejected = str(rejected).strip()

        if chosen == rejected:
            skipped_equal += 1
            continue

        cleaned.append({
            "prompt": str(prompt).strip(),
            "chosen": chosen,
            "rejected": rejected,
        })

    if skipped_equal:
        print(f"Skipped {skipped_equal} rows with chosen == rejected")

    if not cleaned:
        raise ValueError("No valid preference pairs after filtering")

    return cleaned
