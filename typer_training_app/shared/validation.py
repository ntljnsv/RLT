import csv
import sys


def validate_preference_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    required = {"prompt", "chosen", "rejected"}

    if not records:
        raise ValueError("Dataset empty")

    if not required.issubset(records[0].keys()):
        raise ValueError(f"Missing columns: {required}")

    for i, r in enumerate(records):
        if not r["prompt"] or not r["chosen"] or not r["rejected"]:
            raise ValueError(f"Empty field in row {i}")

        if r["chosen"] == r["rejected"]:
            print(f"Warning row {i}: no preference signal")

    return records
