"""
Combine aya preference-pair shards into one JSON array
"""

import json
import os

JSON_FILE = "aya_pref_pairs_10k.json"
JSONL_FILE = "aya_pref_pairs_10k_56k.jsonl"

OUTPUT_FILE = "aya_pref_pairs_combined.json"


def load_json_array(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(data).__name__}")
    return data


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
    return records


def main() -> None:
    records: list[dict] = []

    if os.path.exists(JSON_FILE):
        chunk = load_json_array(JSON_FILE)
        print(f"Loaded {JSON_FILE}: {len(chunk)} rows")
        records.extend(chunk)
    else:
        print(f"WARNING: {JSON_FILE} not found, skipping.")

    if os.path.exists(JSONL_FILE):
        chunk = load_jsonl(JSONL_FILE)
        print(f"Loaded {JSONL_FILE}: {len(chunk)} rows")
        records.extend(chunk)
    else:
        print(f"WARNING: {JSONL_FILE} not found, skipping.")

    if not records:
        raise SystemExit("No input files found; nothing to write.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"\nDone! Combined {len(records)} total rows into '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
