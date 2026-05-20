"""
Usage:
  # Validate your dataset
  python prepare_dataset.py --mode validate --dataset_path data/preference_dataset.csv

  # Stats on an existing dataset
  python prepare_dataset.py --mode stats --dataset_path data/preference_dataset.csv
"""

import argparse
import csv
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "finki-ukim/VezilkaLLM-Instruct"


def read_csv(path: str) -> list:
  with open(path, "r", encoding="utf-8", newline="") as f:
    return list(csv.DictReader(f))


def write_csv(path: str, rows: list, fieldnames: list):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def validate_dataset(
  path: str,
  col_prompt: str = "prompt",
  col_chosen: str = "chosen",
  col_rejected: str = "rejected"
):
  """
  Checks that the CSV has the required columns and that every row has:
    - Non-empty values in the prompt, chosen, and rejected columns
    - chosen != rejected
  """
  errors = []
  warnings = []

  try:
    records = read_csv(path)
  except FileNotFoundError:
    print(f"  X File not found: {path}")
    sys.exit(1)

  if not records:
    print("  X File is empty or has only a header row.")
    sys.exit(1)

  # Check required columns
  required_cols = {col_prompt, col_chosen, col_rejected}
  actual_cols = set(records[0].keys())
  missing_cols = required_cols - actual_cols
  if missing_cols:
    print(f"  X Missing columns: {missing_cols}")
    print(f"    Found columns: {actual_cols}")
    print(f"    Expected: '{col_prompt}', '{col_chosen}', '{col_rejected}'")
    sys.exit(1)

  for i, row in enumerate(records, start=2):  # start=2 accounts for header row
    for key in [col_prompt, col_chosen, col_rejected]:
      if not row.get(key, "").strip():
        errors.append(f"Row {i}: '{key}' is empty")

    if row.get(col_chosen, "").strip() == row.get(col_rejected, "").strip():
      warnings.append(f"Row {i}: chosen == rejected (no preference signal)")

  print(f"\n{'='*50}")
  print(f"  Dataset Validation: {path}")
  print(f"{'='*50}")
  print(f"  Total records: {len(records)}")
  print(f"  Errors:        {len(errors)}")
  print(f"  Warnings:      {len(warnings)}")

  if errors:
    print("\n  ERRORS:")
    for e in errors[:20]:
      print(f"    X {e}")
    if len(errors) > 20:
      print(f"    ... and {len(errors) - 20} more")

  if warnings:
    print("\n  WARNINGS:")
    for w in warnings[:10]:
      print(f"    ! {w}")

  if not errors:
    print("\n  OK Dataset is valid and ready for training.")
  else:
    print("\n  X Fix the errors above before training.")
    sys.exit(1)

  return records


def dataset_stats(
  path: str,
  col_prompt: str = "prompt",
  col_chosen: str = "chosen",
  col_rejected: str = "rejected"
):
  records = read_csv(path)

  if not records:
    print("Dataset is empty.")
    return

  prompt_lens   = [len(r[col_prompt].split())   for r in records]
  chosen_lens   = [len(r[col_chosen].split())   for r in records]
  rejected_lens = [len(r[col_rejected].split()) for r in records]

  def stats(name, vals):
    print(f"\n  {name}:")
    print(f"    min={min(vals)}, max={max(vals)}, "
          f"mean={sum(vals)/len(vals):.1f}, "
          f"median={sorted(vals)[len(vals)//2]}")

  print(f"\n{'='*50}")
  print(f"  Dataset Statistics: {path}")
  print(f"{'='*50}")
  print(f"  Total samples: {len(records)}")
  stats("Prompt length (words)", prompt_lens)
  stats("Chosen length (words)", chosen_lens)
  stats("Rejected length (words)", rejected_lens)



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=["validate", "stats"],
                      required=True)
  parser.add_argument("--dataset_path", type=str, help="Path to training CSV dataset")


  parser.add_argument("--col_prompt",   type=str, default="prompt",   help="CSV column name for the prompt")
  parser.add_argument("--col_chosen",   type=str, default="chosen",   help="CSV column name for the chosen response")
  parser.add_argument("--col_rejected", type=str, default="rejected", help="CSV column name for the rejected response")

  args = parser.parse_args()

  if args.mode == "validate":
    assert args.dataset_path, "--dataset_path required"
    validate_dataset(
      args.dataset_path,
      col_prompt=args.col_prompt,
      col_chosen=args.col_chosen,
      col_rejected=args.col_rejected
    )

  elif args.mode == "stats":
    assert args.dataset_path, "--dataset_path required"
    dataset_stats(
      args.dataset_path,
      col_prompt=args.col_prompt,
      col_chosen=args.col_chosen,
      col_rejected=args.col_rejected
    )



if __name__ == "__main__":
  main()
