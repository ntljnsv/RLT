import pandas as pd
import os

# This script is for combining all files of the dataset into one.

files = [
    "preference_pairs_0_10k.csv",
    "preference_pairs_10k_50k.csv",
    "preference_pairs_50k_100k.csv",
    "preference_pairs_100k_150k.csv",
    "preference_pairs_150k_end.csv"
]

dfs = []
for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        print(f"Loaded {file}: {len(df)} rows")
        dfs.append(df)
    else:
        print(f"WARNING: {file} not found, skipping.")

combined_df = pd.concat(dfs, ignore_index=True)

output_file = "gpp_combined.csv"
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\nDone! Combined {len(combined_df)} total rows into '{output_file}'")
