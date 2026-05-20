# generate_preference_pairs.py

Generates preference pair datasets for RLHF/DPO training by back-translating answers from an existing HuggingFace SFT dataset through a pivot language. The original answer becomes the **preferred** response and the back-translated version becomes the **non-preferred** response.

## How it works

1. Loads a HuggingFace dataset and extracts all user/assistant Q&A pairs from the conversations
2. Assigns each row a `source_index` (its 0-based position in the full flattened dataset)
3. Optionally slices to a specific subset via `--start` / `--num-samples`
4. Translates each answer into a pivot language (e.g. Macedonian → Spanish)
5. Translates it back to the source language (Spanish → Macedonian)
6. Saves a CSV with four columns: `source_index`, `user_question`, `preferred_answer`, `non_preferred_answer`

Translation is done locally using [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP) models — no external API calls are made.

## Installation

```bash
pip install transformers==5.8.1 torch==2.7.0 datasets==4.8.5 pandas==2.3.1 tqdm==4.67.1 sentencepiece==0.2.1
```

## Usage

### Basic run (CPU)
```bash
python generate_preference_pairs.py --output pref_nonpref_dataset.csv
```

### With a GPU
```bash
python generate_preference_pairs.py --output pref_nonpref_dataset.csv --device 0
```

### Recommended for large datasets — with checkpointing
```bash
python generate_preference_pairs.py \
    --output pref_nonpref_dataset.csv \
    --checkpoint ckpt.csv \
    --device 0
```
Progress is saved to `ckpt.csv` every ~1,600 rows. If the job is interrupted, re-running the same command will automatically resume from where it left off.

### Process a specific slice of the dataset
```bash
python generate_preference_pairs.py \
    --start 1000 \
    --num-samples 200 \
    --output slice_1000_200.csv
```
Processes 200 rows beginning at row 1000 of the flattened Q&A dataset. The `source_index` column in the output records the original row number of each result, making it easy to merge slices back together later.

`--start` and `--num-samples` can be used independently:
- `--start 500` alone processes everything from row 500 to the end.
- `--num-samples 100` alone processes the first 100 rows (equivalent to `--limit 100`).

### Quick test on a small subset
```bash
python generate_preference_pairs.py --output test_out.csv --limit 50
```

### Custom pivot language (e.g. English instead of Spanish)
```bash
python generate_preference_pairs.py \
    --forward-model Helsinki-NLP/opus-mt-mk-en \
    --backward-model Helsinki-NLP/opus-mt-en-mk \
    --output pref_nonpref_en_pivot.csv
```

## All options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `LVSTCK/sft-mk` | HuggingFace dataset to load |
| `--output` | `pref_nonpref_dataset.csv` | Path for the final output CSV |
| `--forward-model` | `Helsinki-NLP/opus-mt-mk-es` | Model for source → pivot translation |
| `--backward-model` | `Helsinki-NLP/opus-mt-es-mk` | Model for pivot → source translation |
| `--source-col` | `preferred_answer` | DataFrame column to back-translate |
| `--new-col` | `non_preferred_answer` | Name of the output column |
| `--batch-size` | `16` | Number of texts per inference batch |
| `--device` | `-1` (CPU) | GPU index to use, or `-1` for CPU |
| `--start` | _(beginning)_ | 0-based index of the first row to process |
| `--num-samples` | _(all from start)_ | Number of rows to process from `--start` |
| `--limit` | _(all rows)_ | Process the first N rows — cannot be combined with `--start`/`--num-samples` |
| `--checkpoint` | _(disabled)_ | Path to save/resume a checkpoint CSV |

## Output format

| Column | Description |
|---|---|
| `source_index` | 0-based row number in the full flattened Q&A dataset |
| `user_question` | The original user message |
| `preferred_answer` | The original assistant answer from the dataset |
| `non_preferred_answer` | The back-translated (degraded) version of the answer |

`source_index` lets you trace any output row back to the original dataset and merge independently-produced slices by sorting on this column.

## Notes

- `--limit` and `--start`/`--num-samples` are mutually exclusive. Combining them raises an error.
- `--checkpoint` and `--output` should point to **different files**. The checkpoint is the partial/resumable save written during the run; `--output` is the final clean file written at the very end.
- Checkpoint resumption is slice-aware: only rows whose `source_index` matches the current slice are counted as already done, so a checkpoint from one slice cannot be accidentally applied to another.
- If a GPU is available but `--device` is not set, the script will log a reminder and run on CPU. Pass `--device 0` to use the first GPU.
- Batches that fail due to translation errors are skipped with an empty string and logged — the run continues rather than crashing.
- To redirect all logs to a file: `python generate_preference_pairs.py ... >> run.log 2>&1`
