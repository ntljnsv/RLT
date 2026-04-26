# generate_preference_pairs.py

Generates preference pair datasets for RLHF/DPO training by back-translating answers from an existing HuggingFace SFT dataset through a pivot language. The original answer becomes the **preferred** response and the back-translated version becomes the **non-preferred** response.

## How it works

1. Loads a HuggingFace dataset and extracts all user/assistant Q&A pairs from the conversations
2. Translates each answer into a pivot language (e.g. Macedonian → Spanish)
3. Translates it back to the source language (Spanish → Macedonian)
4. Saves a CSV with three columns: `user_question`, `preferred_answer`, `non_preferred_answer`

Translation is done locally using [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP) models — no external API calls are made.

## Installation

```bash
pip install transformers==4.38.2 torch==2.10.0 datasets==4.0.0 pandas==2.2.2 tqdm==4.67.3
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
| `--limit` | _(all rows)_ | Only process the first N rows — useful for testing |
| `--checkpoint` | _(disabled)_ | Path to save/resume a checkpoint CSV |

## Output format

| Column | Description |
|---|---|
| `user_question` | The original user message |
| `preferred_answer` | The original assistant answer from the dataset |
| `non_preferred_answer` | The back-translated (degraded) version of the answer |

## Notes

- `--checkpoint` and `--output` should point to **different files**. The checkpoint is the partial/resumable save written during the run; `--output` is the final clean file written at the very end.
- If a GPU is available but `--device` is not set, the script will log a reminder and run on CPU. Pass `--device 0` to use the first GPU.
- Batches that fail due to translation errors are skipped with an empty string and logged — the run continues rather than crashing.
- To redirect all logs to a file: `python generate_preference_pairs.py ... >> run.log 2>&1`