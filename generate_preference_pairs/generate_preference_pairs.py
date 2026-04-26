"""
generate_preference_pairs.py
-----------------------------
Loads a HuggingFace SFT dataset, extracts all user/assistant Q&A pairs,
back-translates the answers through a pivot language (spanish) to produce a
"non-preferred" answer, and saves the result as a CSV.


Usage examples:
---------------
# Full run on GPU 0
    python generate_preference_pairs.py --output preference_pairs_dataset.csv --device 0

# Quick smoke-test on the first 50 rows (CPU)
    python generate_preference_pairs.py --output test_out.csv --limit 50

# Custom translation route and batch size with checkpoint csv
    python generate_preference_pairs.py \
        --forward-model  Helsinki-NLP/opus-mt-mk-en \
        --backward-model Helsinki-NLP/opus-mt-en-mk \
        --batch-size 8 \
        --checkpoint pref_pairs_checkpoint.csv \
        --output preference_pairs_dataset.csv

# All other parameters viewable though help in cli


Requirements:
-------------
pip install transformers==4.38.2 torch==2.10.0 datasets==4.0.0 pandas==2.2.2 tqdm==4.67.3
"""


import argparse
import logging
import os
import sys

try:
    import pandas as pd
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    print("Please install needed requirements first.")
    print("Use: pip install transformers==4.38.2 torch==2.10.0 datasets==4.0.0 pandas==2.2.2 tqdm==4.67.3")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def extract_all_qa_pairs(conversations: list) -> list[tuple[str, str]]:
    """
    Extract ALL consecutive user -> assistant turn pairs from a conversation.

    Pairs are matched sequentially: the assistant turn immediately following
    each user turn is treated as its answer.  User turns with no following
    assistant turn are skipped.
    """
    pairs = []
    i = 0
    while i < len(conversations):
        turn = conversations[i]
        if turn["role"] == "user":
            question = turn["content"]
            if i + 1 < len(conversations) and conversations[i + 1]["role"] == "assistant":
                answer = conversations[i + 1]["content"]
                pairs.append((question, answer))
                i += 2
                continue
        i += 1
    return pairs


def load_qa_dataframe(dataset_name: str, limit: int | None = None) -> pd.DataFrame:
    """ Load dataset and flatten all conversations into a Q&A DataFrame. """
    log.info("Loading dataset '%s' (split: train+test)...", dataset_name)
    ds = load_dataset(dataset_name, split="train+test")
    log.info("Total conversations loaded: %d", len(ds))

    records = []
    for row in ds:
        pairs = extract_all_qa_pairs(row["conversations"])
        for question, answer in pairs:
            records.append({"user_question": question, "preferred_answer": answer})

    df = pd.DataFrame(records, columns=["user_question", "preferred_answer"])
    log.info("Total Q&A pairs extracted: %d", len(df))

    if limit is not None:
        log.info("Limiting to first %d rows (--limit flag set).", limit)
        df = df.head(limit).copy()

    return df


def back_translate(
    df: pd.DataFrame,
    new_col_name: str,
    forward_model: str = "Helsinki-NLP/opus-mt-mk-es",
    backward_model: str = "Helsinki-NLP/opus-mt-es-mk",
    source_col: str = "preferred_answer",
    batch_size: int = 16,
    device: int = -1,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Back-translate *source_col* through a pivot language and store the result
    in *new_col_name*.

    Source -> pivot (forward_model) -> source (backward_model).

    Args:
        df: Input DataFrame.
        new_col_name: Column name for the back-translated output.
        forward_model: HuggingFace model ID for source→pivot.
        backward_model: HuggingFace model ID for pivot→source.
        source_col: Column whose text will be translated.
        batch_size: Texts per inference batch.
        device: -1 = CPU; 0, 1, … = GPU index.
        checkpoint_path: If given, partial results are saved here after every
                         100 batches so the job can be resumed after a crash.

    Returns:
        A copy of *df* with *new_col_name* appended.
    """
    if device >= 0 and torch.cuda.is_available():
        device_obj = torch.device(f"cuda:{device}")
    else:
        if device >= 0:
            log.warning("GPU requested but CUDA not available — falling back to CPU.")
        device_obj = torch.device("cpu")

    log.info("Using device: %s", device_obj)

    start_idx = 0
    results: list[str] = []

    if checkpoint_path and os.path.isfile(checkpoint_path):
        log.info("Checkpoint found at '%s', resuming...", checkpoint_path)
        ckpt_df = pd.read_csv(checkpoint_path)
        if new_col_name in ckpt_df.columns:
            already_done = ckpt_df[new_col_name].dropna().tolist()
            results = already_done
            start_idx = len(results)
            log.info("Resuming from row %d.", start_idx)

    log.info("Loading forward model : %s", forward_model)
    fwd_tokenizer = MarianTokenizer.from_pretrained(forward_model)
    fwd_model = MarianMTModel.from_pretrained(forward_model).to(device_obj)
    fwd_model.eval()

    log.info("Loading backward model: %s", backward_model)
    bwd_tokenizer = MarianTokenizer.from_pretrained(backward_model)
    bwd_model = MarianMTModel.from_pretrained(backward_model).to(device_obj)
    bwd_model.eval()

    texts = df[source_col].fillna("").tolist()
    cleaned_texts = [" ".join(str(t).split()) for t in texts]
    remaining = cleaned_texts[start_idx:]

    log.info(
        "Back-translating %d rows (%d remaining) in batches of %d...",
        len(cleaned_texts), len(remaining), batch_size,
    )

    with torch.no_grad():
        for batch_num, i in enumerate(
            tqdm(range(0, len(remaining), batch_size), desc="Translating", unit="batch")
        ):
            batch = remaining[i : i + batch_size]

            try:
                fwd_inputs = fwd_tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                ).to(device_obj)
                fwd_outputs = fwd_model.generate(**fwd_inputs, num_beams=4, early_stopping=True)
                pivot_texts = fwd_tokenizer.batch_decode(fwd_outputs, skip_special_tokens=True)

                bwd_inputs = bwd_tokenizer(
                    pivot_texts, return_tensors="pt", padding=True, truncation=True
                ).to(device_obj)
                bwd_outputs = bwd_model.generate(**bwd_inputs, num_beams=4, early_stopping=True)
                back_texts = bwd_tokenizer.batch_decode(bwd_outputs, skip_special_tokens=True)

                results.extend(back_texts)

            except Exception as exc:
                log.error("Error on batch %d: %s — skipping.", batch_num + 1, exc)
                results.extend([""] * len(batch))

            if device >= 0 and torch.cuda.is_available() and (batch_num + 1) % 10 == 0:
                torch.cuda.empty_cache()

            if checkpoint_path and (batch_num + 1) % 100 == 0:
                partial = df.copy()
                padded = results + [""] * (len(partial) - len(results))
                partial[new_col_name] = padded
                partial.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
                log.info("Checkpoint saved at row %d → '%s'", len(results), checkpoint_path)

    df = df.copy()
    df[new_col_name] = results
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preference pairs via back-translation."
    )
    parser.add_argument(
        "--dataset",
        default="LVSTCK/sft-mk",
        help="HuggingFace dataset name (default: LVSTCK/sft-mk)",
    )
    parser.add_argument(
        "--output",
        default="pref_nonpref_dataset.csv",
        help="Path for the output CSV (default: pref_nonpref_dataset.csv)",
    )
    parser.add_argument(
        "--forward-model",
        default="Helsinki-NLP/opus-mt-mk-es",
        help="HuggingFace model for source→pivot translation (default: mk→es)",
    )
    parser.add_argument(
        "--backward-model",
        default="Helsinki-NLP/opus-mt-es-mk",
        help="HuggingFace model for pivot→source translation (default: es→mk)",
    )
    parser.add_argument(
        "--source-col",
        default="preferred_answer",
        help="DataFrame column to back-translate (default: preferred_answer)",
    )
    parser.add_argument(
        "--new-col",
        default="non_preferred_answer",
        help="Name of the output column (default: non_preferred_answer)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="GPU index to use, or -1 for CPU (default: -1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows — useful for testing (default: all rows)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to save/resume a checkpoint CSV during long runs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == -1 and torch.cuda.is_available():
        log.info("CUDA is available. To use GPU, pass --device 0")

    df = load_qa_dataframe(args.dataset, limit=args.limit)

    log.info(
        "Starting back-translation: %s -> pivot -> %s",
        args.forward_model.split("/")[-1],
        args.backward_model.split("/")[-1],
    )
    df_result = back_translate(
        df=df,
        new_col_name=args.new_col,
        forward_model=args.forward_model,
        backward_model=args.backward_model,
        source_col=args.source_col,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )

    df_result.to_csv(args.output, index=False, encoding="utf-8-sig")
    log.info("Saved %d rows to '%s'.", len(df_result), args.output)
    log.info("Columns: %s", list(df_result.columns))