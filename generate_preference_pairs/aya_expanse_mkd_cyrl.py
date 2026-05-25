"""
aya_expanse_mkd_cyrl.py
-----------------------
Generate DPO-style preference pairs by sampling two temperatures from
DGurgurov/aya-expanse-8b-mkd_cyrl on unique user prompts from LVSTCK/sft-mk.

Usage examples:
---------------
# Full run on (default batch 128)
    python aya_expanse_mkd_cyrl.py --output preference_pairs_raw.jsonl

# Rows 500–699 among *unique* extracted prompts (200 prompts)
    python aya_expanse_mkd_cyrl.py --start 500 --num-samples 200 --output slice_500_200.jsonl

# Re-run appends to the same file; existing source_index values are skipped
    python aya_expanse_mkd_cyrl.py --output preference_pairs_raw.jsonl

# Convert JSONL → JSON array: jq -s '.' preference_pairs_raw.jsonl > preference_pairs_raw.json

# Quick smoke test
    python aya_expanse_mkd_cyrl.py --num-samples 5 --output test_pairs.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import torch
from typing import Iterator, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "DGurgurov/aya-expanse-8b-mkd_cyrl"
DATASET_ID = "LVSTCK/sft-mk"
DEFAULT_OUTPUT = "preference_pairs_raw.jsonl"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 128

# If no prompt was extracted after this many dataset rows, stop (bad schema).
MAX_ROWS_SCAN_WITHOUT_ANY_PROMPT = 10_000

LOG_FILE = "aya_expanse_mkd_cyrl.log"
_logging_configured = False


class _FlushHandler(logging.Handler):
    """Flush after each record so tail -f / Slurm logs update immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _FlushFileHandler(_FlushHandler, logging.FileHandler):
    pass


class _FlushStreamHandler(_FlushHandler, logging.StreamHandler):
    pass


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def setup_logging(log_path: str = LOG_FILE) -> None:
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = _FlushFileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = _FlushStreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(console)
    logging.info("Logging to %s.", log_path)
    _flush_logs()


def ensure_dependencies() -> None:
    """Install missing packages into the current interpreter only when needed."""
    required = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
    ]
    for module_name, pip_name in required:
        if importlib.util.find_spec(module_name) is None:
            logging.info("Missing '%s', installing %s via pip...", module_name, pip_name)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )


def _message_body(m: dict) -> Optional[str]:
    for key in ("content", "text", "value"):
        val = m.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _first_user_content_from_messages(messages: list) -> Optional[str]:
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        if role != "user":
            continue
        text = _message_body(m)
        if text:
            return text
    return None


def _is_chat_turn_list(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if not isinstance(first, dict):
        return False
    return "role" in first and ("content" in first or "text" in first or "value" in first)


def _log_unparsed_row_sample(row: object) -> None:
    if isinstance(row, dict):
        logging.warning("Sample row keys: %s", list(row.keys()))
        for k in list(row.keys())[:5]:
            v = row[k]
            preview = repr(v)
            if len(preview) > 400:
                preview = preview[:400] + "..."
            logging.warning("  [%s] (%s) %s", k, type(v).__name__, preview)
    else:
        preview = repr(row)
        if len(preview) > 500:
            preview = preview[:500] + "..."
        logging.warning("Sample row type=%s value=%s", type(row).__name__, preview)


def user_question_from_row(row: object) -> Optional[str]:
    """
    LVSTCK/sft-mk JSONL uses {"conversations": [...]}. Other shards may use messages,
    a bare list of turns, or Alpaca-style fields. Only the first user turn text is used.
    """
    if isinstance(row, list) and row:
        q = _first_user_content_from_messages(row)
        if q:
            return q

    if isinstance(row, dict):
        for key in ("conversations", "messages", "dialogue", "chat"):
            turns = row.get(key)
            if isinstance(turns, list) and turns:
                q = _first_user_content_from_messages(turns)
                if q:
                    return q

        for value in row.values():
            if _is_chat_turn_list(value):
                q = _first_user_content_from_messages(value)
                if q:
                    return q

        instruction = row.get("instruction")
        if instruction is not None and str(instruction).strip():
            inst = str(instruction).strip()
            inp = row.get("input")
            if inp is not None and str(inp).strip():
                return f"{inst}\n\n{str(inp).strip()}".strip()
            return inst

        for key in ("question", "query", "prompt"):
            val = row.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()

        inp_only = row.get("input")
        if inp_only is not None and str(inp_only).strip():
            return str(inp_only).strip()

    return None


def iter_unique_questions(
    start_index: int = 0,
    num_samples: Optional[int] = None,
    end_index: Optional[int] = None,
    max_scan: Optional[int] = None,
) -> Iterator[tuple[int, str]]:
    """
    Yield (source_index, prompt) for unique user questions in dataset order.

    source_index is the 0-based index among *deduplicated* prompts (case-insensitive).
    Select slice [start_index, end) where end = end_index or start_index + num_samples.
    """
    if start_index < 0:
        raise ValueError(f"--start must be >= 0, got {start_index}")
    if end_index is not None and num_samples is not None:
        raise ValueError("Pass only one of --end-index or --num-samples, not both.")
    if end_index is not None:
        if end_index <= start_index:
            raise ValueError(
                f"--end-index ({end_index}) must be greater than --start ({start_index})."
            )
        end = end_index
    elif num_samples is not None:
        if num_samples <= 0:
            raise ValueError(f"--num-samples must be > 0, got {num_samples}")
        end = start_index + num_samples
    else:
        end = None

    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    seen: set[str] = set()
    unique_count = 0
    scanned = 0
    logged_sample = False

    for row in ds:
        scanned += 1
        if max_scan is not None and scanned > max_scan:
            logging.warning("Reached --max-scan=%s dataset rows; stopping scan.", max_scan)
            break

        q = user_question_from_row(row)
        if not q:
            if not logged_sample:
                logging.warning("Could not parse a user prompt from a row; logging one sample row:")
                _log_unparsed_row_sample(row)
                logged_sample = True
            if unique_count == 0 and scanned >= MAX_ROWS_SCAN_WITHOUT_ANY_PROMPT:
                logging.warning(
                    "No extractable prompt in the first %s rows; stopping scan.",
                    MAX_ROWS_SCAN_WITHOUT_ANY_PROMPT,
                )
                break
            continue

        key = q.casefold()
        if key in seen:
            continue
        seen.add(key)

        if unique_count < start_index:
            unique_count += 1
            continue

        if end is not None and unique_count >= end:
            break

        yield unique_count, q
        unique_count += 1

    if end is not None and unique_count < start_index:
        logging.warning(
            "Dataset scan ended with only %s unique prompts (requested start=%s).",
            unique_count,
            start_index,
        )


def _resolve_attention_implementation(requested: str) -> str:
    """Pick the fastest attention backend available."""
    req = (requested or "auto").strip().lower()
    if req not in ("auto", "flash_attention_2", "sdpa", "eager"):
        raise ValueError(
            f"--attn must be auto, flash_attention_2, sdpa, or eager; got {requested!r}"
        )
    if req != "auto":
        return req

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logging.info(
            "flash-attn not installed; using sdpa. "
        )
        return "sdpa"
    return "flash_attention_2"


def load_model(
    model_id: str,
    *,
    attn_implementation: str = "auto",
) -> tuple[torch.nn.Module, AutoTokenizer]:
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding for batched generate(); slice new tokens at input_ids.shape[1] (HF LLM tutorial).
    tokenizer.padding_side = "left"

    cuda = torch.cuda.is_available()
    dtype_env = (os.environ.get("AYA_TORCH_DTYPE") or "").strip().lower()

    if cuda:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        logging.info(
            "CUDA: %s device(s), current %s (%s, %.1f GiB total)",
            torch.cuda.device_count(),
            idx,
            torch.cuda.get_device_name(idx),
            props.total_memory / (1024**3),
        )

        if dtype_env in ("bf16", "bfloat16"):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("AYA_TORCH_DTYPE=bf16 but this GPU reports no bf16 support.")
            dtype = torch.bfloat16
            logging.info("Using bfloat16 (AYA_TORCH_DTYPE=%s).", dtype_env)
        elif dtype_env in ("fp16", "float16"):
            dtype = torch.float16
            logging.info("Using float16 (AYA_TORCH_DTYPE=%s).", dtype_env)
        elif dtype_env == "":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logging.info(
                "Using %s on CUDA. "
                "Set AYA_TORCH_DTYPE=fp16 to override.",
                "bfloat16" if dtype == torch.bfloat16 else "float16",
            )
        else:
            raise RuntimeError(
                "AYA_TORCH_DTYPE must be fp16, bf16, or unset; got %r" % (dtype_env,)
            )

        map_env = (os.environ.get("AYA_DEVICE_MAP") or "single").strip().lower()
        if map_env in ("single", "one", "first", "1"):
            device_map = {"": 0}
        elif map_env == "auto":
            device_map = "auto"
        else:
            raise RuntimeError(
                "AYA_DEVICE_MAP must be 'single' (default) or 'auto'; got %r" % (map_env,)
            )
        ng = torch.cuda.device_count()
        if isinstance(device_map, dict):
            logging.info(
                "device_map={'':0}: one GPU (index 0 in this process; %s visible). "
                ng,
            )
        else:
            logging.info("device_map=auto: layers may be split across %s visible GPU(s).", ng)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    else:
        logging.warning("CUDA not available; loading on CPU.")
        dtype = torch.bfloat16
        device_map = "cpu"

    attn = _resolve_attention_implementation(attn_implementation)
    logging.info(
        "Loading model (dtype=%s, device_map=%s, attn_implementation=%s)...",
        dtype,
        device_map,
        attn,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        if attn == "flash_attention_2":
            logging.warning(
                "flash_attention_2 load failed (%s); falling back to sdpa.", exc
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        else:
            raise

    model.eval()
    return model, tokenizer


def _input_device_for_model(model: torch.nn.Module) -> torch.device:
    """Device for input_ids (avoid meta/offload parameter devices)."""
    emb = getattr(model, "get_input_embeddings", lambda: None)()
    if emb is not None and emb.weight.device.type != "meta":
        return emb.weight.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _pad_token_id(tokenizer) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id for batching.")
    return pad_id


def _encoding_to_batch_dict(tokenizer, encoded: object) -> dict:
    """
    Normalize apply_chat_template output to {input_ids, attention_mask}.

    Batched calls often return a bare (batch, seq) Tensor
    """
    pad_id = _pad_token_id(tokenizer)

    if isinstance(encoded, torch.Tensor):
        input_ids = encoded if encoded.dim() > 1 else encoded.unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != pad_id).long(),
        }

    if isinstance(encoded, list):
        rows: list[torch.Tensor] = []
        for item in encoded:
            if isinstance(item, torch.Tensor):
                rows.append(item.squeeze(0) if item.dim() > 1 else item)
            elif isinstance(item, dict) and "input_ids" in item:
                rows.append(item["input_ids"].squeeze(0))
            else:
                rows.append(torch.tensor(item, dtype=torch.long))
        padded = tokenizer.pad(
            {"input_ids": rows},
            padding=True,
            return_tensors="pt",
        )
        return dict(padded)

    if hasattr(encoded, "keys"):
        batch = dict(encoded)
        if "attention_mask" not in batch and "input_ids" in batch:
            input_ids = batch["input_ids"]
            batch["attention_mask"] = (input_ids != pad_id).long()
        return batch

    raise TypeError(
        f"Unsupported apply_chat_template return type: {type(encoded).__name__}"
    )


def _prepare_generate_inputs(model: torch.nn.Module, batch: dict) -> dict:
    """Move batch tensors to the model input device for generate()."""
    device = _input_device_for_model(model)
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}


_AYA_TEMPLATE_MARKERS = (
    "<|START_OF_TURN_TOKEN|>",
    "<|END_OF_TURN_TOKEN|>",
    "<|CHATBOT_TOKEN|>",
    "<|USER_TOKEN|>",
    "<|SYSTEM_TOKEN|>",
)
_AYA_CHATBOT_MARKER = "<|CHATBOT_TOKEN|>"


def _eos_token_ids(tokenizer) -> list[int]:
    """Token ids that end a generated turn (model stops early when these are sampled)."""
    ids: list[int] = []
    seen: set[int] = set()
    unk = getattr(tokenizer, "unk_token_id", None)

    candidates = [tokenizer.eos_token, "<|END_OF_TURN_TOKEN|>"]
    if tokenizer.eos_token_id is not None:
        candidates.insert(0, tokenizer.eos_token_id)

    for item in candidates:
        if item is None:
            continue
        if isinstance(item, int):
            tid = item
        else:
            tid = tokenizer.convert_tokens_to_ids(item)
        if isinstance(tid, int) and tid >= 0 and tid != unk and tid not in seen:
            ids.append(tid)
            seen.add(tid)
    if not ids and tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return ids


def _strip_chat_template_markers(text: str) -> str:
    for marker in _AYA_TEMPLATE_MARKERS:
        text = text.replace(marker, "")
    return " ".join(text.split()).strip()


def _trim_token_ids_at_eos(token_ids: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    if not eos_ids:
        return token_ids
    eos_set = set(eos_ids)
    for idx in range(token_ids.shape[0]):
        if token_ids[idx].item() in eos_set:
            return token_ids[:idx]
    return token_ids


def _remove_prompt_echo(text: str, user_prompt: str) -> str:
    """Drop accidental prefix overlap with the user message (BPE boundary artifacts)."""
    text = text.strip()
    prompt = user_prompt.strip()
    if not text or not prompt:
        return text
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    # Last token(s) of the prompt can appear at the start of the decoded completion.
    for n in range(min(len(prompt), 120), 4, -1):
        suffix = prompt[-n:]
        if text.startswith(suffix):
            return text[n:].strip()
    return text


def _decode_new_tokens(
    tokenizer,
    new_token_ids: torch.Tensor,
    *,
    user_prompt: str = "",
) -> str:
    eos_ids = _eos_token_ids(tokenizer)
    new_token_ids = _trim_token_ids_at_eos(new_token_ids, eos_ids)
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    text = _strip_chat_template_markers(text)
    return _remove_prompt_echo(text, user_prompt)


def encode_prompt_batch(tokenizer, prompts: list[str]) -> dict:
    """Left-padded batch encoding (see HF transformers batched generation docs)."""
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    encoded = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    )
    return _encoding_to_batch_dict(tokenizer, encoded)


def _generate_from_inputs(
    model,
    tokenizer,
    inputs: dict,
    prompts: list[str],
    *,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """
    Batched generate() with left padding.

    New tokens start at input_ids.shape[1] for every row (HF batched LLM pattern).
    Decode per row with EOS trim + prompt-echo removal.
    """
    if not prompts:
        return []

    prompt_width = inputs["input_ids"].shape[1]
    eos_ids = _eos_token_ids(tokenizer)
    eos_arg: int | list[int] = eos_ids[0] if len(eos_ids) == 1 else eos_ids

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_arg,
        )

    new_tokens = output[:, prompt_width:]
    return [
        _decode_new_tokens(tokenizer, new_tokens[i], user_prompt=prompts[i])
        for i in range(new_tokens.shape[0])
    ]


def generate_responses_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """Encode a batch of prompts and run one batched generate() call."""
    if not prompts:
        return []
    batch = encode_prompt_batch(tokenizer, prompts)
    inputs = _prepare_generate_inputs(model, batch)
    return _generate_from_inputs(
        model,
        tokenizer,
        inputs,
        prompts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def load_done_source_indices(output_file: str) -> set[int]:
    """Read source_index values already present in a JSONL output file."""
    if not os.path.isfile(output_file):
        return set()

    done: set[int] = set()
    with open(output_file, encoding="utf-8") as f:
        start = f.read(1)
        if not start:
            return done
        f.seek(0)
        if start == "[":
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logging.warning("Could not parse %s as JSON array; ignoring.", output_file)
                return done
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "source_index" in item:
                        done.add(int(item["source_index"]))
            return done

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "source_index" in item:
                done.add(int(item["source_index"]))
    return done


def append_pairs(output_file: str, batch_pairs: list[dict]) -> None:
    """Append one JSON object per line (JSONL)."""
    if not batch_pairs:
        return
    with open(output_file, "a", encoding="utf-8") as f:
        for pair in batch_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_pairs(
    model,
    tokenizer,
    prompts: list[tuple[int, str]],
    *,
    output_file: str,
    batch_size: int,
    max_new_tokens: int,
    temp_low: float,
    temp_high: float,
) -> int:
    done_indices = load_done_source_indices(output_file)
    if done_indices:
        logging.info(
            "Skipping %s source_index values already in %s",
            len(done_indices),
            output_file,
        )

    pending = [
        (source_index, prompt)
        for source_index, prompt in prompts
        if source_index not in done_indices
    ]
    total_pending = len(pending)
    if not pending:
        return len(done_indices)

    logging.info(
        "Generating %s prompts in batches of %s (2 passes: temp %.2f / %.2f)",
        total_pending,
        batch_size,
        temp_low,
        temp_high,
    )

    total_in_file = len(done_indices)
    completed_in_slice = 0

    for batch_start in range(0, total_pending, batch_size):
        chunk = pending[batch_start : batch_start + batch_size]
        source_indices = [idx for idx, _ in chunk]
        batch_prompts = [prompt for _, prompt in chunk]

        t_batch = time.perf_counter()
        batch = encode_prompt_batch(tokenizer, batch_prompts)
        inputs = _prepare_generate_inputs(model, batch)

        responses_a = _generate_from_inputs(
            model,
            tokenizer,
            inputs,
            batch_prompts,
            temperature=temp_low,
            max_new_tokens=max_new_tokens,
        )
        responses_b = _generate_from_inputs(
            model,
            tokenizer,
            inputs,
            batch_prompts,
            temperature=temp_high,
            max_new_tokens=max_new_tokens,
        )
        batch_secs = time.perf_counter() - t_batch
        per_pair_secs = batch_secs / len(chunk)

        batch_pairs = [
            {
                "source_index": source_index,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            for source_index, prompt, chosen, rejected in zip(
                source_indices, batch_prompts, responses_a, responses_b
            )
        ]
        append_pairs(output_file, batch_pairs)

        completed_in_slice += len(batch_pairs)
        total_in_file += len(batch_pairs)
        last_index = source_indices[-1]

        logging.info(
            "Batch done: source_index %s..%s | %s pairs in %.1fs (%.2fs/pair) | "
            "slice %s/%s | %s total → %s",
            source_indices[0],
            last_index,
            len(chunk),
            batch_secs,
            per_pair_secs,
            completed_in_slice,
            total_pending,
            total_in_file,
            output_file,
        )
        _flush_logs()

    return total_in_file


def _resolve_end_index(args: argparse.Namespace) -> Optional[int]:
    if args.end_index is not None:
        return args.end_index
    if args.num_samples is not None:
        return args.start + args.num_samples
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preference pairs with aya-expanse-8b-mkd_cyrl."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            f"JSONL output path — one pair per line, appended each batch "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based index of the first *unique* prompt to generate (default: 0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of unique prompts to generate from --start (default: all from --start)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help=(
            "Exclusive end index among unique prompts (e.g. --start 100 --end-index 300 "
            "generates source_index 100..299). Cannot combine with --num-samples."
        ),
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Stop scanning the HF dataset after this many raw rows (debug / smoke tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Prompts per batched generate() call (default: {DEFAULT_BATCH_SIZE}). "
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            f"Upper bound on new tokens per response (default: {DEFAULT_MAX_NEW_TOKENS}). "
        ),
    )
    parser.add_argument(
        "--temp-low",
        type=float,
        default=0.3,
        help="Temperature for chosen (default: 0.3)",
    )
    parser.add_argument(
        "--temp-high",
        type=float,
        default=1.2,
        help="Temperature for rejected (default: 1.2)",
    )
    parser.add_argument(
        "--attn",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: auto → flash-attn2 if installed else sdpa)",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_ID,
        help=f"HuggingFace dataset id (default: {DATASET_ID})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"Model id (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--log-file",
        default=LOG_FILE,
        help=f"Log file path (default: {LOG_FILE})",
    )

    args = parser.parse_args()
    if args.num_samples is not None and args.end_index is not None:
        parser.error("--num-samples and --end-index are mutually exclusive.")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0.")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    ensure_dependencies()

    global DATASET_ID, MODEL_ID
    DATASET_ID = args.dataset
    MODEL_ID = args.model

    end_index = _resolve_end_index(args)

    logging.info(
        "Slice: source_index [%s, %s), dataset=%s",
        args.start,
        end_index if end_index is not None else "∞",
        DATASET_ID,
    )
    batch_size = args.batch_size
    logging.info(
        "Output: %s (JSONL append) | batch_size=%s | max_new_tokens=%s",
        args.output,
        batch_size,
        args.max_new_tokens,
    )

    prompts = list(
        iter_unique_questions(
            start_index=args.start,
            num_samples=args.num_samples,
            end_index=args.end_index,
            max_scan=args.max_scan,
        )
    )
    if not prompts:
        raise SystemExit(
            "No prompts in this slice; check --start/--end-index/--num-samples or dataset."
        )
    logging.info(
        "Queued %s prompts (source_index %s..%s).",
        len(prompts),
        prompts[0][0],
        prompts[-1][0],
    )

    model, tokenizer = load_model(MODEL_ID, attn_implementation=args.attn)
    total_pairs = build_pairs(
        model,
        tokenizer,
        prompts,
        output_file=args.output,
        batch_size=batch_size,
        max_new_tokens=args.max_new_tokens,
        temp_low=args.temp_low,
        temp_high=args.temp_high,
    )
    logging.info("Done: %s pairs in %s", total_pairs, args.output)


if __name__ == "__main__":
    main()
