from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import torch
from typing import Iterator, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "DGurgurov/aya-expanse-8b-mkd_cyrl"
DATASET_ID = "LVSTCK/sft-mk"
OUTPUT_FILE = "preference_pairs_raw.json"

# Cap how many rows to scan (set None for no cap).
MAX_SAMPLES = None
# If no prompt was extracted after this many rows, stop (avoids scanning the whole split on bad schema).
MAX_ROWS_SCAN_WITHOUT_ANY_PROMPT = 10_000

LOG_FILE = "aya_expanse_mkd_cyrl.log"
_logging_configured = False


class _FlushFileHandler(logging.FileHandler):
    """Flush after each record so `tail -f` on a remote host sees lines immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


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
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(console)
    logging.info("Logging to %s (append mode, flushed every line).", log_path)


setup_logging()


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


ensure_dependencies()

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


def iter_unique_questions(max_samples: Optional[int]) -> Iterator[str]:
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    seen: set[str] = set()
    yielded = 0
    scanned = 0
    logged_sample = False
    for row in ds:
        scanned += 1
        if max_samples is not None and yielded >= max_samples:
            break
        q = user_question_from_row(row)
        if not q:
            if not logged_sample:
                logging.warning("Could not parse a user prompt from a row; logging one sample row:")
                _log_unparsed_row_sample(row)
                logged_sample = True
            if yielded == 0 and scanned >= MAX_ROWS_SCAN_WITHOUT_ANY_PROMPT:
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
        yield q
        yielded += 1


def load_model(model_id: str):
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        elif dtype_env in ("fp16", "float16", ""):
            dtype = torch.float16
            if not dtype_env:
                logging.info(
                    "Using float16 on CUDA (default). Set AYA_TORCH_DTYPE=bf16 to try bfloat16."
                )
            else:
                logging.info("Using float16 (AYA_TORCH_DTYPE=%s).", dtype_env)
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
                "device_map={'':0}: using one GPU (index 0 in this process; %s visible). "
                "Set CUDA_VISIBLE_DEVICES to choose a physical card, or AYA_DEVICE_MAP=auto to use all %s.",
                ng,
                ng,
            )
        else:
            logging.info("device_map=auto: layers may be split across %s visible GPU(s).", ng)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        logging.warning("CUDA not available; loading on CPU.")
        dtype = torch.bfloat16
        device_map = "cpu"

    logging.info("Loading model (dtype=%s, device_map=%s)...", dtype, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

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


def _prepare_generate_inputs(model: torch.nn.Module, encoded: object) -> dict:
    """apply_chat_template may return a Tensor or a BatchEncoding; generate() needs a kwargs dict."""
    device = _input_device_for_model(model)
    if isinstance(encoded, torch.Tensor):
        return {"input_ids": encoded.to(device)}
    batch = dict(encoded)
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}


def generate_response(model, tokenizer, inputs: dict, temperature: float) -> str:
    prompt_len = inputs["input_ids"].shape[-1]
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=temperature,
    )
    return tokenizer.decode(
        output[0][prompt_len:],
        skip_special_tokens=True,
    )


def build_pairs(model, tokenizer, prompts: list) -> list:
    pairs = []

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = _prepare_generate_inputs(model, encoded)

        response_a = generate_response(model, tokenizer, inputs, temperature=0.3)
        response_b = generate_response(model, tokenizer, inputs, temperature=1.2)

        pairs.append(
            {
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
                "chosen": response_a,
                "rejected": response_b,
            }
        )

        if (i + 1) % 100 == 0:
            save_pairs(pairs, OUTPUT_FILE)
            logging.info("[%s/%s] Saved %s pairs to %s", i + 1, len(prompts), len(pairs), OUTPUT_FILE)


    return pairs


def save_pairs(pairs: list, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logging.info("Saved %s pairs to %s", len(pairs), output_file)


def main():
    logging.info("Loading prompts from %s...", DATASET_ID)
    prompts = list(iter_unique_questions(MAX_SAMPLES))
    if not prompts:
        raise SystemExit("No prompts extracted from the dataset; check schema / filters.")
    logging.info("Using %s unique questions.", len(prompts))

    model, tokenizer = load_model(MODEL_ID)
    pairs = build_pairs(model, tokenizer, prompts)
    save_pairs(pairs, OUTPUT_FILE)


if __name__ == "__main__":
    main()
