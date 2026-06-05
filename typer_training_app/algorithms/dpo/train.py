import os
import random

import torch
from transformers import EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig as TRLDPOConfig
from peft import LoraConfig, TaskType, get_peft_model

from shared.model_loader import load_model_and_tokenizer
from shared.dataset_loader import load_dataset_any
from algorithms.dpo.prepare_dataset import format_dpo_dataset
from shared.validation import validate_preference_dataset


def get_lora_config(cfg):
    if cfg.no_lora:
        return None

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )


def detect_precision():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    return False


def split_train_eval(records, eval_fraction: float, seed: int):
    if eval_fraction <= 0 or len(records) < 10:
        return records, None

    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    n_eval = max(1, int(len(shuffled) * eval_fraction))
    eval_records = shuffled[:n_eval]
    train_records = shuffled[n_eval:]
    return train_records, eval_records


def build_trl_config(cfg, use_bf16: bool, has_eval: bool) -> TRLDPOConfig:
    kwargs = dict(
        beta=cfg.beta,
        max_length=cfg.max_length,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        output_dir=cfg.output_dir,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        remove_unused_columns=False,
        report_to=cfg.report_to if cfg.report_to != "none" else "none",
        seed=cfg.seed,
        gradient_checkpointing=cfg.gradient_checkpointing,
        precompute_ref_log_probs=cfg.precompute_ref_log_probs,
        precompute_ref_batch_size=cfg.precompute_ref_batch_size,
    )

    if cfg.max_prompt_length is not None:
        kwargs["max_prompt_length"] = cfg.max_prompt_length
    if cfg.max_completion_length is not None:
        kwargs["max_completion_length"] = cfg.max_completion_length

    if cfg.save_steps > 0:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = cfg.save_steps
    else:
        kwargs["save_strategy"] = "epoch"

    if has_eval:
        kwargs["eval_strategy"] = cfg.eval_strategy
        kwargs["per_device_eval_batch_size"] = cfg.eval_batch_size
        if cfg.early_stopping:
            kwargs["load_best_model_at_end"] = True
            kwargs["metric_for_best_model"] = cfg.metric_for_best_model
            kwargs["greater_is_better"] = False
    else:
        kwargs["eval_strategy"] = "no"

    return TRLDPOConfig(**kwargs)


def build_callbacks(cfg, has_eval: bool) -> list:
    if not cfg.early_stopping:
        return []
    if not has_eval:
        print(
            "Early stopping disabled: need an eval set "
            "(use --eval-fraction > 0, default 0.05)."
        )
        return []

    print(
        f"Early stopping enabled: patience={cfg.early_stopping_patience}, "
        f"metric={cfg.metric_for_best_model}, "
        f"threshold={cfg.early_stopping_threshold}"
    )
    return [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_threshold=cfg.early_stopping_threshold,
        )
    ]


def apply_lora(model, cfg):
    """Attach LoRA before DPOTrainer to avoid fp32 adapter cast OOM spike."""
    peft_config = get_lora_config(cfg)
    if peft_config is None:
        return model, None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Applying LoRA (autocast_adapter_dtype=False)...")
    model = get_peft_model(
        model,
        peft_config,
        autocast_adapter_dtype=False,
    )
    model.print_trainable_parameters()
    return model, None


def run_dpo_training(cfg, dataset=None):
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.output_dir, "run_config.json"))

    if dataset is None:
        print(f"Loading dataset from {cfg.dataset_path} ...")
        dataset = load_dataset_any(cfg.dataset_path, max_samples=cfg.max_samples)

    print("Validating dataset...")
    records = validate_preference_dataset(dataset)
    print(f"Dataset size: {len(records)}")

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.cuda_device)

    print(f"Loading model on cuda:{cfg.cuda_device}")
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_id,
        cfg.load_in_4bit,
        gradient_checkpointing=cfg.gradient_checkpointing,
        device_index=cfg.cuda_device,
        attn_implementation=cfg.attn_implementation,
    )

    print("CUDA AVAILABLE:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("MODEL DEVICE:", next(model.parameters()).device)
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            print(
                f"GPU {i} VRAM: {used / 1e9:.1f} GB used, "
                f"{free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total"
            )

    train_records, eval_records = split_train_eval(
        records, cfg.eval_fraction, cfg.seed
    )
    n_eval = len(eval_records) if eval_records else 0
    print(f"Train: {len(train_records)}, eval: {n_eval}")

    print("Formatting dataset...")
    train_dataset = format_dpo_dataset(train_records, tokenizer)
    eval_dataset = (
        format_dpo_dataset(eval_records, tokenizer) if eval_records else None
    )

    model, peft_config = apply_lora(model, cfg)
    use_bf16 = detect_precision()
    print(
        f"Using bf16={use_bf16}, fp16={not use_bf16}, "
        f"4bit={cfg.load_in_4bit}, batch={cfg.batch_size}, "
        f"grad_accum={cfg.grad_accum}, "
        f"gradient_checkpointing={cfg.gradient_checkpointing}, "
        f"attn={cfg.attn_implementation}, "
        f"precompute_ref_log_probs={cfg.precompute_ref_log_probs}, "
        f"precompute_ref_batch_size={cfg.precompute_ref_batch_size}"
    )

    has_eval = eval_dataset is not None
    dpo_config = build_trl_config(cfg, use_bf16, has_eval=has_eval)
    callbacks = build_callbacks(cfg, has_eval=has_eval)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    print("Training...")
    train_kwargs = {}
    if cfg.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = cfg.resume_from_checkpoint
    trainer.train(**train_kwargs)

    print("Saving final adapter...")
    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    if cfg.early_stopping and has_eval:
        print(
            f"Best checkpoint (by {cfg.metric_for_best_model}) was reloaded before save."
        )
    print(f"Done. Checkpoints in {cfg.output_dir}, final adapter in {final_path}")