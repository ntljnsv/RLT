"""
Usage:
  python train.py --dataset_path data/preference_dataset.csv
  python train.py --dataset_path data/preference_dataset.csv --load_in_4bit
  python train.py --dataset_path data/pref.csv --col_prompt q --col_chosen good --col_rejected bad
  python train.py --dataset_path data/preference_dataset.csv --no_lora
  python train.py --help
"""

import argparse
import csv
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl.experimental.orpo import ORPOTrainer, ORPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="ORPO fine-tuning for VezilkaLLM-Instruct")

    parser.add_argument("--model_id", type=str, default="finki-ukim/VezilkaLLM-Instruct",
                        help="HuggingFace model ID or local path of the base model")
    parser.add_argument("--output_dir", type=str, default="outputs/vezilka-orpo",
                        help="Directory where checkpoints and the final model will be saved")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the CSV preference dataset (prompt, chosen, rejected columns)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of data held out for validation (0 to disable)")

    parser.add_argument("--col_prompt", type=str, default="prompt",
                        help="CSV column name for the prompt")
    parser.add_argument("--col_chosen", type=str, default="chosen",
                        help="CSV column name for the chosen (preferred) response")
    parser.add_argument("--col_rejected", type=str, default="rejected",
                        help="CSV column name for the rejected response")

    parser.add_argument("--lambda_orpo", type=float, default=0.1,
                        help="ORPO odds-ratio loss coefficient (lambda). "
                             "Controls the strength of the preference signal relative to SFT. "
                             "Typical range: 0.05 – 0.5.")

    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device training (and eval) batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=8e-6,
                        help="Peak learning rate. ORPO is typically trained at a higher LR than DPO "
                             "because there is no KL penalty from a reference model.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length (prompt + response). Longer sequences are truncated.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps used for linear LR warm-up")

    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA scaling factor (alpha)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout probability applied to LoRA layers")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA and do full fine-tuning (requires significantly more VRAM)")

    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load the model in 4-bit precision (QLoRA). "
                             "Recommended for GPUs with less than 24 GB VRAM.")

    return parser.parse_args()


def load_preference_dataset(dataset_path: str, tokenizer, val_split: float, col_prompt: str = "prompt",
                            col_chosen: str = "chosen", col_rejected: str = "rejected"):
    """
    Loads a CSV preference dataset and applies the model's chat template to
    each prompt so the model sees the same format it was trained on during SFT.

    Returns (train_dataset, eval_dataset | None).
    """

    records = []
    with open(dataset_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Loaded {len(records)} preference pairs from {dataset_path}")
    print(f"  Columns — prompt: '{col_prompt}', chosen: '{col_chosen}', rejected: '{col_rejected}'")

    if records:
        actual_cols = set(records[0].keys())
        for name, col in [("prompt", col_prompt), ("chosen", col_chosen), ("rejected", col_rejected)]:
            if col not in actual_cols:
                raise ValueError(
                    f"Column '{col}' (--col_{name}) not found in CSV. "
                    f"Available columns: {sorted(actual_cols)}"
                )

    def format_record(record):
        prompt_messages = [{"role": "user", "content": record[col_prompt]}]
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": record[col_chosen], "rejected": record[col_rejected]}

    formatted = [format_record(r) for r in records]
    dataset = Dataset.from_list(formatted)

    if val_split > 0:
        split = dataset.train_test_split(test_size=val_split, seed=42)
        return split["train"], split["test"]
    else:
        return dataset, None


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading model in 4-bit (QLoRA mode)")
    else:
        print("Loading model in bfloat16")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.load_in_4bit else None,
        device_map="auto",
        attn_implementation="eager",
    )

    return model, tokenizer


def get_lora_config(args):
    if args.no_lora:
        return None

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def get_orpo_config(args, has_val: bool):
    return ORPOConfig(
        beta=args.lambda_orpo,
        max_length=args.max_length,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        bf16=True,
        gradient_checkpointing=True,

        output_dir=args.output_dir,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        save_total_limit=2,

        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"  VezilkaLLM-Instruct ORPO Training")
    print(f"{'=' * 50}")
    print(f"  Model:          {args.model_id}")
    print(f"  Lambda (ORPO):  {args.lambda_orpo}")
    print(f"  LR:             {args.learning_rate}")
    print(f"  LoRA:           {'disabled' if args.no_lora else f'r={args.lora_r}, alpha={args.lora_alpha}'}")
    print(f"  4-bit:          {args.load_in_4bit}")
    print(f"  Max length:     {args.max_length} (prompt: {args.max_prompt_length})")
    print(f"  Output:         {args.output_dir}")
    print(f"{'=' * 50}\n")

    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset, eval_dataset = load_preference_dataset(
        args.dataset_path, tokenizer, args.val_split,
        col_prompt=args.col_prompt,
        col_chosen=args.col_chosen,
        col_rejected=args.col_rejected,
    )
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples:  {len(eval_dataset)}")

    peft_config = get_lora_config(args)

    orpo_config = get_orpo_config(args, has_val=eval_dataset is not None)

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    print("Starting ORPO training...\n")
    trainer.train()

    final_path = f"{args.output_dir}/final"
    print(f"\nSaving final model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("Done.")


if __name__ == "__main__":
    main()
