"""
Usage:
  python train_dpo.py --dataset_path data/preference_dataset.csv
  python train_dpo.py --dataset_path data/preference_dataset.csv --loss_type ipo
  python train_dpo.py --help
"""

import argparse
import csv
import os
import torch
from datasets import Dataset
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig


def parse_args():
  parser = argparse.ArgumentParser(description="DPO fine-tuning for VezilkaLLM-Instruct")

  # Model
  parser.add_argument("--model_id", type=str, default="finki-ukim/VezilkaLLM-Instruct")
  parser.add_argument("--output_dir", type=str, default="outputs/vezilka-dpo")

  # Dataset
  parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to your .csv preference dataset with columns: prompt, chosen, rejected")
  parser.add_argument("--val_split", type=float, default=0.05,
                      help="Fraction of data to use for validation")

  # DPO
  parser.add_argument("--beta", type=float, default=0.1,
                      help="KL penalty coefficient. Higher = stay closer to reference model.")
  parser.add_argument("--loss_type", type=str, default="dpo",
                      choices=["dpo", "ipo", "sigmoid"],
                      help="DPO loss variant. Use 'ipo' if you see training instability.")

  # Training
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--grad_accum", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=5e-7)
  parser.add_argument("--max_length", type=int, default=1024)
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--warmup_ratio", type=float, default=0.1)

  # LoRA
  parser.add_argument("--lora_r", type=int, default=16)
  parser.add_argument("--lora_alpha", type=int, default=32)
  parser.add_argument("--lora_dropout", type=float, default=0.05)
  parser.add_argument("--no_lora", action="store_true",
                      help="Disable LoRA and do full fine-tuning (requires more VRAM)")

  # Quantization
  parser.add_argument("--load_in_4bit", action="store_true",
                      help="Load model in 4-bit (QLoRA). Recommended for GPUs < 24GB.")

  return parser.parse_args()


def load_preference_dataset(dataset_path: str, tokenizer, val_split: float):
  """
  Loads a CSV file with columns: prompt, chosen, rejected

  Applies the Gemma 3 chat template to the prompt so the model sees
  the same format it was trained on during SFT.
  """
  records = []
  with open(dataset_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      records.append(row)

  print(f"Loaded {len(records)} preference pairs from {dataset_path}")

  def format_record(record):
    prompt_messages = [{"role": "user", "content": record["prompt"]}]
    prompt = tokenizer.apply_chat_template(
      prompt_messages,
      tokenize=False,
      add_generation_prompt=True,
    )
    return {
      "prompt": prompt,
      "chosen": record["chosen"],
      "rejected": record["rejected"],
    }

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


def get_dpo_config(args, has_val: bool):
  return DPOConfig(
    # Core DPO params
    beta=args.beta,
    loss_type=args.loss_type,
    max_length=args.max_length,
    max_prompt_length=args.max_prompt_length,

    # Training
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type="cosine",

    # Precision
    bf16=True,
    gradient_checkpointing=True,

    # Logging & saving
    output_dir=args.output_dir,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch" if has_val else "no",
    save_total_limit=2,

    # Misc
    remove_unused_columns=False,
    report_to="none",
    seed=42,
  )


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  print(f"\n{'='*50}")
  print(f"  VezilkaLLM-Instruct DPO Training")
  print(f"{'='*50}")
  print(f"  Model:       {args.model_id}")
  print(f"  Loss type:   {args.loss_type}")
  print(f"  Beta:        {args.beta}")
  print(f"  LR:          {args.learning_rate}")
  print(f"  LoRA:        {'disabled' if args.no_lora else f'r={args.lora_r}, alpha={args.lora_alpha}'}")
  print(f"  4-bit:       {args.load_in_4bit}")
  print(f"  Output:      {args.output_dir}")
  print(f"{'='*50}\n")

  # Load model & tokenizer
  model, tokenizer = load_model_and_tokenizer(args)

  # Load dataset
  train_dataset, eval_dataset = load_preference_dataset(
    args.dataset_path, tokenizer, args.val_split
  )
  print(f"Train samples: {len(train_dataset)}")
  if eval_dataset:
    print(f"Eval samples:  {len(eval_dataset)}")

  # LoRA config
  peft_config = get_lora_config(args)

  # DPO config
  dpo_config = get_dpo_config(args, has_val=eval_dataset is not None)

  # DPOTrainer
  trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
  )

  print("Starting DPO training...\n")
  trainer.train()

  print(f"\nSaving final model to {args.output_dir}/final")
  trainer.save_model(f"{args.output_dir}/final")
  tokenizer.save_pretrained(f"{args.output_dir}/final")
  print("Done.")


if __name__ == "__main__":
  main()
