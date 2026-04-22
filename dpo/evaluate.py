"""
Usage:
  # Side-by-side inference comparison
  python evaluate.py --mode compare --model_path outputs/vezilka-dpo/final --prompts_path data/test_prompts.txt

  # Win-rate on a held-out preference set (uses the model's own log-probs as implicit reward)
  python evaluate.py --mode winrate --model_path outputs/vezilka-dpo/final --dataset_path data/test_preferences.csv
"""

import argparse
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np


BASE_MODEL_ID = "finki-ukim/VezilkaLLM-Instruct"


def load_model(model_path: str, is_peft: bool = True):
  """
  Loads either:
    - A PEFT/LoRA adapter (from DPO training with LoRA)
    - A full merged model (from merge_and_save.py)
  """
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  if is_peft:
    base = AutoModelForCausalLM.from_pretrained(
      BASE_MODEL_ID,
      torch_dtype=torch.bfloat16,
      device_map="auto",
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
  else:
    model = AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
      device_map="auto",
    )
    model.eval()

  return model, tokenizer


def load_base_model():
  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
  )
  model.eval()
  return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
  messages = [{"role": "user", "content": prompt}]
  input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
  ).to(model.device)

  with torch.no_grad():
    output = model.generate(
      input_ids,
      max_new_tokens=max_new_tokens,
      do_sample=True,
      temperature=0.7,
      top_p=0.9,
      pad_token_id=tokenizer.eos_token_id,
    )

  new_tokens = output[0][input_ids.shape[-1]:]
  return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compare_models(dpo_model_path: str, prompts_path: str, is_peft: bool):
  print("Loading base model...")
  base_model, base_tokenizer = load_base_model()

  print("Loading DPO model...")
  dpo_model, dpo_tokenizer = load_model(dpo_model_path, is_peft=is_peft)

  with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = [l.strip() for l in f if l.strip()]

  print(f"\nComparing on {len(prompts)} prompts\n")

  for i, prompt in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(prompts)}] PROMPT:")
    print(f"  {prompt}")
    print(f"\n  BASE (VezilkaLLM-Instruct):")
    base_resp = generate_response(base_model, base_tokenizer, prompt)
    print(f"  {base_resp}")
    print(f"\n  DPO ({dpo_model_path}):")
    dpo_resp = generate_response(dpo_model, dpo_tokenizer, prompt)
    print(f"  {dpo_resp}")


def compute_sequence_logprob(model, tokenizer, prompt: str, response: str) -> float:
  """
  Computes the average log-probability of `response` given `prompt`
  under `model`. Used as an implicit reward signal.
  """
  messages = [{"role": "user", "content": prompt}]
  prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
  ).to(model.device)

  response_ids = tokenizer(
    response,
    return_tensors="pt",
    add_special_tokens=False,
  ).input_ids.to(model.device)

  full_ids = torch.cat([prompt_ids, response_ids], dim=1)
  labels = torch.full_like(full_ids, -100)
  labels[:, prompt_ids.shape[1]:] = response_ids  # only score the response tokens

  with torch.no_grad():
    outputs = model(input_ids=full_ids, labels=labels)
    # outputs.loss is the mean NLL over response tokens
    logprob = -outputs.loss.item()

  return logprob


def compute_winrate(dpo_model_path: str, dataset_path: str, is_peft: bool):
  """
  For each (prompt, chosen, rejected) triplet:
  - Compute log P(chosen | prompt) and log P(rejected | prompt) under the DPO model
  - A "win" is when the model assigns higher probability to chosen than rejected
  """
  print("Loading DPO model for win-rate evaluation...")
  dpo_model, dpo_tokenizer = load_model(dpo_model_path, is_peft=is_peft)

  with open(dataset_path, "r", encoding="utf-8", newline="") as f:
    records = list(csv.DictReader(f))

  print(f"Evaluating on {len(records)} preference pairs...\n")

  wins = 0
  margins = []

  for i, record in enumerate(records):
    lp_chosen = compute_sequence_logprob(
      dpo_model, dpo_tokenizer, record["prompt"], record["chosen"]
    )
    lp_rejected = compute_sequence_logprob(
      dpo_model, dpo_tokenizer, record["prompt"], record["rejected"]
    )
    margin = lp_chosen - lp_rejected
    margins.append(margin)
    if margin > 0:
      wins += 1

    if (i + 1) % 20 == 0:
      print(f"  Processed {i+1}/{len(records)} | Running win-rate: {wins/(i+1):.2%}")

  win_rate = wins / len(records)
  print(f"\n{'='*50}")
  print(f"  Win-rate:       {win_rate:.2%}  ({wins}/{len(records)})")
  print(f"  Mean margin:    {np.mean(margins):.4f}")
  print(f"  Median margin:  {np.median(margins):.4f}")
  print(f"  Std margin:     {np.std(margins):.4f}")
  print(f"{'='*50}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=["compare", "winrate"], required=True)
  parser.add_argument("--model_path", type=str, required=True,
                      help="Path to DPO model (LoRA adapter dir or merged model dir)")
  parser.add_argument("--prompts_path", type=str,
                      help="Path to test prompts (one per line)")
  parser.add_argument("--dataset_path", type=str,
                      help="Path to test preference CSV (columns: prompt, chosen, rejected) for win-rate")
  parser.add_argument("--no_peft", action="store_true",
                      help="Model is a full merged model, not a LoRA adapter")
  args = parser.parse_args()

  is_peft = not args.no_peft

  if args.mode == "compare":
    assert args.prompts_path, "--prompts_path required"
    compare_models(args.model_path, args.prompts_path, is_peft)

  elif args.mode == "winrate":
    assert args.dataset_path, "--dataset_path required"
    compute_winrate(args.model_path, args.dataset_path, is_peft)


if __name__ == "__main__":
  main()
