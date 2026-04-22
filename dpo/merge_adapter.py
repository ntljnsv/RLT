"""
Usage:
  python merge_adapter.py --adapter_path outputs/vezilka-dpo/final --output_path outputs/vezilka-dpo-merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL_ID = "finki-ukim/VezilkaLLM-Instruct"


def merge_and_save(adapter_path: str, output_path: str):
  print(f"Loading base model: {BASE_MODEL_ID}")
  model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
  )
  tokenizer = AutoTokenizer.from_pretrained(adapter_path)

  print(f"Loading LoRA adapter: {adapter_path}")
  model = PeftModel.from_pretrained(model, adapter_path)

  print("Merging adapter weights into base model...")
  model = model.merge_and_unload()

  print(f"Saving merged model to: {output_path}")
  model.save_pretrained(output_path, safe_serialization=True)
  tokenizer.save_pretrained(output_path)

  print("Done. You can now use the merged model with --no_peft flag in evaluate.py")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--adapter_path", type=str, required=True,
                      help="Path to LoRA adapter directory (output of train_dpo.py)")
  parser.add_argument("--output_path", type=str, required=True,
                      help="Where to save the merged model")
  args = parser.parse_args()
  merge_and_save(args.adapter_path, args.output_path)


if __name__ == "__main__":
  main()