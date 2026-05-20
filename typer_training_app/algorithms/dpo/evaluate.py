import torch
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_ID = "finki-ukim/VezilkaLLM-Instruct"


def load_model(model_path: str, peft: bool = True):
    """
    Loads either:
    - LoRA adapter (PEFT model)
    - or full merged model
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if peft:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
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

    response = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()


def compare_models_cli(model_path: str, prompts_path: str):
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print("Loading trained model...")
    dpo_model, dpo_tokenizer = load_model(model_path, peft=True)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"\nRunning comparison on {len(prompts)} prompts...\n")

    for i, prompt in enumerate(prompts, 1):
        print("=" * 70)
        print(f"[{i}/{len(prompts)}] PROMPT:\n{prompt}")

        base_out = generate_response(base_model, base_tokenizer, prompt)
        dpo_out = generate_response(dpo_model, dpo_tokenizer, prompt)

        print("\nBASE MODEL:")
        print(base_out)

        print("\nDPO MODEL:")
        print(dpo_out)
        print()


def compute_logprob(model, tokenizer, prompt: str, response: str) -> float:
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
    labels[:, prompt_ids.shape[1]:] = response_ids

    with torch.no_grad():
        outputs = model(input_ids=full_ids, labels=labels)
        return -outputs.loss.item()


def winrate_cli(model_path: str, dataset_path: str):
    model, tokenizer = load_model(model_path, peft=True)

    with open(dataset_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    wins = 0
    margins = []

    print(f"\nEvaluating {len(rows)} samples...\n")

    for i, r in enumerate(rows, 1):
        lp_chosen = compute_logprob(model, tokenizer, r["prompt"], r["chosen"])
        lp_rejected = compute_logprob(model, tokenizer, r["prompt"], r["rejected"])

        margin = lp_chosen - lp_rejected
        margins.append(margin)

        if margin > 0:
            wins += 1

        if i % 20 == 0:
            print(f"Processed {i}/{len(rows)} | win-rate={wins/i:.2%}")

    print("\n" + "=" * 50)
    print(f"WIN RATE: {wins / len(rows):.2%}")
    print(f"Mean margin: {np.mean(margins):.4f}")
    print(f"Median margin: {np.median(margins):.4f}")
    print(f"Std margin: {np.std(margins):.4f}")
    print("=" * 50)