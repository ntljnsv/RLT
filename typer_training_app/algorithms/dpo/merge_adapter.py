import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "finki-ukim/VezilkaLLM-Instruct"


def merge_lora_adapter(adapter_path: str, output_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Merged model saved.")
