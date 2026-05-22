import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool,
    gradient_checkpointing: bool = False,
    device_index: int = 0,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    load_kwargs = dict(
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    if load_in_4bit:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer
