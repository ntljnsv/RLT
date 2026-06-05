import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_attention_implementation(requested: str) -> str:
    """Pick an attention backend: auto prefers flash_attention_2 when installed."""
    req = (requested or "auto").strip().lower()
    if req not in ("auto", "flash_attention_2", "sdpa", "eager"):
        raise ValueError(
            "attn_implementation must be auto, flash_attention_2, sdpa, or eager; "
            f"got {requested!r}"
        )
    if req != "auto":
        return req

    try:
        import flash_attn
    except ImportError:
        print("flash-attn not installed; using sdpa.")
        return "sdpa"
    return "flash_attention_2"


def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool,
    gradient_checkpointing: bool = False,
    device_index: int = 0,
    attn_implementation: str = "auto",
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

    attn = resolve_attention_implementation(attn_implementation)
    print(f"Using attn_implementation={attn}")

    load_kwargs = dict(
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    )
    load_kwargs["torch_dtype"] = torch.bfloat16
    if load_in_4bit:
        load_kwargs["quantization_config"] = bnb_config

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        if attn != "flash_attention_2":
            raise
        print(f"flash_attention_2 load failed ({exc}); falling back to sdpa.")
        load_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer
