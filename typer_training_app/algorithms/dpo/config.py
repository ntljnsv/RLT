from dataclasses import dataclass


@dataclass
class DPOConfig:
    model_id: str = "finki-ukim/VezilkaLLM-Instruct"

    epochs: int = 1
    learning_rate: float = 5e-7
    beta: float = 0.1
    batch_size: int = 2
    grad_accum: int = 4

    max_length: int = 1024
    warmup_ratio: float = 0.1

    load_in_4bit: bool = False
    no_lora: bool = False

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    output_dir: str = "outputs/dpo"
