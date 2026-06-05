from dataclasses import dataclass, asdict
import json
from pathlib import Path

from shared.dataset_loader import DEFAULT_DATASET, resolve_dataset_path


@dataclass
class DPOConfig:
    dataset_path: str = str(resolve_dataset_path(DEFAULT_DATASET))
    max_samples: int | None = None
    model_id: str = "finki-ukim/VezilkaLLM-Instruct"

    epochs: int = 10
    learning_rate: float = 5e-7
    beta: float = 0.1

    batch_size: int = 12
    grad_accum: int = 2
    eval_batch_size: int = 4

    max_length: int = 768
    max_prompt_length: int | None = None
    max_completion_length: int | None = None
    warmup_ratio: float = 0.1

    cuda_device: int = 0
    precompute_ref_log_probs: bool = True
    precompute_ref_batch_size: int = 16

    load_in_4bit: bool = True
    no_lora: bool = False
    gradient_checkpointing: bool = True
    attn_implementation: str = "auto"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    output_dir: str = "outputs/dpo"
    save_total_limit: int = 1
    save_steps: int = 0
    logging_steps: int = 10
    report_to: str = "none"
    seed: int = 42

    eval_fraction: float = 0.05
    eval_strategy: str = "epoch"

    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "eval_loss"

    resume_from_checkpoint: str | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
