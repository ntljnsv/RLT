import typer

from algorithms.dpo.config import DPOConfig
from algorithms.dpo.train import run_dpo_training
from algorithms.dpo.evaluate import compare_models_cli, winrate_cli
from algorithms.dpo.merge_adapter import merge_lora_adapter
from shared.dataset_loader import (
    DEFAULT_DATASET,
    list_dataset_presets,
    resolve_dataset_path,
)

app = typer.Typer()
_defaults = DPOConfig()

_PRESET_HELP = (
    "Preset name or path to .csv/.json. "
    f"Presets: {', '.join(name for name, _ in list_dataset_presets())}. "
    f"Default: {DEFAULT_DATASET}"
)


@app.command("datasets")
def datasets_list():
    """List available dataset presets."""
    typer.echo("Dataset presets (--dataset / -d):\n")
    for name, path in list_dataset_presets():
        typer.echo(f"  {name:12}  {path}")
    typer.echo("\nOr pass any path: --dataset /path/to/file.csv")


@app.command()
def train(
    dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        "-d",
        help=_PRESET_HELP,
    ),
    model_id: str = _defaults.model_id,
    epochs: int = _defaults.epochs,
    lr: float = _defaults.learning_rate,
    beta: float = _defaults.beta,
    batch_size: int = _defaults.batch_size,
    grad_accum: int = _defaults.grad_accum,
    eval_batch_size: int = _defaults.eval_batch_size,
    max_length: int = _defaults.max_length,
    warmup_ratio: float = _defaults.warmup_ratio,
    cuda_device: int = _defaults.cuda_device,
    precompute_batch_size: int = _defaults.precompute_ref_batch_size,
    load_4bit: bool = _defaults.load_in_4bit,
    no_lora: bool = _defaults.no_lora,
    gradient_checkpointing: bool = _defaults.gradient_checkpointing,
    attn: str = typer.Option(
        _defaults.attn_implementation,
        "--attn",
        help="Attention backend: auto, flash_attention_2, sdpa, or eager.",
    ),
    output_dir: str = _defaults.output_dir,
    save_total_limit: int = _defaults.save_total_limit,
    save_steps: int = _defaults.save_steps,
    logging_steps: int = _defaults.logging_steps,
    report_to: str = _defaults.report_to,
    seed: int = _defaults.seed,
    eval_fraction: float = _defaults.eval_fraction,
    early_stopping: bool = typer.Option(
        _defaults.early_stopping,
        "--early-stopping/--no-early-stopping",
        help="Stop when eval_loss stops improving (needs --eval-fraction > 0).",
    ),
    early_stopping_patience: int = _defaults.early_stopping_patience,
    early_stopping_threshold: float = _defaults.early_stopping_threshold,
    resume_from_checkpoint: str | None = _defaults.resume_from_checkpoint,
    max_samples: int | None = _defaults.max_samples,
):
    """DPO training. Pick data with --dataset / -d (preset or file path)."""
    dataset_path = str(resolve_dataset_path(dataset))
    typer.echo(f"Using dataset: {dataset_path}")

    config = DPOConfig(
        dataset_path=dataset_path,
        max_samples=max_samples,
        model_id=model_id,
        epochs=epochs,
        learning_rate=lr,
        beta=beta,
        batch_size=batch_size,
        grad_accum=grad_accum,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        cuda_device=cuda_device,
        precompute_ref_batch_size=precompute_batch_size,
        load_in_4bit=load_4bit,
        no_lora=no_lora,
        gradient_checkpointing=gradient_checkpointing,
        attn_implementation=attn,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to=report_to,
        seed=seed,
        eval_fraction=eval_fraction,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    run_dpo_training(config)


@app.command()
def compare(
    model_path: str,
    prompts_path: str,
    model_id: str = _defaults.model_id,
):
    compare_models_cli(model_path, prompts_path, base_model_id=model_id)


@app.command()
def winrate(
    model_path: str,
    dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        "-d",
        help=_PRESET_HELP,
    ),
    model_id: str = _defaults.model_id,
    max_samples: int | None = 1000,
):
    dataset_path = str(resolve_dataset_path(dataset))
    typer.echo(f"Using dataset: {dataset_path}")
    winrate_cli(
        model_path,
        dataset_path,
        base_model_id=model_id,
        max_samples=max_samples,
    )


@app.command()
def merge(
    adapter_path: str,
    output_path: str,
    model_id: str = _defaults.model_id,
):
    merge_lora_adapter(adapter_path, output_path, base_model_id=model_id)


if __name__ == "__main__":
    app()
