import typer

from algorithms.dpo.config import DPOConfig
from algorithms.dpo.train import run_dpo_training
from algorithms.dpo.evaluate import compare_models_cli, winrate_cli
from algorithms.dpo.merge_adapter import merge_lora_adapter
from shared.dataset_loader import load_dataset_any

app = typer.Typer()


@app.command()
def train(
    dataset: str,
    model_id: str = "finki-ukim/VezilkaLLM-Instruct",
    epochs: int = 1,
    lr: float = 5e-7,
    beta: float = 0.1,
    batch_size: int = 2,
    grad_accum: int = 4,
    load_4bit: bool = False,
    no_lora: bool = False,
    output_dir: str = "outputs/dpo_run",
):

    print("Loading dataset...")
    dataset_obj = load_dataset_any(dataset)

    config = DPOConfig(
        model_id=model_id,
        epochs=epochs,
        learning_rate=lr,
        beta=beta,
        batch_size=batch_size,
        grad_accum=grad_accum,
        load_in_4bit=load_4bit,
        no_lora=no_lora,
        output_dir=output_dir,
    )

    run_dpo_training(config, dataset_obj)


@app.command()
def compare(model_path: str, prompts_path: str):
    compare_models_cli(model_path, prompts_path)


@app.command()
def winrate(model_path: str, dataset_path: str):
    winrate_cli(model_path, dataset_path)


@app.command()
def merge(adapter_path: str, output_path: str):
    merge_lora_adapter(adapter_path, output_path)


if __name__ == "__main__":
    app()