import os
import torch

from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType

from shared.model_loader import load_model_and_tokenizer
from algorithms.dpo.prepare_dataset import format_dpo_dataset
from shared.validation import validate_preference_dataset


def get_lora_config(cfg):
    if cfg.no_lora:
        return None

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )


def run_dpo_training(cfg, dataset):
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(cfg.model_id, cfg.load_in_4bit)

    print("Validating dataset...")
    records = dataset if isinstance(dataset, list) else dataset.to_pandas().to_dict("records")

    records = validate_preference_dataset(cfg.model_id if isinstance(cfg.model_id, str) else cfg.model_id)

    train_dataset = format_dpo_dataset(records, tokenizer)

    peft_config = get_lora_config(cfg)

    dpo_config = DPOConfig(
        beta=cfg.beta,
        max_length=1024,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        output_dir=cfg.output_dir,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Training...")
    trainer.train()

    print("Saving...")
    trainer.save_model(cfg.output_dir + "/final")
    tokenizer.save_pretrained(cfg.output_dir + "/final")
