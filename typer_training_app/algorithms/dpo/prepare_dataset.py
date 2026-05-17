import csv
from datasets import Dataset


def format_dpo_dataset(records, tokenizer):
    def format_row(r):
        prompt_msgs = [{"role": "user", "content": r["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": prompt,
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }

    formatted = [format_row(r) for r in records]
    return Dataset.from_list(formatted)
