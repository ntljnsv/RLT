from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset


def load_dataset_any(source: str):
    """
    Supports:
    - local CSV
    - local JSON
    - HuggingFace datasets
    """

    path = Path(source)

    # Local file
    if path.exists():
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            return Dataset.from_pandas(df)

        if path.suffix == ".json":
            df = pd.read_json(path)
            return Dataset.from_pandas(df)

        raise ValueError(f"Unsupported file type: {path.suffix}")

    # HuggingFace dataset
    return load_dataset(source)
