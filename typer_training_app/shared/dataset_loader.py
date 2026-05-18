from pathlib import Path
import pandas as pd
from datasets import load_dataset


def _to_list(ds):
    if isinstance(ds, list):
        return ds

    if hasattr(ds, "to_pandas"):
        return ds.to_pandas().to_dict("records")

    if hasattr(ds, "to_dict"):
        return ds.to_dict("records")

    raise ValueError("Unsupported dataset format")


def load_dataset_any(source: str):
    """
    Supports:
    - local JSON
    - local CSV
    - HuggingFace dataset
    RETURNS: list[dict]
    """

    path = Path(source)

    if path.exists():

        if path.suffix == ".csv":
            df = pd.read_csv(path)
            return df.to_dict("records")

        if path.suffix == ".json":
            df = pd.read_json(path)
            return df.to_dict("records")

        raise ValueError(f"Unsupported file type: {path.suffix}")

    ds = load_dataset(source)

    if isinstance(ds, dict):
        ds = ds[list(ds.keys())[0]]

    return _to_list(ds)