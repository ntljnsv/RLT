from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Repo layout: typer_training_app/shared/ -> parents[2] == RLT root
REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
PREFERENCE_DATA = REPO_ROOT / "preference_data"
APP_DATA = APP_ROOT / "data"

# Named presets: use --dataset <name> or -d <name>
DATASET_PRESETS: dict[str, Path] = {
    "gpp": PREFERENCE_DATA / "gpp_combined.csv",
    "gpp_combined": PREFERENCE_DATA / "gpp_combined.csv",
    "0_1k": PREFERENCE_DATA / "preference_pairs_0_1k.csv",
    "0_10k": PREFERENCE_DATA / "preference_pairs_0_10k.csv",
    "10k_50k": PREFERENCE_DATA / "preference_pairs_10k_50k.csv",
    "50k_100k": PREFERENCE_DATA / "preference_pairs_50k_100k.csv",
    "100k_150k": PREFERENCE_DATA / "preference_pairs_100k_150k.csv",
    "150k_end": PREFERENCE_DATA / "preference_pairs_150k_end.csv",
    "short": APP_DATA / "preference_pairs_super_duper_short.json",
    "json_combined": PREFERENCE_DATA / "aya_pref_pairs_combined.json",
}

DEFAULT_DATASET = "json_combined"
DEFAULT_DATASET_PATH = DATASET_PRESETS[DEFAULT_DATASET]


def list_dataset_presets() -> list[tuple[str, Path]]:
    """Unique preset names (skip duplicate paths like gpp vs gpp_combined)."""
    seen: set[Path] = set()
    out: list[tuple[str, Path]] = []
    for name, path in DATASET_PRESETS.items():
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((name, path))
    return sorted(out, key=lambda x: x[0])


def resolve_dataset_path(dataset: str) -> Path:
    """
    Resolve --dataset value to a concrete path.
    Accepts a preset name (e.g. 'gpp', 'json') or a path to CSV/JSON.
    """
    key = dataset.strip()
    if key in DATASET_PRESETS:
        path = DATASET_PRESETS[key]
        if not path.exists():
            raise FileNotFoundError(f"Preset '{key}' points to missing file: {path}")
        return path

    path = Path(key)
    if path.exists():
        return path

    raise ValueError(
        f"Unknown dataset '{dataset}'. "
        f"Use a preset name ({', '.join(sorted(DATASET_PRESETS))}) "
        f"or a path to an existing .csv / .json file."
    )

# gpp_combined.csv columns -> DPO fields
GPP_COLUMN_ALIASES = {
    "prompt": ("prompt", "user_question"),
    "chosen": ("chosen", "response_a", "preferred_answer"),
    "rejected": ("rejected", "response_b", "non_preferred_answer"),
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map existing CSV columns to prompt / chosen / rejected."""
    rename = {}
    for target, aliases in GPP_COLUMN_ALIASES.items():
        for name in aliases:
            if name in df.columns:
                rename[name] = target
                break
        else:
            raise ValueError(
                f"CSV missing a column for '{target}'. "
                f"Expected one of {aliases}. Found: {list(df.columns)}"
            )
    return rename


def _coalesce_columns(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    """Pick the first non-empty value across alias columns (e.g. chosen vs response_a)."""
    present = [name for name in aliases if name in df.columns]
    if not present:
        raise ValueError(
            f"Dataset missing a column for one of {aliases}. Found: {list(df.columns)}"
        )

    def _non_empty(series: pd.Series) -> pd.Series:
        as_str = series.astype("string")
        return as_str.notna() & (as_str.str.strip() != "")

    result = df[present[0]].astype("string")
    for name in present[1:]:
        candidate = df[name].astype("string")
        use_candidate = _non_empty(candidate) & ~_non_empty(result)
        result = result.where(~use_candidate, candidate)
    return result


def normalize_preference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt": _coalesce_columns(df, GPP_COLUMN_ALIASES["prompt"]),
            "chosen": _coalesce_columns(df, GPP_COLUMN_ALIASES["chosen"]),
            "rejected": _coalesce_columns(df, GPP_COLUMN_ALIASES["rejected"]),
        }
    )


def load_csv_preference_pairs(path: Path, max_samples: int | None = None) -> list[dict]:
    df = pd.read_csv(path, nrows=max_samples)
    df = normalize_preference_dataframe(df)

    for col in ("prompt", "chosen", "rejected"):
        df[col] = df[col].astype(str).str.strip()

    df = df[(df["prompt"] != "") & (df["chosen"] != "") & (df["rejected"] != "")]
    df = df[df["chosen"] != df["rejected"]]

    return df.to_dict("records")


def _to_list(ds):
    if isinstance(ds, list):
        return ds

    if hasattr(ds, "to_pandas"):
        return ds.to_pandas().to_dict("records")

    if hasattr(ds, "to_dict"):
        return ds.to_dict("records")

    raise ValueError("Unsupported dataset format")


def load_dataset_any(source: str, max_samples: int | None = None):
    """
    Load preference pairs from a preset name, local CSV/JSON, or HuggingFace dataset id.
    """
    path = resolve_dataset_path(source)

    if path.exists():
        if path.suffix == ".csv":
            return load_csv_preference_pairs(path, max_samples=max_samples)

        if path.suffix == ".json":
            df = pd.read_json(path)
            if max_samples is not None:
                df = df.head(max_samples)
            df = normalize_preference_dataframe(df)
            for col in ("prompt", "chosen", "rejected"):
                df[col] = df[col].astype(str).str.strip()
            df = df[(df["prompt"] != "") & (df["chosen"] != "") & (df["rejected"] != "")]
            df = df[df["chosen"] != df["rejected"]]
            return df.to_dict("records")

        raise ValueError(f"Unsupported file type: {path.suffix}")

    ds = load_dataset(source)
    if isinstance(ds, dict):
        ds = ds[list(ds.keys())[0]]

    records = _to_list(ds)
    if max_samples is not None:
        records = records[:max_samples]
    return records
