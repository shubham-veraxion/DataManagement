from src.core.logging.logger import get_logger

logger = get_logger(__name__)

def generate_code(prompt: str, target_dataset: str | None = None) -> str:
    """
    Generate a deterministic transformation scaffold.
    """

    target_dataset_repr = repr(target_dataset) if target_dataset else "None"
    prompt_literal = repr(prompt.lower())

    code = f"""
import pandas as pd

def transform(df_dict):
    target_dataset = {target_dataset_repr}
    prompt = {prompt_literal}

    if not df_dict:
        raise ValueError("No datasets were provided for execution")

    if target_dataset is None:
        target_dataset = next(iter(df_dict.keys()))

    if target_dataset not in df_dict:
        raise KeyError(f"Target dataset '{{target_dataset}}' is not available")

    df = df_dict[target_dataset].copy()

    if any(word in prompt for word in ["null", "missing", "na", "empty"]):
        if "price" in prompt and "price" in df.columns:
            df = df.loc[df["price"].notna()].copy()
        else:
            df = df.dropna().copy()

    if "duplicate" in prompt:
        df = df.drop_duplicates().copy()

    if "zero" in prompt and "price" in prompt and "price" in df.columns:
        df = df.loc[df["price"].notna() & (df["price"] != 0)].copy()

    return df
"""
    logger.info("Code generated")
    return code