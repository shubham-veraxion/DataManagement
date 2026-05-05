
import pandas as pd

def transform(df_dict):
    target_dataset = 'EOD'
    prompt = 'remove rows where price is null or zero'

    if not df_dict:
        raise ValueError("No datasets were provided for execution")

    if target_dataset is None:
        target_dataset = next(iter(df_dict.keys()))

    if target_dataset not in df_dict:
        raise KeyError(f"Target dataset '{target_dataset}' is not available")

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
