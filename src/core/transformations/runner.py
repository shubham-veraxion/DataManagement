import pandas as pd
from pathlib import Path
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

def save_output(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved output: {path}")