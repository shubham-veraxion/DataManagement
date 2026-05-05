import pandas as pd
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

def load_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise