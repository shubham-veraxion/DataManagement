from src.core.logging.logger import get_logger
from pyspark.sql import SparkSession, DataFrame

logger = get_logger(__name__)

_SPARK_SESSION: SparkSession | None = None


def get_spark_session() -> SparkSession:
    global _SPARK_SESSION
    if _SPARK_SESSION is None:
        _SPARK_SESSION = (
            SparkSession.builder
            .appName("AgenticMDMTransformer")
            .getOrCreate()
        )
    return _SPARK_SESSION

def load_csv(file_path: str) -> DataFrame:
    try:
        spark = get_spark_session()
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(file_path)
        )
        logger.info(f"Loaded CSV: {file_path}")
        return df
    except Exception as e:
        logger.exception("Error loading CSV: %s", file_path)
        raise RuntimeError(f"Failed to load CSV: {file_path}. {e}") from e