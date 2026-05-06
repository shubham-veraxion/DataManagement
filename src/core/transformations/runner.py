from pathlib import Path
import shutil
import uuid
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

def _is_spark_df(df) -> bool:
    return df.__class__.__module__.startswith("pyspark.sql")


def save_output(df, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if _is_spark_df(df):
        tmp_dir = Path(path).parent / f"_spark_tmp_{uuid.uuid4().hex}"
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp_dir))

        part_files = list(tmp_dir.glob("part-*.csv"))
        if not part_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise FileNotFoundError("Spark output missing part file")

        shutil.move(str(part_files[0]), path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        df.to_csv(path, index=False)
    logger.info(f"Saved output: {path}")