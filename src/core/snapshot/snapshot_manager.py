import os
import json
import shutil
from datetime import datetime
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

BASE = "storage/snapshots"

def create_snapshot(data_path: str, code: str, metadata: dict):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snap_id = f"snap_{ts}"

    data_dir = os.path.join(BASE, "data", snap_id)
    code_dir = os.path.join(BASE, "code", snap_id)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)

    shutil.copy(data_path, os.path.join(data_dir, "data.csv"))

    with open(os.path.join(code_dir, "transform.py"), "w") as f:
        f.write(code)

    with open(os.path.join(code_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Snapshot created: {snap_id}")
    return snap_id