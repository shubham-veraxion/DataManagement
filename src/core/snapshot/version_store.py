from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("storage/snapshots")


def list_snapshots() -> list[dict]:
    snapshots: list[dict] = []

    data_root = BASE / "data"
    if not data_root.exists():
        return snapshots

    for snapshot_dir in sorted(data_root.iterdir()):
        if not snapshot_dir.is_dir():
            continue

        data_file = snapshot_dir / "data.csv"
        code_file = BASE / "code" / snapshot_dir.name / "transform.py"
        meta_file = BASE / "code" / snapshot_dir.name / "meta.json"

        snapshots.append(
            {
                "snapshot_id": snapshot_dir.name,
                "data_path": str(data_file),
                "code_path": str(code_file),
                "meta_path": str(meta_file),
                "exists": data_file.exists(),
            }
        )

    return snapshots


def load_snapshot_data(snapshot_id: str) -> pd.DataFrame:
    data_file = BASE / "data" / snapshot_id / "data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Snapshot data not found: {data_file}")

    return pd.read_csv(data_file)