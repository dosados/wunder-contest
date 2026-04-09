from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_dir(artifacts_root: str | Path, process_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(artifacts_root) / process_name / ts
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot").mkdir(parents=True, exist_ok=True)
    return run_dir


def update_latest_link(run_dir: str | Path) -> None:
    run_dir = Path(run_dir).resolve()
    latest = run_dir.parent / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)


def snapshot_config(config_path: str | Path, run_dir: str | Path) -> None:
    dst = Path(run_dir) / "config_snapshot" / Path(config_path).name
    shutil.copy2(config_path, dst)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
