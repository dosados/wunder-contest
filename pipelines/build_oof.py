from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from constants import ARTIFACTS_ROOT
from stacking import build_oof_dataset
from utils import (
    load_json_config,
    make_run_dir,
    save_json,
    snapshot_config,
    get_logger,
    update_latest_link,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build OOF datasets for selected models"
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "build_oof.json")
    )
    parser.add_argument(
        "--model", default=None, help="Optional single model name to rebuild"
    )
    args = parser.parse_args()
    cfg = load_json_config(args.config)
    if args.model:
        cfg["models"] = [
            m for m in cfg.get("models", []) if m.get("name") == args.model
        ]
    logger = get_logger("build_oof")
    run_dir = make_run_dir(ARTIFACTS_ROOT, "build_oof")
    snapshot_config(args.config, run_dir)
    logger.info("Run dir: %s", run_dir)
    outputs = build_oof_dataset(cfg, run_dir)
    save_json({"outputs": outputs}, Path(run_dir) / "manifest.json")
    update_latest_link(run_dir)
    logger.info("OOF done. merged=%s", outputs.get("merged"))


if __name__ == "__main__":
    main()
