from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from constants import ARTIFACTS_ROOT
from stacking import train_stack
from utils import (
    load_json_config,
    make_run_dir,
    plot_history,
    save_json,
    snapshot_config,
    get_logger,
    update_latest_link,
)


def main():
    parser = argparse.ArgumentParser(description="Train stack model (MLP or Ridge)")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "train_stack_mlp.json")
    )
    args = parser.parse_args()
    cfg = load_json_config(args.config)
    logger = get_logger("train_stack")
    run_dir = make_run_dir(ARTIFACTS_ROOT, "train_stack")
    snapshot_config(args.config, run_dir)
    logger.info("Run dir: %s", run_dir)
    result = train_stack(cfg, run_dir)
    save_json(result, Path(run_dir) / "manifest.json")
    if "history" in result:
        save_json(result["history"], Path(run_dir) / "metrics_history.json")
        plot_history(result["history"], Path(run_dir) / "plots")
    update_latest_link(run_dir)
    logger.info("Stack done. mode=%s", result.get("mode"))


if __name__ == "__main__":
    main()
