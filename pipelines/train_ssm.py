from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from training.tasks import run_base_training
from utils import load_json_config


def main():
    parser = argparse.ArgumentParser(description="Train SSM model")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "train_ssm.json")
    )
    args = parser.parse_args()
    cfg = load_json_config(args.config)
    run_base_training(cfg, args.config, "train_ssm")


if __name__ == "__main__":
    main()
