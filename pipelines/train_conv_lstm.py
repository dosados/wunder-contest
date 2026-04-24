from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from constants import ARTIFACTS_ROOT
from training.tasks import run_training_job
from utils import (
    make_run_dir,
    plot_history,
    resolve_train_config_path,
    load_json_config,
    save_json,
    snapshot_config,
    update_latest_link,
)


def main():
    parser = argparse.ArgumentParser(description="Train Conv+LSTM model")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--no-optuna-fallback",
        action="store_true",
        help="Use default training config when --config is not provided",
    )
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--output-manifest", default=None)
    args = parser.parse_args()
    process_name = "train_conv_lstm"
    config_path = resolve_train_config_path(
        model_name="conv_lstm",
        explicit_path=args.config,
        prefer_optuna_best=not args.no_optuna_fallback,
    )
    cfg = load_json_config(config_path)
    root = Path(args.artifacts_root) if args.artifacts_root else Path(ARTIFACTS_ROOT)
    run_dir = make_run_dir(root, process_name)
    snapshot_config(config_path, run_dir)
    result, _ = run_training_job(cfg, process_name, run_dir=run_dir, snapshot_original_config=None)
    save_json(result.history, run_dir / "metrics_history.json")
    plot_history(result.history, run_dir / "plots")
    manifest = {
        "status": "success",
        "process": process_name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "weights_path": result.best_weights_path,
        "best_metric": result.best_metric,
        "best_epoch": result.best_epoch,
        "config_path": str(config_path.resolve()),
    }
    manifest_path = Path(args.output_manifest) if args.output_manifest else run_dir / "manifest.json"
    save_json(manifest, manifest_path)
    update_latest_link(run_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
