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
    update_latest_link,
)


def _extract_best_metric(history):
    if not isinstance(history, dict):
        return (None, None)
    val_metrics = history.get("val", {}).get("contest_metric")
    if isinstance(val_metrics, list) and val_metrics:
        best_metric = max(val_metrics)
        best_epoch = val_metrics.index(best_metric) + 1
        return (float(best_metric), int(best_epoch))
    train_metrics = history.get("train", {}).get("contest_metric")
    if isinstance(train_metrics, list) and train_metrics:
        best_metric = max(train_metrics)
        best_epoch = train_metrics.index(best_metric) + 1
        return (float(best_metric), int(best_epoch))
    return (None, None)


def main():
    parser = argparse.ArgumentParser(description="Train stack model (MLP, Ridge, XGBoost)")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "train_stack_mlp.json")
    )
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--output-manifest", default=None)
    args = parser.parse_args()
    process_name = "train_stack"
    cfg = load_json_config(args.config)
    root = Path(args.artifacts_root) if args.artifacts_root else Path(ARTIFACTS_ROOT)
    run_dir = make_run_dir(root, process_name)
    snapshot_config(args.config, run_dir)
    outputs = train_stack(cfg, run_dir)
    history = outputs.get("history")
    if isinstance(history, dict):
        save_json(history, run_dir / "metrics_history.json")
        plot_history(history, run_dir / "plots")
    feature_spec = outputs.get("feature_spec")
    if isinstance(feature_spec, dict):
        save_json(feature_spec, run_dir / "stack_feature_spec.json")
    best_metric, best_epoch = _extract_best_metric(history)
    manifest = {
        "status": "success",
        "process": process_name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "weights_path": outputs.get("weights_path"),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "outputs": outputs,
        "config_path": str(Path(args.config).resolve()),
    }
    manifest_path = Path(args.output_manifest) if args.output_manifest else run_dir / "manifest.json"
    save_json(manifest, manifest_path)
    update_latest_link(run_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
