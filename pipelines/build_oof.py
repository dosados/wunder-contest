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
    default_train_config_path,
    load_json_config,
    make_run_dir,
    optuna_best_config_path,
    save_json,
    snapshot_config,
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
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument(
        "--no-optuna-fallback",
        action="store_true",
        help="Use default train configs instead of optuna best configs for missing model_config",
    )
    args = parser.parse_args()
    cfg = load_json_config(args.config)
    if args.model:
        cfg["models"] = [
            m for m in cfg.get("models", []) if m.get("name") == args.model
        ]
    for model_entry in cfg.get("models", []):
        if "model_config" in model_entry:
            continue
        model_name = str(model_entry.get("name", "")).strip().lower()
        if not model_name:
            continue
        best_path = optuna_best_config_path(model_name)
        source_path = (
            default_train_config_path(model_name)
            if args.no_optuna_fallback
            else (best_path if best_path.is_file() else default_train_config_path(model_name))
        )
        source_cfg = load_json_config(source_path)
        model_entry["model_config"] = (source_cfg.get("model") or {}).get("params", {})
    process_name = "build_oof"
    root = Path(args.artifacts_root) if args.artifacts_root else Path(ARTIFACTS_ROOT)
    run_dir = make_run_dir(root, process_name)
    snapshot_config(args.config, run_dir)
    outputs = build_oof_dataset(cfg, run_dir)
    manifest = {
        "status": "success",
        "process": process_name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "outputs": outputs,
        "config_path": str(Path(args.config).resolve()),
    }
    manifest_path = Path(args.output_manifest) if args.output_manifest else run_dir / "manifest.json"
    save_json(manifest, manifest_path)
    update_latest_link(run_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
