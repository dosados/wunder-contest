from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import ARTIFACTS_ROOT
from orchestration.train_meta_oof import train_meta_oof
from utils import load_json_config
from utils import make_run_dir, save_json, snapshot_config, update_latest_link


def main() -> None:
    parser = argparse.ArgumentParser(description="Train orchestration meta head")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "train_meta_oof.json")
    )
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--output-manifest", default=None)
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    process_name = f"train_meta_{cfg.get('variant', 'default')}"
    root = Path(args.artifacts_root) if args.artifacts_root else Path(ARTIFACTS_ROOT)
    run_dir = make_run_dir(root, process_name)
    snapshot_config(args.config, run_dir)
    outputs = train_meta_oof(cfg, run_dir)
    manifest = {
        "status": "success",
        "process": process_name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "weights_path": outputs.get("weights_path"),
        "best_metric": outputs.get("score"),
        "outputs": outputs,
        "config_path": str(Path(args.config).resolve()),
    }
    manifest_path = Path(args.output_manifest) if args.output_manifest else run_dir / "manifest.json"
    save_json(manifest, manifest_path)
    update_latest_link(run_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
