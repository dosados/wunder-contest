from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from orchestration.jobs import build_job_spec, execute_job
from utils import load_json_config


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
    args = parser.parse_args()
    cfg = load_json_config(args.config)
    if args.model:
        cfg["models"] = [
            m for m in cfg.get("models", []) if m.get("name") == args.model
        ]
    spec = build_job_spec(
        job_type="build_oof",
        process_name="build_oof",
        config=cfg,
        config_path=args.config,
        artifacts_root=args.artifacts_root,
        output_manifest=args.output_manifest,
        write_latest_link=True,
    )
    result = execute_job(spec)
    print(result.manifest_path)


if __name__ == "__main__":
    main()
