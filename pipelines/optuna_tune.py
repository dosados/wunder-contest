from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from constants import ARTIFACTS_ROOT
from training.optuna_runner import (
    export_best_config_from_storage,
    run_optuna_study,
    save_best_optuna_result,
)
from utils import (
    default_train_config_path,
    load_json_config,
    make_run_dir,
    save_json,
    snapshot_config,
    update_latest_link,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for a base training config"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["gru", "conv_lstm", "ssm", "transformer", "window_transformer"],
        help="Which search space and factory model name to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Base JSON config (defaults per model under configs/)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials (overrides config optuna.n_trials when set)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (default: config optuna.study_name or tune_<model>)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL for persistent studies (optional)",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="With --storage, resume an existing study",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write best config JSON (default: configs/optuna_best_<model>.json)",
    )
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip tuning and export best config from existing study storage",
    )
    parser.add_argument(
        "--force-save",
        action="store_true",
        help="Always overwrite output config even when metric is not improved",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else default_train_config_path(args.model)
    base = load_json_config(cfg_path)
    optuna_cfg = base.get("optuna") or {}
    n_trials = int(
        args.trials if args.trials is not None else optuna_cfg.get("n_trials", 20)
    )
    study_name = args.study_name or optuna_cfg.get("study_name")
    process_name = f"optuna_tune_{args.model}"
    root = Path(args.artifacts_root) if args.artifacts_root else Path(ARTIFACTS_ROOT)
    run_dir = make_run_dir(root, process_name)
    snapshot_config(cfg_path, run_dir)
    if args.export_only:
        if not args.storage or not study_name:
            raise ValueError("export-only requires both --storage and --study-name")
        outputs = export_best_config_from_storage(
            base_config_path=cfg_path,
            model_name=args.model,
            study_name=study_name,
            storage=args.storage,
            output_path=args.output,
            force_save=args.force_save,
        )
        best_metric = outputs.get("best_value")
        best_cfg = outputs.get("train_ready_config_path")
    else:
        study = run_optuna_study(
            base_config_path=cfg_path,
            model_name=args.model,
            n_trials=n_trials,
            study_name_override=study_name,
            storage=args.storage,
            load_if_exists=args.load_if_exists,
        )
        save_result = save_best_optuna_result(
            study,
            cfg_path,
            args.model,
            output_path=args.output,
            force_save=args.force_save,
        )
        outputs = {
            "study_name": study.study_name,
            "best_value": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
            "best_config_path": str(save_result["path"]),
            "config_saved": bool(save_result["saved"]),
            "previous_best_value": save_result["previous_best_value"],
            "new_best_value": save_result["new_best_value"],
        }
        save_json(outputs, run_dir / "optuna_summary.json")
        best_metric = outputs["best_value"]
        best_cfg = outputs["best_config_path"]
    manifest = {
        "status": "success",
        "process": process_name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "best_metric": best_metric,
        "outputs": outputs,
        "config_path": str(cfg_path.resolve()),
    }
    manifest_path = Path(args.output_manifest) if args.output_manifest else run_dir / "manifest.json"
    save_json(manifest, manifest_path)
    update_latest_link(run_dir)
    if best_metric is not None:
        print(f"Best metric: {best_metric}")
    if isinstance(best_cfg, str):
        print(f"Saved: {best_cfg}")
    print(manifest_path)


if __name__ == "__main__":
    main()
