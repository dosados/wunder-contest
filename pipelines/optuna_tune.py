from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from training.optuna_runner import run_optuna_study, save_best_optuna_result


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
    args = parser.parse_args()
    defaults = {
        "gru": REPO_ROOT / "configs" / "train_gru.json",
        "conv_lstm": REPO_ROOT / "configs" / "train_conv_lstm.json",
        "ssm": REPO_ROOT / "configs" / "train_ssm.json",
        "transformer": REPO_ROOT / "configs" / "train_transformer.json",
        "window_transformer": REPO_ROOT / "configs" / "train_transformer.json",
    }
    cfg_path = Path(args.config or defaults[args.model])
    from utils import load_json_config

    base = load_json_config(cfg_path)
    optuna_cfg = base.get("optuna") or {}
    n_trials = int(
        args.trials if args.trials is not None else optuna_cfg.get("n_trials", 20)
    )
    study_name = args.study_name or optuna_cfg.get("study_name")
    study = run_optuna_study(
        base_config_path=cfg_path,
        model_name=args.model,
        n_trials=n_trials,
        study_name_override=study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )
    out = save_best_optuna_result(
        study,
        cfg_path,
        args.model,
        output_path=args.output,
    )
    print(f"Best metric: {study.best_value}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
