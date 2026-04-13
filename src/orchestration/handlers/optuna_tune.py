from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestration.job_contracts import JobSpec
from training.optuna_runner import (
    export_best_config_from_storage,
    run_optuna_study,
    save_best_optuna_result,
)
from utils import save_json


def run_optuna_tune(
    spec: JobSpec, run_dir: Path
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    cfg = spec.config
    model_name = str(cfg["model_name"])
    base_config_candidate = cfg.get("base_config_path") or spec.config_path
    if not base_config_candidate:
        raise ValueError("optuna_tune job requires base_config_path or spec.config_path")
    base_config_path = Path(base_config_candidate)
    if base_config_path.is_dir():
        raise ValueError(
            f"optuna_tune base config path must be a file, got directory: {base_config_path}"
        )
    if not base_config_path.exists():
        raise FileNotFoundError(f"optuna_tune base config not found: {base_config_path}")
    export_only = bool(cfg.get("export_only", False))
    if export_only:
        if not cfg.get("storage") or not cfg.get("study_name"):
            raise ValueError("export_only requires both storage and study_name")
        best = export_best_config_from_storage(
            base_config_path=base_config_path,
            model_name=model_name,
            study_name=str(cfg["study_name"]),
            storage=str(cfg["storage"]),
            output_path=cfg.get("output_path"),
        )
        return ({}, {}, best)
    study = run_optuna_study(
        base_config_path=base_config_path,
        model_name=model_name,
        n_trials=int(cfg["n_trials"]),
        study_name_override=cfg.get("study_name"),
        storage=cfg.get("storage"),
        load_if_exists=bool(cfg.get("load_if_exists", False)),
    )
    best = save_best_optuna_result(
        study,
        base_config_path,
        model_name,
        output_path=cfg.get("output_path"),
    )
    out = {
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_config_path": str(best),
    }
    save_json(out, run_dir / "optuna_summary.json")
    return ({}, {"best_metric": out["best_value"]}, out)
