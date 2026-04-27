from __future__ import annotations
import copy
import gc
import json
from pathlib import Path
from typing import Any
import optuna
import torch
from constants import ARTIFACTS_ROOT
from training.optuna_spaces import normalize_model_name, sample_model_params
from training.tasks import run_training_job
from utils import get_logger, load_json_config, make_run_dir, save_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return _repo_root() / "configs"


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_trial_training_config(
    base: dict[str, Any],
    model_name: str,
    trial: optuna.Trial,
    *,
    trial_epochs: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["epochs"] = int(trial_epochs)
    cfg["lr"] = float(trial.suggest_float("lr", 1e-4, 5e-3, log=True))
    bs = trial.suggest_categorical("batch_size", [8, 16, 32])
    cfg["batch_size"] = int(bs)
    mname = normalize_model_name(model_name)
    cfg["model"] = dict(cfg.get("model") or {})
    cfg["model"]["name"] = mname
    cfg["model"]["params"] = sample_model_params(trial, model_name)
    return cfg


def run_optuna_study(
    *,
    base_config_path: str | Path,
    model_name: str,
    n_trials: int,
    study_name_override: str | None = None,
    storage: str | None = None,
    load_if_exists: bool = False,
) -> optuna.Study:
    logger = get_logger("optuna")
    base = load_json_config(base_config_path)
    optuna_section = base.get("optuna") or {}
    trial_epochs = int(optuna_section.get("trial_epochs", 5))
    resolved_study_name = study_name_override or optuna_section.get(
        "study_name", f"tune_{normalize_model_name(model_name)}"
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = None
        process = None
        run_dir = None
        result = None
        rd = None
        try:
            cfg = build_trial_training_config(
                base, model_name, trial, trial_epochs=trial_epochs
            )
            process = f"optuna_{normalize_model_name(model_name)}"
            run_dir = make_run_dir(
                Path(ARTIFACTS_ROOT) / "optuna",
                f"{resolved_study_name}/trial",
            )
            result, rd = run_training_job(
                cfg,
                process,
                run_dir=run_dir,
                snapshot_original_config=None,
            )
            trial.set_user_attr("run_dir", str(rd))
            trial.set_user_attr("best_metric", float(result.best_metric))
            trial.set_user_attr("best_epoch", int(result.best_epoch))
            return float(result.best_metric)
        finally:
            del cfg, process, run_dir, result, rd
            _cleanup_memory()

    create_kwargs: dict[str, Any] = {"direction": "maximize"}
    if resolved_study_name:
        create_kwargs["study_name"] = resolved_study_name
    if storage:
        create_kwargs["storage"] = storage
        create_kwargs["load_if_exists"] = load_if_exists
    study = optuna.create_study(**create_kwargs)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(
        "Optuna finished. best_value=%s trial=%s",
        study.best_value,
        study.best_trial.number,
    )
    _cleanup_memory()
    return study


def save_best_optuna_result(
    study: optuna.Study,
    base_config_path: str | Path,
    model_name: str,
    *,
    output_path: str | Path | None = None,
    force_save: bool = False,
) -> dict[str, Any]:
    base = load_json_config(base_config_path)
    best = study.best_trial
    run_dir = Path(best.user_attrs["run_dir"])
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        best_trial_cfg = json.load(f)
    best_trial_cfg.pop("optuna", None)
    production_epochs = int(base.get("epochs", best_trial_cfg.get("epochs", 20)))
    best_trial_cfg["epochs"] = production_epochs
    out = copy.deepcopy(best_trial_cfg)
    new_best_value = float(study.best_value)
    out["_meta"] = {
        "model_name": normalize_model_name(model_name),
        "best_value": new_best_value,
        "best_trial_number": int(best.number),
        "metric": "contest_metric",
        "base_config_path": str(Path(base_config_path).resolve()),
        "best_trial_run_dir": str(run_dir),
        "optuna_params": best.params,
    }
    out_path = Path(
        output_path
        if output_path is not None
        else configs_dir() / f"optuna_best_{normalize_model_name(model_name)}.json"
    )
    previous_best_value = None
    if out_path.is_file():
        previous = load_json_config(out_path)
        if isinstance(previous, dict):
            meta = previous.get("_meta")
            if isinstance(meta, dict):
                prev_value = meta.get("best_value")
                if isinstance(prev_value, (float, int)):
                    previous_best_value = float(prev_value)
    should_save = (
        force_save
        or previous_best_value is None
        or new_best_value > previous_best_value
    )
    if should_save:
        save_json(out, out_path)
    return {
        "path": out_path,
        "saved": should_save,
        "previous_best_value": previous_best_value,
        "new_best_value": new_best_value,
    }


def export_best_config_from_storage(
    *,
    base_config_path: str | Path,
    model_name: str,
    study_name: str,
    storage: str,
    output_path: str | Path | None = None,
    force_save: bool = False,
) -> dict[str, Any]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    save_result = save_best_optuna_result(
        study,
        base_config_path,
        model_name,
        output_path=output_path,
        force_save=force_save,
    )
    best = study.best_trial
    run_dir = Path(best.user_attrs["run_dir"])
    return {
        "best_value": float(study.best_value),
        "best_trial_number": int(best.number),
        "best_trial_run_dir": str(run_dir),
        "train_ready_config_path": str(save_result["path"]),
        "config_saved": bool(save_result["saved"]),
        "previous_best_value": save_result["previous_best_value"],
        "new_best_value": save_result["new_best_value"],
        "study_name": study.study_name,
    }
