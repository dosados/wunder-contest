from __future__ import annotations
import argparse
import copy
import gc
import sys
from pathlib import Path
from typing import Any
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import ARTIFACTS_ROOT
from stacking import build_oof_dataset, train_stack
from training.optuna_runner import run_optuna_study, save_best_optuna_result
from training.tasks import run_training_job
from utils import (
    default_train_config_path,
    get_logger,
    load_json_config,
    make_run_dir,
    optuna_best_config_path,
    plot_history,
    save_json,
    snapshot_config,
    update_latest_link,
)

SUPPORTED_META_MODELS = {"ridge", "mlp", "xgboost"}


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _normalize_model_name(name: str) -> str:
    n = name.strip().lower()
    return "transformer" if n == "window_transformer" else n


def _extract_best_metric(history: dict[str, Any] | None) -> tuple[float | None, int | None]:
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


def _read_train_model_params(config_path: Path) -> dict[str, Any]:
    cfg = load_json_config(config_path)
    return copy.deepcopy((cfg.get("model") or {}).get("params") or {})


def _run_optuna_for_model(
    model_name: str,
    base_config_path: Path,
    trials: int | None,
    study_name_prefix: str | None,
    storage: str | None,
    load_if_exists: bool,
    output_path: Path,
    force_save: bool,
) -> dict[str, Any]:
    base_cfg = load_json_config(base_config_path)
    optuna_section = base_cfg.get("optuna") or {}
    n_trials = int(trials if trials is not None else optuna_section.get("n_trials", 20))
    default_study_name = optuna_section.get("study_name", f"tune_{_normalize_model_name(model_name)}")
    study_name = f"{study_name_prefix}_{_normalize_model_name(model_name)}" if study_name_prefix else default_study_name
    study = run_optuna_study(
        base_config_path=base_config_path,
        model_name=model_name,
        n_trials=n_trials,
        study_name_override=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    save_result = save_best_optuna_result(
        study,
        base_config_path,
        model_name,
        output_path=output_path,
        force_save=force_save,
    )
    return {
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_config_path": str(save_result["path"]),
        "config_saved": bool(save_result["saved"]),
        "previous_best_value": save_result["previous_best_value"],
        "new_best_value": save_result["new_best_value"],
    }


def _train_model_with_config(
    model_name: str,
    config_path: Path,
    artifacts_root: Path,
) -> dict[str, Any]:
    process_name = f"train_{_normalize_model_name(model_name)}"
    cfg = load_json_config(config_path)
    run_dir = make_run_dir(artifacts_root, process_name)
    snapshot_config(config_path, run_dir)
    result, _ = run_training_job(
        cfg,
        process_name,
        run_dir=run_dir,
        snapshot_original_config=None,
    )
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
    save_json(manifest, run_dir / "manifest.json")
    update_latest_link(run_dir)
    return manifest


def _meta_config_from_mode(
    mode: str,
    oof_path: str,
    selected_models: list[str],
    mlp_template: dict[str, Any],
    ridge_template: dict[str, Any],
    xgboost_template: dict[str, Any],
) -> dict[str, Any]:
    if mode == "mlp":
        cfg = copy.deepcopy(mlp_template)
    elif mode == "ridge":
        cfg = copy.deepcopy(ridge_template)
    elif mode == "xgboost":
        cfg = copy.deepcopy(xgboost_template)
    else:
        raise ValueError(f"Unsupported meta model mode: {mode}")
    cfg["mode"] = mode
    cfg["oof_path"] = oof_path
    cfg["selected_models"] = selected_models
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full training and stacking pipeline")
    parser.add_argument(
        "--models",
        default="conv_lstm,gru,ssm",
        help="Comma-separated base models to include",
    )
    parser.add_argument(
        "--meta-models",
        default="ridge,mlp",
        help="Comma-separated meta models to try: ridge,mlp,xgboost",
    )
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument(
        "--skip-optuna-models",
        default="",
        help="Comma-separated base models to skip optuna for",
    )
    parser.add_argument("--trials-per-model", type=int, default=None)
    parser.add_argument("--study-name-prefix", type=str, default=None)
    parser.add_argument("--optuna-storage", type=str, default=None)
    parser.add_argument("--load-if-exists", action="store_true")
    parser.add_argument(
        "--force-save",
        action="store_true",
        help="Always overwrite optuna best config output files",
    )
    parser.add_argument("--oof-folds", type=int, default=None)
    parser.add_argument("--oof-seed", type=int, default=None)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--meta-epochs", type=int, default=None)
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--output-summary", default=None)
    args = parser.parse_args()

    logger = get_logger("full_pipeline")
    requested_models = [_normalize_model_name(m) for m in _split_csv(args.models)]
    requested_meta = [m.strip().lower() for m in _split_csv(args.meta_models)]
    skip_optuna_models = {_normalize_model_name(m) for m in _split_csv(args.skip_optuna_models)}

    if not requested_models:
        raise ValueError("No base models selected")
    unknown_base = [m for m in requested_models if m not in {"conv_lstm", "gru", "ssm", "transformer"}]
    if unknown_base:
        raise ValueError(f"Unsupported base models: {unknown_base}")
    unknown_meta = [m for m in requested_meta if m not in SUPPORTED_META_MODELS]
    if unknown_meta:
        raise ValueError(f"Unsupported meta models: {unknown_meta}")
    unknown_skip = [m for m in skip_optuna_models if m not in requested_models]
    if unknown_skip:
        raise ValueError(
            f"skip-optuna-models must be subset of --models, got extra: {unknown_skip}"
        )

    artifacts_root = Path(args.artifacts_root) if args.artifacts_root else Path(ARTIFACTS_ROOT)
    run_dir = make_run_dir(artifacts_root, "full_pipeline")
    update_latest_link(run_dir)
    runtime_cfg_dir = run_dir / "runtime_configs"
    runtime_cfg_dir.mkdir(parents=True, exist_ok=True)

    optuna_results: dict[str, Any] = {}
    chosen_config_paths: dict[str, Path] = {}
    training_results: dict[str, Any] = {}

    for model in requested_models:
        base_cfg_path = default_train_config_path(model)
        best_cfg_path = optuna_best_config_path(model)
        should_skip_optuna = args.skip_optuna or (model in skip_optuna_models)
        if should_skip_optuna:
            chosen_config_paths[model] = best_cfg_path if best_cfg_path.is_file() else base_cfg_path
            logger.info("Skip optuna for %s, using %s", model, chosen_config_paths[model])
            _cleanup_memory()
            continue
        logger.info("Run optuna for %s", model)
        optuna_info = _run_optuna_for_model(
            model_name=model,
            base_config_path=base_cfg_path,
            trials=args.trials_per_model,
            study_name_prefix=args.study_name_prefix,
            storage=args.optuna_storage,
            load_if_exists=args.load_if_exists,
            output_path=best_cfg_path,
            force_save=args.force_save,
        )
        optuna_results[model] = optuna_info
        chosen_config_paths[model] = Path(optuna_info["best_config_path"])
        _cleanup_memory()

    for model in requested_models:
        config_path = chosen_config_paths.get(model, default_train_config_path(model))
        if args.base_epochs is not None:
            cfg = load_json_config(config_path)
            cfg["epochs"] = int(args.base_epochs)
            overridden_path = runtime_cfg_dir / f"runtime_{model}.json"
            save_json(cfg, overridden_path)
            config_path = overridden_path
        logger.info("Train %s with %s", model, config_path)
        training_results[model] = _train_model_with_config(
            model_name=model,
            config_path=config_path,
            artifacts_root=artifacts_root,
        )

    build_oof_cfg = load_json_config(REPO_ROOT / "configs" / "build_oof.json")
    if args.oof_folds is not None:
        build_oof_cfg["folds"] = int(args.oof_folds)
    if args.oof_seed is not None:
        build_oof_cfg["seed"] = int(args.oof_seed)
    build_oof_cfg["models"] = []
    for model in requested_models:
        cfg_path = chosen_config_paths.get(model, default_train_config_path(model))
        model_params = _read_train_model_params(cfg_path)
        build_oof_cfg["models"].append(
            {
                "name": model,
                "weights_path": training_results[model]["weights_path"],
                "model_config": model_params,
            }
        )

    oof_run_dir = make_run_dir(artifacts_root, "build_oof")
    oof_outputs = build_oof_dataset(build_oof_cfg, oof_run_dir)
    update_latest_link(oof_run_dir)
    save_json(
        {
            "status": "success",
            "process": "build_oof",
            "run_id": oof_run_dir.name,
            "run_dir": str(oof_run_dir),
            "outputs": oof_outputs,
            "models": build_oof_cfg["models"],
        },
        oof_run_dir / "manifest.json",
    )

    oof_path = oof_outputs["merged"]
    mlp_template = load_json_config(REPO_ROOT / "configs" / "train_stack_mlp.json")
    ridge_template = load_json_config(REPO_ROOT / "configs" / "train_stack_ridge.json")
    xgboost_template = load_json_config(REPO_ROOT / "configs" / "train_stack_xgboost.json")
    meta_results: dict[str, Any] = {}
    best_meta_name: str | None = None
    best_meta_metric: float = -1e18

    for mode in requested_meta:
        stack_cfg = _meta_config_from_mode(
            mode=mode,
            oof_path=oof_path,
            selected_models=requested_models,
            mlp_template=mlp_template,
            ridge_template=ridge_template,
            xgboost_template=xgboost_template,
        )
        if mode == "mlp" and args.meta_epochs is not None:
            stack_cfg["epochs"] = int(args.meta_epochs)
        stack_run_dir = make_run_dir(artifacts_root, f"train_stack_{mode}")
        outputs = train_stack(stack_cfg, stack_run_dir)
        history = outputs.get("history")
        if isinstance(history, dict):
            save_json(history, stack_run_dir / "metrics_history.json")
            plot_history(history, stack_run_dir / "plots")
        if isinstance(outputs.get("feature_spec"), dict):
            save_json(outputs["feature_spec"], stack_run_dir / "stack_feature_spec.json")
        best_metric, best_epoch = _extract_best_metric(history if isinstance(history, dict) else None)
        manifest = {
            "status": "success",
            "process": f"train_stack_{mode}",
            "run_id": stack_run_dir.name,
            "run_dir": str(stack_run_dir),
            "weights_path": outputs.get("weights_path"),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "outputs": outputs,
            "used_models": requested_models,
            "oof_path": oof_path,
        }
        save_json(manifest, stack_run_dir / "manifest.json")
        update_latest_link(stack_run_dir)
        meta_results[mode] = manifest
        if best_metric is not None and best_metric > best_meta_metric:
            best_meta_metric = float(best_metric)
            best_meta_name = mode

    summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "base_models": requested_models,
        "meta_models": requested_meta,
        "skip_optuna": bool(args.skip_optuna),
        "skip_optuna_models": sorted(skip_optuna_models),
        "optuna": optuna_results,
        "selected_configs": {k: str(v) for k, v in chosen_config_paths.items()},
        "training": training_results,
        "build_oof": {
            "run_dir": str(oof_run_dir),
            "outputs": oof_outputs,
            "used_models": build_oof_cfg["models"],
        },
        "meta_training": meta_results,
        "best_meta_model": best_meta_name,
        "best_meta_metric": None if best_meta_name is None else best_meta_metric,
    }
    summary_path = Path(args.output_summary) if args.output_summary else run_dir / "pipeline_summary.json"
    save_json(summary, summary_path)
    logger.info("Pipeline completed. Summary: %s", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()
