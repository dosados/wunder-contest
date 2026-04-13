from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestration.job_contracts import JobSpec
from stacking import train_stack
from utils import plot_history, save_json


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


def run_train_stack(
    spec: JobSpec, run_dir: Path
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    result = train_stack(spec.config, run_dir)
    history = result.get("history")
    if isinstance(history, dict):
        save_json(history, run_dir / "metrics_history.json")
        plot_history(history, run_dir / "plots")
    feature_spec = result.get("feature_spec")
    if isinstance(feature_spec, dict):
        save_json(feature_spec, run_dir / "stack_feature_spec.json")
    weights_path = result.get("weights_path")
    best_metric, best_epoch = _extract_best_metric(history if isinstance(history, dict) else None)
    metrics = {"best_metric": best_metric}
    if best_epoch is not None:
        metrics["best_epoch"] = best_epoch
    return (
        {"stack": weights_path} if isinstance(weights_path, str) else {},
        metrics,
        result,
    )
