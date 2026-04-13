from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestration.job_contracts import JobSpec
from training.tasks import run_training_job
from utils import plot_history, save_json


def run_base_train(
    spec: JobSpec, run_dir: Path
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    result, _ = run_training_job(
        spec.config,
        spec.process_name,
        run_dir=run_dir,
        snapshot_original_config=None,
    )
    save_json(result.history, run_dir / "metrics_history.json")
    plot_history(result.history, run_dir / "plots")
    return (
        {"best": result.best_weights_path},
        {"best_metric": result.best_metric, "best_epoch": result.best_epoch},
        {
            "best_metric": result.best_metric,
            "best_epoch": result.best_epoch,
            "weights_path": result.best_weights_path,
        },
    )
