from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestration.job_contracts import JobSpec
from orchestration.train_meta_oof import train_meta_oof


def run_train_meta(
    spec: JobSpec, run_dir: Path
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    result = train_meta_oof(spec.config, run_dir)
    score = float(result["score"])
    return (
        {"meta_head": result["weights_path"]},
        {"best_metric": score},
        result,
    )
