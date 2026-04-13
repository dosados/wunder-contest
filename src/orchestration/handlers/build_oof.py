from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestration.job_contracts import JobSpec
from stacking import build_oof_dataset


def run_build_oof(
    spec: JobSpec, run_dir: Path
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    outputs = build_oof_dataset(spec.config, run_dir)
    return ({}, {}, outputs)
