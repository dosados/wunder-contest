from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from constants import ARTIFACTS_ROOT
from orchestration.handlers import (
    run_base_train,
    run_build_oof,
    run_optuna_tune,
    run_train_meta,
    run_train_stack,
)
from orchestration.job_contracts import (
    MANIFEST_SCHEMA_VERSION,
    JobResult,
    JobSpec,
    JobType,
    RunManifest,
    config_hash,
)
from utils import get_logger, make_run_dir, save_json, snapshot_config, update_latest_link


def _ensure_run_dir(
    artifacts_root: str | Path,
    process_name: str,
    run_id: str | None,
    *,
    allow_overwrite: bool,
) -> Path:
    if run_id:
        run_dir = Path(artifacts_root) / process_name / run_id
        if run_dir.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Use a new run_id or allow overwrite."
            )
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (run_dir / "config_snapshot").mkdir(parents=True, exist_ok=True)
        return run_dir
    return make_run_dir(artifacts_root, process_name)


def _materialize_manifest(
    *,
    spec: JobSpec,
    run_dir: Path,
    weights: dict[str, str],
    metrics: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[RunManifest, Path]:
    run_id = run_dir.name
    manifest = RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        status="success",
        job_type=spec.job_type,
        process_name=spec.process_name,
        run_id=run_id,
        run_dir=str(run_dir),
        config_hash=config_hash(spec.config),
        config_path=str(Path(spec.config_path).resolve()) if spec.config_path else None,
        weights=weights,
        metrics=metrics,
        outputs=outputs,
        inputs={"config": spec.config},
        metadata=spec.metadata,
    )
    manifest_path = (
        Path(spec.output_manifest)
        if spec.output_manifest is not None
        else run_dir / "manifest.json"
    )
    save_json(manifest.to_dict(), manifest_path)
    return manifest, manifest_path


def _save_config_snapshot(spec: JobSpec, run_dir: Path) -> None:
    if spec.config_path:
        snapshot_config(spec.config_path, run_dir)
    else:
        save_json(spec.config, run_dir / "config.json")


_RUNNERS: dict[str, Callable[[JobSpec, Path], tuple[dict[str, str], dict[str, Any], dict[str, Any]]]] = {
    "base_train": run_base_train,
    "build_oof": run_build_oof,
    "train_stack": run_train_stack,
    "train_meta": run_train_meta,
    "optuna_tune": run_optuna_tune,
}


def execute_job(spec: JobSpec) -> JobResult:
    logger = get_logger("jobs")
    artifacts_root = Path(spec.artifacts_root or ARTIFACTS_ROOT)
    allow_overwrite = bool(spec.metadata.get("allow_overwrite_run_dir", False))
    run_dir = _ensure_run_dir(
        artifacts_root,
        spec.process_name,
        spec.run_id,
        allow_overwrite=allow_overwrite,
    )
    _save_config_snapshot(spec, run_dir)
    if spec.job_type not in _RUNNERS:
        raise ValueError(f"Unsupported job type: {spec.job_type}")
    weights, metrics, outputs = _RUNNERS[spec.job_type](spec, run_dir)
    manifest, manifest_path = _materialize_manifest(
        spec=spec, run_dir=run_dir, weights=weights, metrics=metrics, outputs=outputs
    )
    if spec.write_latest_link:
        update_latest_link(run_dir)
    logger.info(
        "Job %s finished run_id=%s manifest=%s",
        spec.job_type,
        manifest.run_id,
        manifest_path,
    )
    return JobResult(
        job_type=spec.job_type,
        status=manifest.status,
        run_id=manifest.run_id,
        run_dir=manifest.run_dir,
        manifest_path=str(manifest_path),
        weights_path=next(iter(manifest.weights.values()), None),
        best_metric=manifest.metrics.get("best_metric"),
        best_epoch=manifest.metrics.get("best_epoch"),
        outputs=manifest.outputs,
        metrics=manifest.metrics,
    )


def build_job_spec(
    *,
    job_type: JobType,
    process_name: str,
    config: dict[str, Any],
    config_path: str | Path | None = None,
    run_id: str | None = None,
    artifacts_root: str | Path | None = None,
    output_manifest: str | Path | None = None,
    write_latest_link: bool = False,
) -> JobSpec:
    return JobSpec(
        job_type=job_type,
        process_name=process_name,
        config=config,
        config_path=config_path,
        run_id=run_id,
        artifacts_root=artifacts_root,
        output_manifest=output_manifest,
        write_latest_link=write_latest_link,
    )
