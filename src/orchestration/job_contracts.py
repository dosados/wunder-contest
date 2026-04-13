from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal


MANIFEST_SCHEMA_VERSION = "1.0"
JobType = Literal["base_train", "optuna_tune", "build_oof", "train_stack", "train_meta"]
JobStatus = Literal["success", "failed"]


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JobSpec:
    job_type: JobType
    process_name: str
    config: dict[str, Any]
    config_path: str | Path | None = None
    artifacts_root: str | Path | None = None
    run_id: str | None = None
    output_manifest: str | Path | None = None
    write_latest_link: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    schema_version: str
    status: JobStatus
    job_type: JobType
    process_name: str
    run_id: str
    run_dir: str
    config_hash: str
    config_path: str | None = None
    weights: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "job_type": self.job_type,
            "process_name": self.process_name,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "config_hash": self.config_hash,
            "config_path": self.config_path,
            "weights": self.weights,
            "metrics": self.metrics,
            "outputs": self.outputs,
            "inputs": self.inputs,
            "metadata": self.metadata,
        }


@dataclass
class JobResult:
    job_type: JobType
    status: JobStatus
    run_dir: str
    manifest_path: str
    run_id: str
    weights_path: str | None = None
    best_metric: float | None = None
    best_epoch: int | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_type": self.job_type,
            "status": self.status,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "manifest_path": self.manifest_path,
            "weights_path": self.weights_path,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "outputs": self.outputs,
            "metrics": self.metrics,
        }
