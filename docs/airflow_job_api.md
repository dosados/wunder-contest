# Unified Job API for Airflow

This project exposes a single orchestration layer in `src/orchestration/jobs.py`.

## Job types

- `base_train`
- `optuna_tune`
- `build_oof`
- `train_stack`
- `train_meta`

Each job is represented by `JobSpec` and returns `JobResult` (`src/orchestration/job_contracts.py`).

## Stable output contract

Every run writes `manifest.json` using `RunManifest` schema version `1.0`.

Required manifest fields:

- `schema_version`
- `status`
- `job_type`
- `process_name`
- `run_id`
- `run_dir`
- `config_hash`
- `weights`
- `metrics`
- `outputs`
- `inputs`

## DAG task IO contract

Use immutable run references (`run_id`, `run_dir`, `manifest_path`) between tasks.
Do not rely on `latest` symlink in Airflow tasks.

Recommended chain:

1. `optuna_tune` (compute)
2. `optuna_tune` with `export_only=true` (optional when study and export are separate tasks)
3. `base_train` using exported train-ready config
4. `build_oof` with explicit `weights_path` values from upstream manifests
5. `train_stack`

## Idempotency

- Run id must come from the system. In CLI wrappers it is always auto-generated.
- In Airflow Python tasks, use system context values (for example `dag_run.run_id` or `ts_nodash`) when you need deterministic ids.
- Existing run directory causes failure by default.
- To force replacement, pass `metadata={"allow_overwrite_run_dir": true}` in `JobSpec`.

## Airflow usage patterns

### PythonOperator (recommended)

Import `build_job_spec` and `execute_job`, build a spec in task callable, return `result.to_dict()`.

### BashOperator / CLI wrappers

Each script in `pipelines/` supports:

- `--config`
- `--artifacts-root`
- `--output-manifest`

and prints manifest path to stdout.

## Optuna split execution

`optuna_tune` job accepts:

- full tuning mode (`export_only=false`)
- export-only mode (`export_only=true`) with required `study_name` and `storage`

This allows deterministic `study -> export_best` decomposition across separate Airflow tasks.
