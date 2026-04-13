# Wunder Contest — Time Series Forecasting

Repository with training, orchestration, stacking, and inference code for the Wunder Fund sequence forecasting task.

## Current status

This repo is in a **pipeline-oriented** state with:

- Unified Job API (`src/orchestration/jobs.py`) used by all CLI pipelines.
- Dedicated job handlers (`src/orchestration/handlers/`) for `base_train`, `build_oof`, `train_stack`, `train_meta`, and `optuna_tune`.
- Stable run artifacts (`artifacts/<process>/<run_id>/...`) with per-run `manifest.json`.
- Unit test coverage for orchestration, handlers, and edge cases in `tests/unit/test_orchestration_unit.py`.

## Competition context (short)

- Public leaderboard: approximately top ~18%.
- Metric: weighted Pearson-based score over targets `t0`, `t1` (see `src/metrics.py`).
- Inference design: paired models via `PairOrchestrator` (`lstm_gru` or `lstm_ssm`) with optional OOF-trained meta-head.

## Architecture

### Top-level layout

| Path | Role |
|---|---|
| `src/` | Core code: models, training, stacking, orchestration, inference, utils |
| `pipelines/` | CLI entrypoints used for local runs / Airflow-style execution |
| `configs/` | JSON configs for all training/tuning/stacking flows |
| `artifacts/` | Outputs: weights, plots, metrics history, manifests, config snapshots |
| `datasets/` | Expected location for `train.parquet` and `valid.parquet` |
| `tests/` | Unit tests (`unittest`) |

### Orchestration flow

1. `pipelines/*.py` parse args and build `JobSpec`.
2. `execute_job(...)` in `src/orchestration/jobs.py` creates run dir + snapshot.
3. Job is routed to one of `src/orchestration/handlers/*.py`.
4. Handler returns `(weights, metrics, outputs)`.
5. `manifest.json` is materialized and returned via `JobResult`.

### Job handlers

- `src/orchestration/handlers/base_train.py`
- `src/orchestration/handlers/build_oof.py`
- `src/orchestration/handlers/train_stack.py`
- `src/orchestration/handlers/train_meta.py`
- `src/orchestration/handlers/optuna_tune.py`

## Data layout

Put competition parquet files here:

- `datasets/train.parquet`
- `datasets/valid.parquet`

Feature schema is defined by constants in `src/constants.py` (`p*`, `v*`, `dp*`, `dv*`), targets are `t0`, `t1`.

## Main workflows

### 1) Train base models

```bash
python pipelines/train_conv_lstm.py --config configs/train_conv_lstm.json
python pipelines/train_gru.py --config configs/train_gru.json
python pipelines/train_ssm.py --config configs/train_ssm.json
python pipelines/train_transformer.py --config configs/train_transformer.json
```

### 2) Build OOF dataset

```bash
python pipelines/build_oof.py --config configs/build_oof.json
# optional single-model OOF
python pipelines/build_oof.py --config configs/build_oof.json --model conv_lstm
```

### 3) Train stack model

```bash
python pipelines/train_stack.py --config configs/train_stack_mlp.json
python pipelines/train_stack.py --config configs/train_stack_ridge.json
```

### 4) Run Optuna tuning / export

```bash
python pipelines/optuna_tune.py --model gru --config configs/train_gru.json --trials 30 --storage sqlite:///artifacts/optuna/optuna.db
```

### 5) Train meta-head for pair orchestrator

```bash
python pipelines/train_meta.py --config configs/train_meta_oof.json
```

## Inference notes

`src/solution.py` contains the contest-style `PredictionModel`.

By default, pair variant is `lstm_ssm`. To switch:

```bash
export ORCHESTRATION_VARIANT=lstm_gru
```

or explicitly keep default:

```bash
export ORCHESTRATION_VARIANT=lstm_ssm
```

The model expects weights/config layout referenced from `src/constants.py`.

## Artifacts contract

Each run writes into:

`artifacts/<process>/<run_id>/`

Typical contents:

- `config_snapshot/*.json` (or `config.json` when config path is inline)
- `metrics_history.json` (where applicable)
- `plots/*.png` (where applicable)
- `weights/*`
- `manifest.json`

See `docs/airflow_job_api.md` for detailed Job API and manifest contract.

## Testing

Current unit tests:

- `tests/unit/test_orchestration_unit.py`

Run tests (recommended through conda env):

```bash
conda run -n wunder-ts python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- `pipelines/*.py` currently prepend `src/` to `sys.path` so they work from repo root without `PYTHONPATH`.
- Large trained checkpoints may be excluded from git. Recreate them by running pipelines or copying your existing `artifacts/` layout.

---

## Installation and Usage Guide (Step-by-Step)

This section is intentionally practical and end-to-end.

### Step 0. Prerequisites

- Linux/macOS (Windows via WSL is recommended).
- Conda installed.
- Enough disk space for datasets and artifacts.

### Step 1. Clone and enter project

```bash
git clone <your-repo-url> wunder-contest
cd wunder-contest
```

### Step 2. Create / update environment

```bash
conda env create -f environment.yml
conda activate wunder-ts
```

If env already exists:

```bash
conda env update -f environment.yml --prune
```

### Step 3. Prepare data

Place files:

- `datasets/train.parquet`
- `datasets/valid.parquet`

### Step 4. Run baseline training flow

```bash
python pipelines/train_conv_lstm.py --config configs/train_conv_lstm.json
python pipelines/train_gru.py --config configs/train_gru.json
python pipelines/train_ssm.py --config configs/train_ssm.json
```

### Step 5. Build OOF and train stack

```bash
python pipelines/build_oof.py --config configs/build_oof.json
python pipelines/train_stack.py --config configs/train_stack_mlp.json
```

### Step 6. (Optional) Tune with Optuna

```bash
python pipelines/optuna_tune.py --model gru --config configs/train_gru.json --trials 30 --storage sqlite:///artifacts/optuna/optuna.db
```

### Step 7. Train meta-head

```bash
python pipelines/train_meta.py --config configs/train_meta_oof.json
```

### Step 8. Validate with unit tests

```bash
conda run -n wunder-ts python -m unittest discover -s tests -p "test_*.py"
```

### Step 9. Use for inference

Ensure required weights exist in paths referenced by `src/constants.py`, then integrate `PredictionModel` from `src/solution.py` in your host runtime.

### Troubleshooting quick tips

- If `python: command not found`, use `python3` or activate conda env.
- If `torch` import fails, verify environment activation and pip section install from `environment.yml`.
- If `optuna_tune` errors on base config path, pass a valid file path via config or CLI.
