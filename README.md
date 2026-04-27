# Wunder Contest — Time Series Forecasting

This repository contains training, stacking, tuning, and inference code for my solution to the Wunder Fund sequence-forecasting task.

### Competition overview

The competition task is to forecast market indicators from time-series data.

The dataset contains multiple market series, and each snapshot includes many market features.

The key metric is the weighted Pearson correlation between predicted target columns and ground truth values.

Inference is executed on a single-threaded CPU with a 1-hour time limit, so large models (for example, heavy transformers) are not practical.

### About my solution and results

My best solution is a stacked ensemble: LSTM and GRU base models with an MLP meta-model.

It achieved a metric value of `0.289`, which is in the top 18% of the leaderboard.

### About this repository

This repository is a system designed for automatic model tuning, stacking, and evaluation of resulting predictions.

Key capabilities:
- Hyperparameter search with Optuna for each model
- Building OOF datasets for stacking
- Training stack models with different meta-heads
- Evaluating predictions on a validation holdout

# Architecture

### Top-level layout

| Path | Role |
|---|---|
| `src/` | Core code: models, training, stacking, inference, utils |
| `pipelines/` | CLI entrypoints used for local runs |
| `configs/` | JSON configs for all training/tuning/stacking flows |
| `artifacts/` | Outputs: weights, plots, metrics history, manifests, config snapshots |
| `datasets/` | Expected location for `train.parquet` and `valid.parquet` |
| `tests/` | Unit tests (`unittest`) |


### Data layout

Put competition parquet files here:

- `datasets/train.parquet`
- `datasets/valid.parquet`

Feature schema is defined by constants in `src/constants.py` (`p*`, `v*`, `dp*`, `dv*`), targets are `t0`, `t1`.

## Workflows

### Entrypoints

| Entrypoint | Purpose |
|---|---|
| `pipelines/optuna_tune.py` | Runs Optuna for one base model and exports `configs/optuna_best_<model>.json` |
| `pipelines/train_conv_lstm.py` | Trains Conv-LSTM base model |
| `pipelines/train_gru.py` | Trains GRU base model |
| `pipelines/train_ssm.py` | Trains SSM base model |
| `pipelines/train_transformer.py` | Trains Transformer base model |
| `pipelines/build_oof.py` | Builds OOF predictions for selected base models |
| `pipelines/train_stack.py` | Trains stack/meta model (`mlp`, `ridge`, `xgboost`) on OOF features |
| `pipelines/train_meta.py` | Trains pair-orchestrator meta-head over OOF-derived inputs |
| `pipelines/run_full_pipeline.py` | Runs end-to-end flow: optuna -> base train -> build_oof -> meta train |

### 1) Train base models

```bash
python pipelines/train_conv_lstm.py
python pipelines/train_gru.py
python pipelines/train_ssm.py
python pipelines/train_transformer.py
```

When `--config` is omitted, each train pipeline first tries `configs/optuna_best_<model>.json`, and falls back to default `configs/train_<model>.json`.

### 2) Run Optuna tuning / export

```bash
python pipelines/optuna_tune.py --model gru --trials 30
```

Use `--force-save` to always overwrite the output config.
Without `--force-save`, the config is overwritten only if the new Optuna `best_value` is higher than the saved one in config `_meta`.

### 3) Build OOF dataset

```bash
python pipelines/build_oof.py --config configs/build_oof.json
python pipelines/build_oof.py --config configs/build_oof.json --model conv_lstm
```

When `model_config` is missing for a model entry, `build_oof` resolves it from `optuna_best_<model>.json` if present, otherwise from default train config.

### 4) Train stack model

```bash
python pipelines/train_stack.py --config configs/train_stack_mlp.json
python pipelines/train_stack.py --config configs/train_stack_ridge.json
python pipelines/train_stack.py --config configs/train_stack_xgboost.json
```

### 5) Train meta-head for pair orchestrator

```bash
python pipelines/train_meta.py --config configs/train_meta_oof.json
```

### 6) Run full end-to-end pipeline

```bash
python pipelines/run_full_pipeline.py \
  --models conv_lstm,gru,ssm \
  --meta-models ridge,mlp \
  --trials-per-model 20 \
  --skip-optuna-models ssm \
  --force-save
```

The script runs `optuna -> base train -> build_oof -> meta train`, supports model subsets, and writes `pipeline_summary.json`.
`build_oof` receives both `weights_path` and `model_config` from the selected best configs so OOF is built with tuned parameters.
For skipped models it reuses `configs/optuna_best_<model>.json`; if missing, it falls back to default train config.




## Artifacts contract

Each run writes into:

`artifacts/<process>/<run_id>/`

Typical contents:

- `config_snapshot/*.json` (or `config.json` when config path is inline)
- `metrics_history.json` (where applicable)
- `weights/*`
- `manifest.json`


## Installation  


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

### Step 3. Prepare data

Place files:

- `datasets/train.parquet`
- `datasets/valid.parquet`
