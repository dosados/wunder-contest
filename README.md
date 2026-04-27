# Wunder Contest — Time Series Forecasting

Repository with training, stacking, tuning, and inference code for my solution of the Wunder Fund sequence forecasting task. 

### Competition overview

Competition task is to forecast market markers using time series data.

Data is a set of different market series, each snapshot in which includes many market features.

Key metric is weighted Pearson correlation of target columns to true values. 

Models inference are ran on single-thread cpu with 1 hour time limit, therefore large models such as transformers cannot be computed in time. 

### About my solution and results

My best solution was a stacking of models. It has lstm and gru models as base models and MLP as meta model.

It resulted a metric value of 0.289, which is top 18% of leaderboard.

### About this repository

This repository is a system, designed for a automatic tuning of included models, stacking them and evaluating resulted predictions.

Functions:
-Hyperparameters search with optuna for each model
-Building OOF datasets for stacking 
-Stacking chosen models with many different meta-heads
-Evaluation of predictions on validation holdout

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

## Main workflows

### 1) Train base models

```bash
python pipelines/train_conv_lstm.py
python pipelines/train_gru.py
python pipelines/train_ssm.py
python pipelines/train_transformer.py
```

When `--config` is omitted, each train pipeline first tries `configs/optuna_best_<model>.json`, and falls back to default `configs/train_<model>.json`.

### 2) Build OOF dataset

```bash
python pipelines/build_oof.py --config configs/build_oof.json
python pipelines/build_oof.py --config configs/build_oof.json --model conv_lstm
```

When `model_config` is missing for a model entry, `build_oof` resolves it from `optuna_best_<model>.json` if present, otherwise from default train config.

### 3) Train stack model

```bash
python pipelines/train_stack.py --config configs/train_stack_mlp.json
python pipelines/train_stack.py --config configs/train_stack_ridge.json
python pipelines/train_stack.py --config configs/train_stack_xgboost.json
```

### 4) Run Optuna tuning / export

```bash
python pipelines/optuna_tune.py --model gru --trials 30 
```

Use `--force-save` to always overwrite output config.
Without `--force-save`, config is overwritten only if the new Optuna `best_value` is higher than the saved one in config `_meta`.

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


## Installation and Usage Guide 


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
python pipelines/optuna_tune.py --model gru --config configs/train_gru.json --trials 30 
```

### Step 7. Train meta-head

```bash
python pipelines/train_meta.py --config configs/train_meta_oof.json
```

### Step 8. Validate with unit tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```



