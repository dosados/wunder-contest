# Wunder Contest — Time Series Forecasting

This repository contains the solution and training code for the **Wunder Fund** time-series prediction competition (Predictorium-style setup: streaming state, two targets, strict inference constraints).

## Competition results

- **Leaderboard:** approximately **top ~18%** on the overall (public) leaderboard.
- **Contest metric:** **28.9** (official objective: mean of two weighted Pearson correlations between predictions and targets; see `src/metrics.py`).
- **Modeling:** **LSTM** (conv + LSTM backbone) **orchestrated with GRU** — a `PairOrchestrator` runs both towers and either sums their outputs or applies a small meta-head trained on out-of-fold predictions.
- **Inference constraint (per rules):** full test inference had to finish within **one hour** on a **single-threaded CPU** (no GPU, no multi-core parallelism for the submission path).

## What this project is today

Main ML code lives in `src/`; runnable workflows are the scripts under `pipelines/`. The layout is a refactored, pipeline-oriented version of the competition stack:

| Area | Role |
|------|------|
| **`src/`** | Importable package: parquet sequence datasets, Conv+LSTM, GRU, and SSM (structured state-space) models, training loop, contest metric, OOF generation, stack trainers (MLP / Ridge), orchestration, and the `PredictionModel` used for inference. |
| **`pipelines/`** | CLI entrypoints for training base models, building OOF tables, and training the stack / meta head. |
| **`configs/`** | JSON configs for each pipeline. |
| **`artifacts/`** | Run outputs: config snapshots, `metrics_history.json`, plots, weights, `manifest.json`, and `latest` symlinks where applicable. |
| **`datasets/`** | Expected location for `train.parquet` and `valid.parquet` (and optional residual files for SSM-style training). |

**Capabilities:**

- Train **Conv+LSTM**, **GRU**, and **SSM** models on long sequences from Parquet.
- Build **out-of-fold (OOF)** prediction datasets for stacking across models.
- Train a **second-level stack**: **MLP** or **sklearn Ridge** on OOF features.
- Run **inference** via `PredictionModel` in `src/solution.py`: combines LSTM with either GRU (`lstm_gru`) or SSM (`lstm_ssm`), optionally loading a trained meta-head if present on disk.

Default orchestration variant is **`lstm_ssm`**; set the environment variable below to match the competition **LSTM + GRU** setup.

---

## Setup

Dependencies are listed in **`environment.yml`** (Conda environment `wunder-ts`: Python 3.10; matplotlib, pyarrow, scikit-learn, tqdm from Conda channels; **PyTorch** / torchvision / torchaudio and NVIDIA CUDA wheels via `pip`).

```bash
cd /path/to/wunder-contest
conda env create -f environment.yml
conda activate wunder-ts
```

To refresh an existing environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

`pipelines/*.py` prepends `src/` to `sys.path`, so you do not need `PYTHONPATH` when running `python pipelines/...`.

---

## Data layout

Place competition-style data under `datasets/`:

- `datasets/train.parquet`
- `datasets/valid.parquet`

Column layout matches `src/constants.py`: features `p0`–`p11`, `v0`–`v11`, `dp0`–`dp3`, `dv0`–`dv3` (32 dims), targets `t0`, `t1`.

---

## Usage

### 1. Train base models

```bash
python pipelines/train_conv_lstm.py --config configs/train_conv_lstm.json
python pipelines/train_gru.py       --config configs/train_gru.json
python pipelines/train_ssm.py        --config configs/train_ssm.json
```

Each run writes under `artifacts/<process>/<run_id>/` (weights, metrics, plots). Update `configs/build_oof.json` if your weight paths differ from `artifacts/.../latest/...`.

### 2. Build OOF dataset for stacking

```bash
python pipelines/build_oof.py --config configs/build_oof.json
# Optional: single model
python pipelines/build_oof.py --config configs/build_oof.json --model conv_lstm
```

### 3. Train stack (MLP or Ridge)

```bash
python pipelines/train_stack.py --config configs/train_stack_mlp.json
# or
python pipelines/train_stack.py --config configs/train_stack_ridge.json
```

Point `oof_path` in the stack config at the merged OOF parquet produced by `build_oof` (e.g. `artifacts/build_oof/latest/oof/oof_stack_train.parquet`).

### 4. Inference layout (competition-style `PredictionModel`)

`src/solution.py` expects trained weights and configs under paths defined in `src/constants.py`, including:

- LSTM: `artifacts/inference/weights/<run_dir>/` (`best_model.pt`, `config.json`)
- GRU: `artifacts/inference_gru/weights/best_model.pt` and `config.json`
- SSM (default pair): `artifacts/weights_ssm/ssm_model.pt`
- Optional meta-head: `artifacts/stack/lstm_gru/meta_head_oof.pt` or `artifacts/stack/lstm_ssm/meta_head_oof.pt`

**LSTM + GRU** (competition orchestration):

```bash
export ORCHESTRATION_VARIANT=lstm_gru
```

**LSTM + SSM** (default in code):

```bash
export ORCHESTRATION_VARIANT=lstm_ssm
# or omit — this is the default
```

Integrate `PredictionModel` from `src/solution.py` with the host that provides `data_point` objects (`state`, `seq_ix`, `need_prediction`), as in the original competition API.

---

## Project structure

### CLI entry points

- `pipelines/train_conv_lstm.py`
- `pipelines/train_gru.py`
- `pipelines/train_ssm.py`
- `pipelines/build_oof.py` — full model list or a single model via `--model`
- `pipelines/train_stack.py` — second-level model from config (`mlp` or `ridge`)

### Artifact layout

Each run writes under `artifacts/<process>/<run_id>/` with:

- `config_snapshot/*.json`
- `metrics_history.json` (where applicable)
- `plots/*.png`
- `weights/*.pt` or `weights/ridge.joblib`
- `manifest.json`

---

## Note on weights

Trained checkpoints and competition submission bundles are large and may not be committed. After cloning, either train with the pipelines above or copy your saved `artifacts/` layout so paths in `src/constants.py` resolve.
