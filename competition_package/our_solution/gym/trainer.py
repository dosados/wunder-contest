"""
Тренер в gym: несколько прогонов обучения с конфигом из inference/config.json,
лучшие веса в gym/best, после каждой эпохи — метрика на валидации и на 20% обучающей выборки.

Обучение идёт только на одной половине train.parquet (по границе целых последовательностей).
Какую половину использовать задаётся константой TRAIN_DATA_PART (см. также gym/README.md).
"""
import json
import os
import sys
import logging
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Путь к our_solution и competition_package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUR_SOLUTION_DIR = os.path.dirname(CURRENT_DIR)
PKG_DIR = os.path.dirname(OUR_SOLUTION_DIR)
for d in (OUR_SOLUTION_DIR, PKG_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

# Загружаем конфиг и патчим константы ДО импорта модели
INFERENCE_CONFIG_PATH = os.path.join(OUR_SOLUTION_DIR, "inference", "config.json")
with open(INFERENCE_CONFIG_PATH, "r", encoding="utf-8") as f:
    _inference_config = json.load(f)
_best_hp = _inference_config["best_hyperparameters"]

import constants
constants.convo_constants = _best_hp

from utils import weighted_pearson_correlation
from constants import (
    DEVICE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    WARMUP_STEPS,
    TRAIN_PATH,
    VAL_PATH,
    SEQUENCE_BATCH_SIZE,
    EPOCHS,
    LR,
    PRED_CLIP_LOW,
    PRED_CLIP_HIGH,
    WEIGHT_EPS,
    VAR_EPS,
)
from model import FullModel
from dataset import ParquetSequenceDataset

# Импорты функций обучения из основного trainer (модель уже с патчем констант)
sys.path.insert(0, OUR_SOLUTION_DIR)
from trainer import train_epoch, validate, _masked_weighted_pearson_loss

# Вес Pearson в комбинированном лоссе: loss = LAMBDA_L * pearson_loss + (1 - LAMBDA_L) * mse
LAMBDA_L = 0.8


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, warmup_steps: int) -> torch.Tensor:
    """MSE по шагам [warmup_steps, T). pred/target: (B, T, 2). pred клипается как в pearson."""
    if warmup_steps > 0:
        pred = pred[:, warmup_steps:]
        target = target[:, warmup_steps:]
    pred = torch.clamp(pred, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    return ((pred - target) ** 2).mean()


def _combined_loss(
    pred: torch.Tensor, target: torch.Tensor, warmup_steps: int, l: float
) -> torch.Tensor:
    """loss = l * pearson_loss + (1 - l) * mse (только внутри gym)."""
    pearson_loss = _masked_weighted_pearson_loss(pred, target, warmup_steps)
    mse = _masked_mse(pred, target, warmup_steps)
    return l * pearson_loss + (1.0 - l) * mse


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --- Разбивка train.parquet по половинам (по границе seq_ix) ---
# В parquet последовательности идут подряд: seq_0 (1000 строк), seq_1 (1000 строк), ...
# Индекс в ParquetSequenceDataset = номер последовательности. Половина — по целым последовательностям.
# Чтобы обучать модель на второй половине, смените на "second_half" (см. gym/README.md).
TRAIN_DATA_PART = "first_half"   # "first_half" | "second_half"

# Базовый каталог для лучших весов (внутри создаётся run_DDMMYY_HHMMSS)
GYM_BEST_DIR = os.path.join(CURRENT_DIR, "best" if TRAIN_DATA_PART == "first_half" else "best_second_half")

# Доля обучающей выборки для отчёта метрики (20% от той половины, на которой учим)
TRAIN_METRIC_FRACTION = 0.2
# Количество прогонов обучения
NUM_RUNS = 3
# Seed для воспроизводимости 20% сэмпла
TRAIN_20_SEED = 42


def _train_half_indices(n_sequences: int):
    """Индексы последовательностей для выбранной половины (граница по началу новой seq)."""
    half = n_sequences // 2
    if TRAIN_DATA_PART == "first_half":
        return list(range(0, half)), 0, half
    if TRAIN_DATA_PART == "second_half":
        return list(range(half, n_sequences)), half, n_sequences
    raise ValueError(f"TRAIN_DATA_PART must be 'first_half' or 'second_half', got {TRAIN_DATA_PART!r}")


def _make_train_20_loader(part_ds, part_size: int, batch_size=SEQUENCE_BATCH_SIZE):
    """DataLoader на 20% от part_ds (той половины, на которой учим; фиксированный seed)."""
    n = part_size
    k = max(1, int(n * TRAIN_METRIC_FRACTION))
    rng = random.Random(TRAIN_20_SEED)
    indices = rng.sample(range(n), k)
    subset = Subset(part_ds, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def run_gym(num_runs=NUM_RUNS, epochs=None):
    if epochs is None:
        epochs = EPOCHS

    run_folder = f"run_{datetime.now():%d%m%y_%H%M%S}"
    run_dir = os.path.join(GYM_BEST_DIR, run_folder)
    os.makedirs(run_dir, exist_ok=True)
    best_weights_path = os.path.join(run_dir, "best_model.pt")
    config_path = os.path.join(run_dir, "config.json")
    run_config = {
        "model": _inference_config.get("best_hyperparameters", _best_hp),
        "train_data_part": TRAIN_DATA_PART,
        "source_config": INFERENCE_CONFIG_PATH,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    log.info("Run dir: %s (config saved to %s)", run_dir, config_path)
    log.info("Config from %s: %s", INFERENCE_CONFIG_PATH, _best_hp)
    log.info("Device: %s", DEVICE)
    log.info("Runs: %d, Epochs per run: %d", num_runs, epochs)

    if not os.path.isfile(TRAIN_PATH):
        log.error("Train file not found: %s", TRAIN_PATH)
        return

    train_ds_full = ParquetSequenceDataset(
        TRAIN_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    n_sequences = len(train_ds_full)
    train_indices, seq_start, seq_end = _train_half_indices(n_sequences)
    train_ds = Subset(train_ds_full, train_indices)
    part_size = len(train_indices)
    log.info(
        "Train data: %s of train.parquet — sequences [%d, %d) (total %d sequences)",
        TRAIN_DATA_PART, seq_start, seq_end, part_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    train_20_loader = _make_train_20_loader(train_ds, part_size)
    total_train_batches = len(train_loader)
    total_train_20_batches = len(train_20_loader)

    val_loader = None
    total_val_batches = None
    if VAL_PATH and os.path.isfile(VAL_PATH):
        val_ds = ParquetSequenceDataset(
            VAL_PATH,
            feature_columns=FEATURE_COLUMNS,
            target_columns=TARGET_COLUMNS,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        total_val_batches = len(val_loader)
        log.info("Validation from %s", VAL_PATH)
    else:
        log.warning("No validation file; will save last weights to gym/best")

    criterion = lambda pred, y: _combined_loss(pred, y, WARMUP_STEPS, LAMBDA_L)
    best_val_pearson_global = -float("inf")

    for run in range(1, num_runs + 1):
        log.info("=== Run %d / %d ===", run, num_runs)
        model = FullModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_val_pearson_run = -float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, DEVICE,
                epoch=epoch, total_epochs=epochs,
                total_batches=total_train_batches,
            )

            # Метрика на 20% обучающей выборки
            _, train_20_pearson = validate(
                model, train_20_loader, criterion, DEVICE,
                total_val_batches=total_train_20_batches,
                epoch=epoch, total_epochs=epochs,
            )

            if val_loader is not None:
                val_loss, val_pearson = validate(
                    model, val_loader, criterion, DEVICE,
                    total_val_batches=total_val_batches,
                    epoch=epoch, total_epochs=epochs,
                )
                log.info(
                    "Epoch %d | train_loss: %.6f | train_20%% pearson: %.6f | val_loss: %.6f | val pearson: %.6f",
                    epoch, train_loss, train_20_pearson, val_loss, val_pearson,
                )
                if val_pearson > best_val_pearson_run:
                    best_val_pearson_run = val_pearson
                if val_pearson > best_val_pearson_global:
                    best_val_pearson_global = val_pearson
                    torch.save(model.state_dict(), best_weights_path)
                    log.info("New best weights saved to %s (val pearson: %.6f)", best_weights_path, val_pearson)
            else:
                log.info(
                    "Epoch %d | train_loss: %.6f | train_20%% pearson: %.6f",
                    epoch, train_loss, train_20_pearson,
                )
                # Без валидации сохраняем по метрике на 20% train
                if train_20_pearson > best_val_pearson_run:
                    best_val_pearson_run = train_20_pearson
                if train_20_pearson > best_val_pearson_global:
                    best_val_pearson_global = train_20_pearson
                    torch.save(model.state_dict(), best_weights_path)
                    log.info("New best weights saved to %s (train_20 pearson: %.6f)", best_weights_path, train_20_pearson)

        log.info("Run %d finished, best val pearson this run: %.6f", run, best_val_pearson_run)

    log.info("All runs finished. Best val pearson overall: %.6f. Weights in %s", best_val_pearson_global, best_weights_path)


def main():
    run_gym(num_runs=NUM_RUNS, epochs=EPOCHS)


if __name__ == "__main__":
    main()
