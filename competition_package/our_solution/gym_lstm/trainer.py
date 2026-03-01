"""
Тренер второй LSTM на датасете остатков (аналог gym_ssm, но LSTM вместо SSM).

Читает train_residual.parquet и valid_residual.parquet (таргеты = остатки первой LSTM),
обучает ResidualLSTM с комбинированным лоссом: loss = LAMBDA_L * pearson_loss + (1 - LAMBDA_L) * mse.
Лучшие веса сохраняются в weights_lstm2/lstm2_model.pt для использования в lstm_check.py
(pred = LSTM1(x) + LSTM2(x)).

Запуск (из каталога our_solution):
  python -m gym_lstm.trainer
"""
import json
import logging
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUR_SOLUTION_DIR = os.path.dirname(CURRENT_DIR)
PKG_DIR = os.path.dirname(OUR_SOLUTION_DIR)
for d in (OUR_SOLUTION_DIR, PKG_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from constants import (
    DEVICE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    WARMUP_STEPS,
    PRED_CLIP_LOW,
    PRED_CLIP_HIGH,
    TRAIN_RESIDUAL_PATH,
    VALID_RESIDUAL_PATH,
    SEQUENCE_BATCH_SIZE,
    WEIGHTS_LSTM2_DIR,
    SAVE_NAME_LSTM2,
    EPOCHS_LSTM2,
    LR_LSTM2,
)
from dataset import ParquetSequenceDataset
from residual_lstm_model import ResidualLSTM

sys.path.insert(0, OUR_SOLUTION_DIR)
from trainer import train_epoch, validate, _masked_weighted_pearson_loss

# Вес Pearson в комбинированном лоссе: loss = LAMBDA_L * pearson_loss + (1 - LAMBDA_L) * mse (как в gym_ssm)
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
    """loss = l * pearson_loss + (1 - l) * mse."""
    pearson_loss = _masked_weighted_pearson_loss(pred, target, warmup_steps)
    mse = _masked_mse(pred, target, warmup_steps)
    return l * pearson_loss + (1.0 - l) * mse


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

NUM_RUNS = 3


def run_gym_lstm(num_runs=NUM_RUNS, epochs=None):
    if epochs is None:
        epochs = EPOCHS_LSTM2

    run_folder = f"run_{datetime.now():%d%m%y_%H%M%S}"
    run_dir = os.path.join(WEIGHTS_LSTM2_DIR, run_folder)
    os.makedirs(run_dir, exist_ok=True)
    lstm2_filename = os.path.basename(SAVE_NAME_LSTM2)
    best_weights_path = os.path.join(run_dir, lstm2_filename)
    default_weights_path = os.path.join(WEIGHTS_LSTM2_DIR, lstm2_filename)
    config_path = os.path.join(run_dir, "config.json")

    run_config = {
        "lstm2_constants": _get_lstm2_config(),
        "train_data": "train_residual.parquet",
        "valid_data": "valid_residual.parquet",
    }
    os.makedirs(WEIGHTS_LSTM2_DIR, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    log.info("Run dir: %s", run_dir)
    log.info("Device: %s", DEVICE)
    log.info("Runs: %d, Epochs: %d", num_runs, epochs)

    if not os.path.isfile(TRAIN_RESIDUAL_PATH):
        log.error(
            "Не найден датасет остатков: %s. Сначала запустите build_residual_dataset.py",
            TRAIN_RESIDUAL_PATH,
        )
        return

    train_ds = ParquetSequenceDataset(
        TRAIN_RESIDUAL_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    total_train_batches = len(train_loader)

    val_loader = None
    total_val_batches = None
    if os.path.isfile(VALID_RESIDUAL_PATH):
        val_ds = ParquetSequenceDataset(
            VALID_RESIDUAL_PATH,
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
        log.info(
            "Train: %s (%d seq), Valid: %s (%d seq)",
            TRAIN_RESIDUAL_PATH,
            len(train_ds),
            VALID_RESIDUAL_PATH,
            len(val_ds),
        )
    else:
        log.warning("Valid residual не найден: %s; валидация по train", VALID_RESIDUAL_PATH)

    criterion = lambda pred, y: _combined_loss(pred, y, WARMUP_STEPS, LAMBDA_L)
    best_val_pearson_global = -float("inf")

    for run in range(1, num_runs + 1):
        log.info("=== Run %d / %d ===", run, num_runs)
        model = ResidualLSTM().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_LSTM2)
        best_val_pearson_run = -float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                DEVICE,
                epoch=epoch,
                total_epochs=epochs,
                total_batches=total_train_batches,
            )

            if val_loader is not None:
                val_loss, val_pearson = validate(
                    model,
                    val_loader,
                    criterion,
                    DEVICE,
                    total_val_batches=total_val_batches,
                    epoch=epoch,
                    total_epochs=epochs,
                )
                log.info(
                    "Epoch %d | train_loss: %.6f | val_loss: %.6f | val pearson (residual): %.6f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_pearson,
                )
                if val_pearson > best_val_pearson_run:
                    best_val_pearson_run = val_pearson
                if val_pearson > best_val_pearson_global:
                    best_val_pearson_global = val_pearson
                    torch.save(model.state_dict(), best_weights_path)
                    torch.save(model.state_dict(), default_weights_path)
                    _save_default_config(WEIGHTS_LSTM2_DIR)
                    log.info(
                        "New best weights → %s (val pearson: %.6f)",
                        default_weights_path,
                        val_pearson,
                    )
            else:
                _, train_pearson = validate(
                    model,
                    DataLoader(
                        train_ds,
                        batch_size=SEQUENCE_BATCH_SIZE,
                        shuffle=False,
                        num_workers=0,
                    ),
                    criterion,
                    DEVICE,
                    total_val_batches=len(train_loader),
                    epoch=epoch,
                    total_epochs=epochs,
                )
                log.info(
                    "Epoch %d | train_loss: %.6f | train pearson: %.6f",
                    epoch,
                    train_loss,
                    train_pearson,
                )
                if train_pearson > best_val_pearson_run:
                    best_val_pearson_run = train_pearson
                if train_pearson > best_val_pearson_global:
                    best_val_pearson_global = train_pearson
                    torch.save(model.state_dict(), best_weights_path)
                    torch.save(model.state_dict(), default_weights_path)
                    _save_default_config(WEIGHTS_LSTM2_DIR)
                    log.info(
                        "New best weights → %s (train pearson: %.6f)",
                        default_weights_path,
                        train_pearson,
                    )

        log.info(
            "Run %d finished, best val pearson this run: %.6f",
            run,
            best_val_pearson_run,
        )

    log.info(
        "All runs finished. Best val pearson (on residual): %.6f. Weights: %s",
        best_val_pearson_global,
        default_weights_path,
    )


def _get_lstm2_config():
    import constants
    return getattr(constants, "lstm2_constants", {}).copy()


def _save_default_config(weights_dir: str) -> None:
    """Сохраняет текущий lstm2_constants в weights_dir/config.json."""
    config_path = os.path.join(weights_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"lstm2_constants": _get_lstm2_config()}, f, indent=2, ensure_ascii=False)


def main():
    run_gym_lstm(num_runs=NUM_RUNS, epochs=EPOCHS_LSTM2)


if __name__ == "__main__":
    main()
