"""
Простой трейнер: обучение FullModel на ParquetDataset с валидацией и сохранением весов.
"""
import os
import sys
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# импорты из our_solution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# импорт utils из competition_package
PKG_DIR = os.path.join(CURRENT_DIR, "..")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
from utils import weighted_pearson_correlation

from constants import (
    DEVICE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    WARMUP_STEPS,
    WEIGHTS_DIR,
    TRAIN_PATH,
    VAL_PATH,
    SEQUENCE_BATCH_SIZE,
    EPOCHS,
    LR,
    SAVE_NAME,
    PRED_CLIP_LOW,
    PRED_CLIP_HIGH,
    WEIGHT_EPS,
    VAR_EPS,
)
from model import FullModel
from dataset import ParquetSequenceDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _weighted_pearson_1d(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Взвешенный Pearson для одной 1D выборки (дифференцируемый).
    Веса = |y_true|, предсказания клипаются в [-6, 6]. Возвращает скаляр корреляции.
    """
    y_pred = torch.clamp(y_pred, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    weights = torch.clamp(y_true.abs(), min=WEIGHT_EPS)
    sum_w = weights.sum()
    if sum_w <= 0:
        return torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    mean_true = (y_true * weights).sum() / sum_w
    mean_pred = (y_pred * weights).sum() / sum_w
    dev_true = y_true - mean_true
    dev_pred = y_pred - mean_pred
    cov = (weights * dev_true * dev_pred).sum() / sum_w
    var_true = (weights * dev_true ** 2).sum() / sum_w
    var_pred = (weights * dev_pred ** 2).sum() / sum_w
    denom = torch.sqrt(var_true * var_pred).clamp(min=VAR_EPS)
    corr = cov / denom
    return corr.clamp(-1.0, 1.0)


def _masked_weighted_pearson_loss(
    pred: torch.Tensor, target: torch.Tensor, warmup_steps: int
) -> torch.Tensor:
    """
    Loss = 1 - mean(weighted_pearson по t0, weighted_pearson по t1).
    Учитываются только шаги [warmup_steps, T). pred/target: (B, T, 2).
    Минимизация этого лосса эквивалентна максимизации метрики из utils.
    """
    if warmup_steps > 0:
        pred = pred[:, warmup_steps:]
        target = target[:, warmup_steps:]
    # (B, T', 2) -> (N, 2)
    pred_flat = pred.reshape(-1, 2)
    target_flat = target.reshape(-1, 2)
    corr0 = _weighted_pearson_1d(target_flat[:, 0], pred_flat[:, 0])
    corr1 = _weighted_pearson_1d(target_flat[:, 1], pred_flat[:, 1])
    mean_corr = (corr0 + corr1) * 0.5
    return 1.0 - mean_corr


def train_epoch(model, loader, optimizer, criterion, device, epoch=None, total_epochs=None, total_batches=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    desc = "Train"
    if epoch is not None and total_epochs is not None:
        desc = f"Train Epoch {epoch}/{total_epochs}"
    pbar = tqdm(total=total_batches, desc=desc, unit="batch", leave=True)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model.forward_sequence(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        avg_loss = total_loss / n_batches
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
    pbar.close()
    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device, max_batches=None, epoch=None, total_epochs=None, total_val_batches=None):
    """Валидация: накапливаем pred/target после warmup, считаем метрику из utils (weighted_pearson)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds_list = []
    targets_list = []
    desc = "Val"
    if epoch is not None and total_epochs is not None:
        desc = f"Val Epoch {epoch}/{total_epochs}"
    pbar = tqdm(total=total_val_batches, desc=desc, unit="batch", leave=True)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model.forward_sequence(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            n_batches += 1
            # только шаги [WARMUP_STEPS:) для метрики из utils
            p = pred[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            t = y[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            preds_list.append(p)
            targets_list.append(t)
            avg_loss = total_loss / n_batches
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
            if max_batches is not None and n_batches >= max_batches:
                break
    pbar.close()
    avg_loss = total_loss / max(n_batches, 1)
    # метрика из utils.py по всему валидационному набору
    preds_all = np.concatenate(preds_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    corr0 = weighted_pearson_correlation(targets_all[:, 0], preds_all[:, 0])
    corr1 = weighted_pearson_correlation(targets_all[:, 1], preds_all[:, 1])
    val_pearson = (corr0 + corr1) / 2.0
    return avg_loss, val_pearson


def run_training_loop(save_path=None, epochs=None):
    """
    Запускает полный цикл обучения. Возвращает лучший val weighted_pearson (среднее по t0/t1).
    save_path: куда сохранять лучшие веса; None — не сохранять.
    epochs: число эпох; None — использовать EPOCHS из модуля.
    """
    if save_path is None:
        save_path = os.path.join(WEIGHTS_DIR, SAVE_NAME)
    if epochs is None:
        epochs = EPOCHS

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    log.info("Device: %s", DEVICE)
    model = FullModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = lambda pred, y: _masked_weighted_pearson_loss(pred, y, WARMUP_STEPS)

    if not os.path.isfile(TRAIN_PATH):
        log.error("Train file not found: %s", TRAIN_PATH)
        return -float("inf")

    train_ds = ParquetSequenceDataset(
        TRAIN_PATH,
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
    val_max_batches = None
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

    best_val_pearson = -float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            epoch=epoch, total_epochs=epochs,
            total_batches=total_train_batches,
        )
        log.info("Epoch %d train loss: %.6f", epoch, train_loss)

        if val_loader is not None:
            val_loss, val_pearson = validate(
                model, val_loader, criterion, DEVICE,
                max_batches=val_max_batches,
                epoch=epoch, total_epochs=epochs,
                total_val_batches=total_val_batches,
            )
            log.info("Epoch %d val loss: %.6f val weighted_pearson: %.6f", epoch, val_loss, val_pearson)
            if val_pearson > best_val_pearson:
                best_val_pearson = val_pearson
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
                    log.info("Saved best weights to %s", save_path)

    if val_loader is None:
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            log.info("Saved weights to %s", save_path)

    log.info("Done.")
    return best_val_pearson if val_loader is not None else -float("inf")


def main():
    save_path = os.path.join(WEIGHTS_DIR, SAVE_NAME)
    run_training_loop(save_path=save_path, epochs=EPOCHS)


if __name__ == "__main__":
    main()
