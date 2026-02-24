"""
Обучение метамодели (stack) на валидации: базовые FullModel и GRUModel заморожены,
обучается только meta_head по weighted Pearson loss.
"""
import os
import sys
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)  # our_solution
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PKG = os.path.dirname(ROOT)
if PKG not in sys.path:
    sys.path.insert(0, PKG)
from utils import weighted_pearson_correlation
from dataset import ParquetSequenceDataset

from constants import DEVICE, convo_constants, gru_constants
from stack.stack_model import StackModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Пути к весам базовых моделей (относительно our_solution)
INFERENCE_DIR = os.path.join(ROOT, "inference")
FULL_MODEL_CONFIG_PATH = os.path.join(INFERENCE_DIR, "config.json")
FULL_MODEL_WEIGHTS_PATH = os.path.join(INFERENCE_DIR, "weights", "best_model.pt")
INFERENCE_GRU_DIR = os.path.join(ROOT, "inference_gru")
GRU_CONFIG_PATH = os.path.join(INFERENCE_GRU_DIR, "config.json")
GRU_WEIGHTS_PATH = os.path.join(INFERENCE_GRU_DIR, "weights", "best_model.pt")

VAL_PATH = os.path.join(ROOT, "..", "datasets", "valid.parquet")
TRAIN_PATH = os.path.join(ROOT, "..", "datasets", "train.parquet")
WEIGHTS_STACK_DIR = os.path.join(CURRENT_DIR, "weights_stack")
SAVE_NAME = "best_stack.pt"

FEATURE_COLUMNS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)
TARGET_COLUMNS = ["t0", "t1"]
WARMUP_STEPS = 99
SEQUENCE_BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

PRED_CLIP_LOW = -6.0
PRED_CLIP_HIGH = 6.0
WEIGHT_EPS = 1e-8
VAR_EPS = 1e-8

# Доля train.parquet для валидации (выбор лучшей эпохи); остальное не используется
TRAIN_VAL_FRACTION = 0.1


def _apply_convo_config(config: dict) -> None:
    convo_constants.clear()
    convo_constants.update(config)


def _load_full_model():
    if not os.path.isfile(FULL_MODEL_CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {FULL_MODEL_CONFIG_PATH}")
    if not os.path.isfile(FULL_MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found: {FULL_MODEL_WEIGHTS_PATH}")
    with open(FULL_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
    _apply_convo_config(config)
    from model import FullModel
    model = FullModel()
    model.load_state_dict(torch.load(FULL_MODEL_WEIGHTS_PATH, map_location="cpu"), strict=True)
    return model


def _load_gru_model():
    if not os.path.isfile(GRU_WEIGHTS_PATH):
        raise FileNotFoundError(f"GRU weights not found: {GRU_WEIGHTS_PATH}")
    if os.path.isfile(GRU_CONFIG_PATH):
        with open(GRU_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
        gru_constants.clear()
        gru_constants.update(config)
    from model import GRUModel
    model = GRUModel()
    model.load_state_dict(torch.load(GRU_WEIGHTS_PATH, map_location="cpu"), strict=True)
    return model


def _weighted_pearson_1d(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
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
    if warmup_steps > 0:
        pred = pred[:, warmup_steps:]
        target = target[:, warmup_steps:]
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
    preds_all = np.concatenate(preds_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    corr0 = weighted_pearson_correlation(targets_all[:, 0], preds_all[:, 0])
    corr1 = weighted_pearson_correlation(targets_all[:, 1], preds_all[:, 1])
    val_pearson = (corr0 + corr1) / 2.0
    return avg_loss, val_pearson


def run_training_loop(
    meta_hidden_dims=None,
    save_path=None,
    epochs=None,
):
    """
    Загружает замороженные FullModel и GRUModel, строит StackModel и обучает только meta_head на valid.
    Валидация (выбор лучшей эпохи и сохранение весов) — по weighted_pearson на train.parquet.
    meta_hidden_dims: список скрытых размеров мета-головы, например [64] или [32, 16].
    Возвращает лучший train weighted_pearson (по нему выбирается чекпоинт).
    """
    if meta_hidden_dims is None:
        meta_hidden_dims = DEFAULT_META_HIDDEN_DIMS
    if save_path is None:
        save_path = os.path.join(WEIGHTS_STACK_DIR, SAVE_NAME)
    if epochs is None:
        epochs = EPOCHS

    os.makedirs(WEIGHTS_STACK_DIR, exist_ok=True)

    log.info("Loading FullModel from %s", FULL_MODEL_WEIGHTS_PATH)
    full_model = _load_full_model()
    log.info("Loading GRUModel from %s", GRU_WEIGHTS_PATH)
    gru_model = _load_gru_model()

    for p in full_model.parameters():
        p.requires_grad = False
    for p in gru_model.parameters():
        p.requires_grad = False

    stack_model = StackModel(
        full_model=full_model,
        gru_model=gru_model,
        meta_hidden_dims=meta_hidden_dims,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(stack_model.meta_head.parameters(), lr=LR)
    criterion = lambda pred, y: _masked_weighted_pearson_loss(pred, y, WARMUP_STEPS)

    if not os.path.isfile(VAL_PATH):
        log.error("Validation file not found: %s", VAL_PATH)
        return -float("inf")
    if not os.path.isfile(TRAIN_PATH):
        log.error("Train file not found: %s (need it for validation)", TRAIN_PATH)
        return -float("inf")

    val_ds = ParquetSequenceDataset(
        VAL_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    total_val_batches = len(val_loader)

    train_ds = ParquetSequenceDataset(
        TRAIN_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    n_train = len(train_ds)
    n_train_val = max(1, int(n_train * TRAIN_VAL_FRACTION))
    train_val_indices = list(range(n_train_val))
    train_val_ds = Subset(train_ds, train_val_indices)
    log.info("Validation on train set: using %d / %d sequences (%.1f%%)", n_train_val, n_train, TRAIN_VAL_FRACTION * 100)
    train_loader = DataLoader(
        train_val_ds,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    total_train_batches = len(train_loader)

    best_train_pearson = -float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            stack_model,
            val_loader,
            optimizer,
            criterion,
            DEVICE,
            epoch=epoch,
            total_epochs=epochs,
            total_batches=total_val_batches,
        )
        log.info("Epoch %d train loss (on valid): %.6f", epoch, train_loss)

        train_metric_loss, train_pearson = validate(
            stack_model,
            train_loader,
            criterion,
            DEVICE,
            epoch=epoch,
            total_epochs=epochs,
            total_val_batches=total_train_batches,
        )
        log.info("Epoch %d train set loss: %.6f train weighted_pearson: %.6f", epoch, train_metric_loss, train_pearson)
        if train_pearson > best_train_pearson:
            best_train_pearson = train_pearson
            torch.save(stack_model.state_dict(), save_path)
            log.info("Saved best stack weights to %s", save_path)

    log.info("Done. Best train weighted_pearson: %.6f", best_train_pearson)
    return best_train_pearson


def main():
    run_training_loop(meta_hidden_dims=DEFAULT_META_HIDDEN_DIMS)


if __name__ == "__main__":
    main()
