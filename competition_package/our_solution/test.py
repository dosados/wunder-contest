"""
Тест модели: загрузка конфига и весов из inference/, валидация двумя способами —
батчами (как при обучении) и по одному вектору (как в PredictionModel).
"""
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Пути для импортов: our_solution и competition_package (utils)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
PKG_DIR = os.path.join(CURRENT_DIR, "..")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

from constants import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TRAIN_PATH,
    WARMUP_STEPS,
    SEQUENCE_BATCH_SIZE,
    SEQ_LEN,
    PRED_CLIP_LOW,
    PRED_CLIP_HIGH,
    DEVICE,
    INFERENCE_CONFIG_PATH,
    INFERENCE_WEIGHTS_PATH,
)
from dataset import ParquetSequenceDataset
from utils import weighted_pearson_correlation


def _remap_legacy_state_dict(raw: dict, model: torch.nn.Module) -> dict:
    """
    Преобразует старый state_dict (один lstm + linear2) в формат текущей FullModel
    (lstm0, lstm1, linear0, linear1, residual_proj). Остальные ключи копируются как есть.
    """
    state = dict(model.state_dict())
    for key, value in raw.items():
        if key.startswith("lstm."):
            for prefix in ("lstm0.", "lstm1."):
                new_key = prefix + key[5:]  # "lstm." -> "lstm0." / "lstm1."
                if new_key in state:
                    state[new_key] = value.clone()
        elif key == "linear2.weight":
            # (2, hidden) -> linear0 (1, hidden), linear1 (1, hidden)
            state["linear0.weight"] = value[0:1].clone()
            state["linear1.weight"] = value[1:2].clone()
        elif key == "linear2.bias":
            state["linear0.bias"] = value[0:1].clone()
            state["linear1.bias"] = value[1:2].clone()
        elif key in state:
            state[key] = value.clone()
    # residual_proj в старом чекпоинте не было — обнуляем, чтобы residual=0
    if "residual_proj.weight" in state and "residual_proj.weight" not in raw:
        state["residual_proj.weight"] = torch.zeros_like(state["residual_proj.weight"])
        state["residual_proj.bias"] = torch.zeros_like(state["residual_proj.bias"])
    return state


def load_config_and_apply(config_path: str) -> dict:
    """Загружает config.json и применяет best_hyperparameters к constants.convo_constants."""
    import constants
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
    constants.convo_constants.clear()
    constants.convo_constants.update(config)
    return config


def validate_batched(model, val_loader, device):
    """
    Валидация батчами, как при обучении: forward_sequence(x), метрика по шагам [WARMUP_STEPS:).
    """
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Val (batched)", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            pred = model.forward_sequence(x)
            p = pred[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            t = y[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            preds_list.append(p)
            targets_list.append(t)
    preds_all = np.concatenate(preds_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    preds_all = np.clip(preds_all, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    corr0 = weighted_pearson_correlation(targets_all[:, 0], preds_all[:, 0])
    corr1 = weighted_pearson_correlation(targets_all[:, 1], preds_all[:, 1])
    return (corr0 + corr1) / 2.0


def validate_one_vector_at_a_time(model, val_ds, device):
    """
    Валидация по одному вектору за раз, как в PredictionModel: для каждой последовательности
    прогоняем шаги через model.forward(x_t, seq_ix=seq_idx), собираем предсказания [WARMUP_STEPS:).
    """
    model.eval()
    preds_list = []
    targets_list = []
    n_seqs = len(val_ds)
    with torch.no_grad():
        for seq_idx in tqdm(range(n_seqs), desc="Val (one vector at a time)", unit="seq"):
            x, y = val_ds[seq_idx]
            # x: (SEQ_LEN, n_feat), y: (SEQ_LEN, 2)
            seq_preds = []
            for step in range(SEQ_LEN):
                x_t = x[step : step + 1].to(device)  # (1, D)
                out = model.forward(x_t, seq_ix=seq_idx)
                if out.dim() == 2:
                    out = out[-1]
                pred_t = out.cpu().numpy().flatten()
                seq_preds.append(pred_t)
            seq_preds = np.array(seq_preds)  # (SEQ_LEN, 2)
            # только шаги [WARMUP_STEPS:)
            p = np.clip(seq_preds[WARMUP_STEPS:], PRED_CLIP_LOW, PRED_CLIP_HIGH)
            t = y[WARMUP_STEPS:].numpy()
            preds_list.append(p)
            targets_list.append(t)
    preds_all = np.concatenate(preds_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    corr0 = weighted_pearson_correlation(targets_all[:, 0], preds_all[:, 0])
    corr1 = weighted_pearson_correlation(targets_all[:, 1], preds_all[:, 1])
    return (corr0 + corr1) / 2.0


def main():
    config_path = INFERENCE_CONFIG_PATH
    weights_path = INFERENCE_WEIGHTS_PATH

    if not os.path.isfile(config_path):
        log.error("Конфиг не найден: %s", config_path)
        return
    if not os.path.isfile(weights_path):
        log.error("Веса не найдены: %s", weights_path)
        return
    if not os.path.isfile(TRAIN_PATH):
        log.error("Датасет для валидации не найден: %s", TRAIN_PATH)
        return

    log.info("Загрузка конфига из %s", config_path)
    config = load_config_and_apply(config_path)
    print("Конфиг (best_hyperparameters):")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # Импорт после применения конфига (FullModel читает convo_constants при создании)
    from model.full_model import FullModel

    log.info("Загрузка модели и весов из %s", weights_path)
    raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, dict):
        log.error("Файл весов должен содержать state_dict (dict), получено: %s", type(raw_state))
        return
    model = FullModel()
    try:
        model.load_state_dict(raw_state, strict=True)
    except RuntimeError as e:
        if "Missing key(s)" in str(e) and "lstm0" in str(e) and "linear2.weight" in raw_state:
            log.info("Обнаружен старый формат весов (lstm + linear2), применяю маппинг в lstm0/lstm1 + linear0/linear1")
            state = _remap_legacy_state_dict(raw_state, model)
            model.load_state_dict(state, strict=True)
        else:
            raise
    model = model.to(DEVICE)
    model.eval()

    val_ds = ParquetSequenceDataset(
        TRAIN_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    log.info("Валидация батчами (как при обучении)...")
    pearson_batched = validate_batched(model, val_loader, DEVICE)
    log.info("Weighted Pearson (batched): %.6f", pearson_batched)

    log.info("Валидация по одному вектору (как в PredictionModel)...")
    pearson_streaming = validate_one_vector_at_a_time(model, val_ds, DEVICE)
    log.info("Weighted Pearson (one vector at a time): %.6f", pearson_streaming)

    print("\nИтог:")
    print(f"  Batched (как при обучении):     {pearson_batched:.6f}")
    print(f"  One vector at a time (как inference): {pearson_streaming:.6f}")


if __name__ == "__main__":
    main()
