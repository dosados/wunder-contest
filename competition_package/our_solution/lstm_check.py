"""
Проверка совместной работы первой LSTM + второй LSTM (стек LSTM+LSTM2) на valid.parquet.

Загружает веса обеих моделей, прогоняет валидацию по шагам (как в inference):
pred = lstm1(x) + lstm2(x), клип [-6, 6], считает Weighted Pearson по таргетам.
Дополнительно выводит метрики только первой LSTM и только второй LSTM для сравнения.

Использование:
  python lstm_check.py [--lstm-run DIR] [--lstm2-weights PATH] [--valid PATH]
  python lstm_check.py --batched   # быстрая проверка батчами (forward_sequence)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(CURRENT_DIR, "..")
for d in (CURRENT_DIR, PKG_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import constants
from constants import (
    DEVICE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    VAL_PATH,
    WARMUP_STEPS,
    SEQ_LEN,
    SEQUENCE_BATCH_SIZE,
    LSTM_RUN_DIR,
    WEIGHTS_LSTM2_DIR,
    SAVE_NAME_LSTM2,
    PRED_CLIP_LOW,
    PRED_CLIP_HIGH,
)
from dataset import ParquetSequenceDataset
from torch.utils.data import DataLoader
from utils import weighted_pearson_correlation

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _remap_legacy_state_dict(raw: dict, model: torch.nn.Module) -> dict:
    """Маппинг старых весов (lstm + linear2) в lstm0/lstm1 + linear0/linear1."""
    state = dict(model.state_dict())
    for key, value in raw.items():
        if key.startswith("lstm."):
            for prefix in ("lstm0.", "lstm1."):
                new_key = prefix + key[5:]
                if new_key in state:
                    state[new_key] = value.clone()
        elif key == "linear2.weight":
            state["linear0.weight"] = value[0:1].clone()
            state["linear1.weight"] = value[1:2].clone()
        elif key == "linear2.bias":
            state["linear0.bias"] = value[0:1].clone()
            state["linear1.bias"] = value[1:2].clone()
        elif key in state:
            state[key] = value.clone()
    if "residual_proj.weight" in state and "residual_proj.weight" not in raw:
        state["residual_proj.weight"] = torch.zeros_like(state["residual_proj.weight"])
        state["residual_proj.bias"] = torch.zeros_like(state["residual_proj.bias"])
    return state


def load_lstm(run_dir: str):
    """Загружает первую LSTM из run_dir (config.json + best_model.pt)."""
    config_path = os.path.join(run_dir, "config.json")
    weights_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"LSTM: нужны {config_path} и {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hp = data.get("model", data.get("best_hyperparameters", data))
    constants.convo_constants.clear()
    constants.convo_constants.update(hp)

    from model import FullModel

    model = FullModel()
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Ожидается state_dict, получено: {type(raw)}")
    try:
        model.load_state_dict(raw, strict=True)
    except RuntimeError:
        raw = _remap_legacy_state_dict(raw, model)
        model.load_state_dict(raw, strict=True)
    return model.to(DEVICE).eval()


def load_lstm2(weights_path: str):
    """Загружает вторую LSTM (ResidualLSTM) по пути к весам (конфиг из constants.lstm2_constants)."""
    from residual_lstm_model import ResidualLSTM

    model = ResidualLSTM()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"Ожидается state_dict, получено: {type(state)}")
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def run_validation_streaming(lstm1, lstm2, val_ds, device):
    """
    Прогон valid по одной последовательности, по шагам: для каждого шага
    pred = lstm1(x_t, seq_ix) + lstm2(x_t, seq_ix), клип. Собираем предсказания
    только для шагов [WARMUP_STEPS:).
    Возвращает (preds_lstm1, preds_lstm2, preds_combined, targets) — массивы (N, 2).
    """
    n_seqs = len(val_ds)
    preds_lstm1 = []
    preds_lstm2 = []
    targets_list = []

    with torch.no_grad():
        for seq_idx in tqdm(range(n_seqs), desc="Valid (streaming)", unit="seq"):
            x, y = val_ds[seq_idx]
            x = x.to(device)
            y = y.numpy()
            seq_lstm1 = []
            seq_lstm2 = []
            for step in range(SEQ_LEN):
                x_t = x[step : step + 1]
                out_lstm1 = lstm1(x_t, seq_ix=seq_idx)
                out_lstm2 = lstm2(x_t, seq_ix=seq_idx)
                if out_lstm1.dim() == 2:
                    out_lstm1 = out_lstm1.squeeze(0)
                if out_lstm2.dim() == 2:
                    out_lstm2 = out_lstm2.squeeze(0)
                seq_lstm1.append(out_lstm1.cpu().numpy())
                seq_lstm2.append(out_lstm2.cpu().numpy())
            seq_lstm1 = np.array(seq_lstm1)
            seq_lstm2 = np.array(seq_lstm2)
            combined = np.clip(seq_lstm1 + seq_lstm2, PRED_CLIP_LOW, PRED_CLIP_HIGH)
            preds_lstm1.append(seq_lstm1[WARMUP_STEPS:])
            preds_lstm2.append(seq_lstm2[WARMUP_STEPS:])
            targets_list.append(y[WARMUP_STEPS:])

    preds_lstm1 = np.concatenate(preds_lstm1, axis=0)
    preds_lstm2 = np.concatenate(preds_lstm2, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds_combined = np.clip(preds_lstm1 + preds_lstm2, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    preds_lstm1 = np.clip(preds_lstm1, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    preds_lstm2 = np.clip(preds_lstm2, PRED_CLIP_LOW, PRED_CLIP_HIGH)

    return preds_lstm1, preds_lstm2, preds_combined, targets


def run_validation_batched(lstm1, lstm2, val_loader, device):
    """
    Быстрая валидация батчами: forward_sequence по целым последовательностям.
    pred = lstm1.forward_sequence(x) + lstm2.forward_sequence(x), клип. Шаги [WARMUP_STEPS:).
    Возвращает (preds_lstm1, preds_lstm2, preds_combined, targets) — массивы (N, 2).
    """
    preds_lstm1 = []
    preds_lstm2 = []
    targets_list = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Valid (batched)", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            out_lstm1 = lstm1.forward_sequence(x)
            out_lstm2 = lstm2.forward_sequence(x)
            p_lstm1 = out_lstm1[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            p_lstm2 = out_lstm2[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            t = y[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            preds_lstm1.append(p_lstm1)
            preds_lstm2.append(p_lstm2)
            targets_list.append(t)

    preds_lstm1 = np.concatenate(preds_lstm1, axis=0)
    preds_lstm2 = np.concatenate(preds_lstm2, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds_combined = np.clip(preds_lstm1 + preds_lstm2, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    preds_lstm1 = np.clip(preds_lstm1, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    preds_lstm2 = np.clip(preds_lstm2, PRED_CLIP_LOW, PRED_CLIP_HIGH)

    return preds_lstm1, preds_lstm2, preds_combined, targets


def main():
    parser = argparse.ArgumentParser(
        description="Проверка LSTM + LSTM2 (вторая LSTM на остатках) на valid.parquet"
    )
    parser.add_argument("--valid", default=VAL_PATH, help="Путь к valid.parquet")
    parser.add_argument("--lstm-run", default=LSTM_RUN_DIR, help="Директория с весами первой LSTM")
    parser.add_argument(
        "--lstm2-weights",
        default=os.path.join(WEIGHTS_LSTM2_DIR, SAVE_NAME_LSTM2),
        help="Путь к весам второй LSTM (.pt)",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Быстрая валидация батчами (forward_sequence вместо пошагового режима)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.valid):
        log.error("Не найден valid.parquet: %s", args.valid)
        sys.exit(1)
    if not os.path.isdir(args.lstm_run):
        log.error("Не найдена директория первой LSTM: %s", args.lstm_run)
        sys.exit(1)
    if not os.path.isfile(args.lstm2_weights):
        log.error("Не найдены веса второй LSTM: %s", args.lstm2_weights)
        sys.exit(1)

    log.info("Загрузка первой LSTM из %s", args.lstm_run)
    lstm1 = load_lstm(args.lstm_run)
    log.info("Загрузка второй LSTM из %s", args.lstm2_weights)
    lstm2 = load_lstm2(args.lstm2_weights)

    val_ds = ParquetSequenceDataset(
        args.valid,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    log.info("Валидация: %d последовательностей", len(val_ds))

    if args.batched:
        val_loader = DataLoader(
            val_ds,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        preds_lstm1, preds_lstm2, preds_combined, targets = run_validation_batched(
            lstm1, lstm2, val_loader, DEVICE
        )
    else:
        preds_lstm1, preds_lstm2, preds_combined, targets = run_validation_streaming(
            lstm1, lstm2, val_ds, DEVICE
        )

    def pearson_two_targets(pred, tgt):
        c0 = weighted_pearson_correlation(tgt[:, 0], pred[:, 0])
        c1 = weighted_pearson_correlation(tgt[:, 1], pred[:, 1])
        return (c0 + c1) / 2.0, c0, c1

    score_lstm1, l0, l1 = pearson_two_targets(preds_lstm1, targets)
    score_lstm2, s0, s1 = pearson_two_targets(preds_lstm2, targets)
    score_combined, c0, c1 = pearson_two_targets(preds_combined, targets)

    print("\n--- Результаты на valid.parquet (стек LSTM + LSTM2) ---")
    print(f"  Только LSTM (1-я):   {score_lstm1:.6f}  (t0: {l0:.6f}, t1: {l1:.6f})")
    print(f"  Только LSTM (2-я):   {score_lstm2:.6f}  (t0: {s0:.6f}, t1: {s1:.6f})")
    print(f"  LSTM + LSTM2 (итог): {score_combined:.6f}  (t0: {c0:.6f}, t1: {c1:.6f})")
    print("\nИтоговая метрика (среднее по t0/t1): %.6f" % score_combined)


if __name__ == "__main__":
    main()
