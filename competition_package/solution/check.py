"""
Проверка совместной работы LSTM + SSM на valid.parquet.

Загружает веса обеих моделей, прогоняет валидацию по шагам (как в inference):
pred = f1(x) + f2(x), клип [-6, 6], считает Weighted Pearson по таргетам.
Дополнительно выводит метрики только LSTM и только SSM для сравнения.

Использование:
  python check.py [--lstm-run DIR] [--ssm-weights PATH] [--valid PATH]
  python check.py --batched   # быстрая проверка батчами (forward_sequence)
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
    WEIGHTS_SSM_DIR,
    SAVE_NAME_SSM,
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
    """Загружает LSTM из run_dir (config.json + best_model.pt)."""
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


def load_ssm(weights_path: str):
    """Загружает SSM по пути к весам (конфиг из constants.ssm_constants)."""
    from ssm_model import SSMModel

    model = SSMModel()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"Ожидается state_dict, получено: {type(state)}")
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def run_validation_streaming(lstm, ssm, val_ds, device):
    """
    Прогон valid по одной последовательности, по шагам: для каждого шага
    pred = lstm(x_t, seq_ix) + ssm(x_t, seq_ix), клип. Собираем предсказания
    только для шагов [WARMUP_STEPS:).
    Возвращает (preds_lstm, preds_ssm, preds_combined, targets) — массивы (N, 2).
    """
    n_seqs = len(val_ds)
    preds_lstm = []
    preds_ssm = []
    targets_list = []

    with torch.no_grad():
        for seq_idx in tqdm(range(n_seqs), desc="Valid (streaming)", unit="seq"):
            x, y = val_ds[seq_idx]
            x = x.to(device)
            y = y.numpy()
            seq_lstm = []
            seq_ssm = []
            for step in range(SEQ_LEN):
                x_t = x[step : step + 1]
                out_lstm = lstm(x_t, seq_ix=seq_idx)
                out_ssm = ssm(x_t, seq_ix=seq_idx)
                if out_lstm.dim() == 2:
                    out_lstm = out_lstm.squeeze(0)
                if out_ssm.dim() == 2:
                    out_ssm = out_ssm.squeeze(0)
                seq_lstm.append(out_lstm.cpu().numpy())
                seq_ssm.append(out_ssm.cpu().numpy())
            seq_lstm = np.array(seq_lstm)
            seq_ssm = np.array(seq_ssm)
            combined = np.clip(seq_lstm + seq_ssm, PRED_CLIP_LOW, PRED_CLIP_HIGH)
            preds_lstm.append(seq_lstm[WARMUP_STEPS:])
            preds_ssm.append(seq_ssm[WARMUP_STEPS:])
            targets_list.append(y[WARMUP_STEPS:])

    preds_lstm = np.concatenate(preds_lstm, axis=0)
    preds_ssm = np.concatenate(preds_ssm, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds_combined = np.clip(preds_lstm + preds_ssm, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    preds_lstm = np.clip(preds_lstm, PRED_CLIP_LOW, PRED_CLIP_HIGH)

    return preds_lstm, preds_ssm, preds_combined, targets


def run_validation_batched(lstm, ssm, val_loader, device):
    """
    Быстрая валидация батчами: forward_sequence по целым последовательностям.
    pred = lstm.forward_sequence(x) + ssm.forward_sequence(x), клип. Шаги [WARMUP_STEPS:).
    Возвращает (preds_lstm, preds_ssm, preds_combined, targets) — массивы (N, 2).
    """
    preds_lstm = []
    preds_ssm = []
    targets_list = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Valid (batched)", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            out_lstm = lstm.forward_sequence(x)
            out_ssm = ssm.forward_sequence(x)
            p_lstm = out_lstm[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            p_ssm = out_ssm[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            t = y[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            preds_lstm.append(p_lstm)
            preds_ssm.append(p_ssm)
            targets_list.append(t)

    preds_lstm = np.concatenate(preds_lstm, axis=0)
    preds_ssm = np.concatenate(preds_ssm, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds_combined = np.clip(preds_lstm + preds_ssm, PRED_CLIP_LOW, PRED_CLIP_HIGH)
    preds_lstm = np.clip(preds_lstm, PRED_CLIP_LOW, PRED_CLIP_HIGH)

    return preds_lstm, preds_ssm, preds_combined, targets


def main():
    parser = argparse.ArgumentParser(description="Проверка LSTM + SSM на valid.parquet")
    parser.add_argument("--valid", default=VAL_PATH, help="Путь к valid.parquet")
    parser.add_argument("--lstm-run", default=LSTM_RUN_DIR, help="Директория с весами LSTM")
    parser.add_argument(
        "--ssm-weights",
        default=os.path.join(WEIGHTS_SSM_DIR, SAVE_NAME_SSM),
        help="Путь к весам SSM (.pt)",
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
        log.error("Не найдена директория LSTM: %s", args.lstm_run)
        sys.exit(1)
    if not os.path.isfile(args.ssm_weights):
        log.error("Не найдены веса SSM: %s", args.ssm_weights)
        sys.exit(1)

    log.info("Загрузка LSTM из %s", args.lstm_run)
    lstm = load_lstm(args.lstm_run)
    log.info("Загрузка SSM из %s", args.ssm_weights)
    ssm = load_ssm(args.ssm_weights)

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
        preds_lstm, preds_ssm, preds_combined, targets = run_validation_batched(
            lstm, ssm, val_loader, DEVICE
        )
    else:
        preds_lstm, preds_ssm, preds_combined, targets = run_validation_streaming(
            lstm, ssm, val_ds, DEVICE
        )

    def pearson_two_targets(pred, tgt):
        c0 = weighted_pearson_correlation(tgt[:, 0], pred[:, 0])
        c1 = weighted_pearson_correlation(tgt[:, 1], pred[:, 1])
        return (c0 + c1) / 2.0, c0, c1

    score_lstm, l0, l1 = pearson_two_targets(preds_lstm, targets)
    score_ssm, s0, s1 = pearson_two_targets(preds_ssm, targets)
    score_combined, c0, c1 = pearson_two_targets(preds_combined, targets)

    print("\n--- Результаты на valid.parquet ---")
    print(f"  Только LSTM:        {score_lstm:.6f}  (t0: {l0:.6f}, t1: {l1:.6f})")
    print(f"  Только SSM:         {score_ssm:.6f}  (t0: {s0:.6f}, t1: {s1:.6f})")
    print(f"  LSTM + SSM (итог):  {score_combined:.6f}  (t0: {c0:.6f}, t1: {c1:.6f})")
    print("\nИтоговая метрика (среднее по t0/t1): %.6f" % score_combined)


if __name__ == "__main__":
    main()
