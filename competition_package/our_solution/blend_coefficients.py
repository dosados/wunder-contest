"""
Обучение коэффициентов смешивания LSTM + SSM на половине valid, валидация на второй половине.

pred = alpha * lstm_pred + beta * ssm_pred (по каждому таргету отдельно), затем клип [-6, 6].
Коэффициенты обучаются максимизацией Weighted Pearson на первой половине последовательностей valid.

Запуск (из our_solution):
  python blend_coefficients.py [--valid PATH] [--lstm-run DIR] [--ssm-weights PATH]
  python blend_coefficients.py --batched
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
from scipy.optimize import minimize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(CURRENT_DIR, "..")
for d in (CURRENT_DIR, PKG_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import constants  # noqa: F401
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

from check import load_lstm, load_ssm, run_validation_batched, run_validation_streaming

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _split_sequences_by_half(n_sequences: int):
    """Индексы первой и второй половины последовательностей."""
    half = n_sequences // 2
    train_ix = np.arange(0, half)
    val_ix = np.arange(half, n_sequences)
    return train_ix, val_ix


def _blend(alpha0, beta0, alpha1, beta1, pred_lstm, pred_ssm):
    """pred = alpha * lstm + beta * ssm по каждому таргету, затем клип."""
    out = np.zeros_like(pred_lstm)
    out[:, 0] = alpha0 * pred_lstm[:, 0] + beta0 * pred_ssm[:, 0]
    out[:, 1] = alpha1 * pred_lstm[:, 1] + beta1 * pred_ssm[:, 1]
    return np.clip(out, PRED_CLIP_LOW, PRED_CLIP_HIGH)


def _weighted_pearson_two_targets(pred, target):
    c0 = weighted_pearson_correlation(target[:, 0], pred[:, 0])
    c1 = weighted_pearson_correlation(target[:, 1], pred[:, 1])
    return (c0 + c1) / 2.0


def _loss_for_optimizer(coeffs, pred_lstm, pred_ssm, target):
    """Минус средний Weighted Pearson по двум таргетам (для минимизации)."""
    alpha0, beta0, alpha1, beta1 = coeffs
    pred = _blend(alpha0, beta0, alpha1, beta1, pred_lstm, pred_ssm)
    return -_weighted_pearson_two_targets(pred, target)


def train_coefficients(preds_lstm, preds_ssm, targets, train_ix, steps_per_seq, epochs: int = 5):
    """
    Обучает 4 коэффициента на предсказаниях и таргетах для последовательностей train_ix.
    preds_* и targets — плоские (N, 2). Несколько эпох: каждый раз старт с результата предыдущей.
    """
    idx = np.concatenate([np.arange(ix * steps_per_seq, (ix + 1) * steps_per_seq) for ix in train_ix])
    pred_lstm = preds_lstm[idx]
    pred_ssm = preds_ssm[idx]
    target = targets[idx]

    def loss(c):
        return _loss_for_optimizer(c, pred_lstm, pred_ssm, target)

    bounds = [(0.0, 3.0)] * 4
    x0 = np.array([1.0, 1.0, 1.0, 1.0])  # начальное: как линейная сумма
    coeffs = x0
    for ep in range(1, epochs + 1):
        res = minimize(loss, coeffs, method="L-BFGS-B", bounds=bounds)
        coeffs = res.x
        score = -res.fun
        log.info("Эпоха %d/%d: coeffs=%s, train Pearson=%.6f", ep, epochs, coeffs, score)
    return coeffs


def main():
    parser = argparse.ArgumentParser(
        description="Обучение коэффициентов смешивания LSTM+SSM на половине valid"
    )
    parser.add_argument("--valid", default=VAL_PATH, help="Путь к valid.parquet")
    parser.add_argument("--lstm-run", default=LSTM_RUN_DIR, help="Директория с весами LSTM")
    parser.add_argument(
        "--ssm-weights",
        default=os.path.join(WEIGHTS_SSM_DIR, SAVE_NAME_SSM),
        help="Путь к весам SSM",
    )
    parser.add_argument("--batched", action="store_true", help="Использовать батчевую валидацию")
    parser.add_argument("--epochs", type=int, default=5, help="Число эпох оптимизации коэффициентов (по умолчанию 5)")
    args = parser.parse_args()

    if not os.path.isfile(args.valid):
        log.error("Не найден valid.parquet: %s", args.valid)
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
    n_seqs = len(val_ds)
    steps_per_seq = SEQ_LEN - WARMUP_STEPS

    if args.batched:
        loader = DataLoader(
            val_ds,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        preds_lstm, preds_ssm, preds_combined, targets = run_validation_batched(
            lstm, ssm, loader, DEVICE
        )
    else:
        preds_lstm, preds_ssm, preds_combined, targets = run_validation_streaming(
            lstm, ssm, val_ds, DEVICE
        )

    train_ix, val_ix = _split_sequences_by_half(n_seqs)
    coeffs = train_coefficients(
        preds_lstm, preds_ssm, targets, train_ix, steps_per_seq, epochs=args.epochs
    )
    alpha0, beta0, alpha1, beta1 = coeffs

    # Валидация на второй половине
    idx_val = np.concatenate(
        [np.arange(ix * steps_per_seq, (ix + 1) * steps_per_seq) for ix in val_ix]
    )
    pred_lstm_val = preds_lstm[idx_val]
    pred_ssm_val = preds_ssm[idx_val]
    target_val = targets[idx_val]
    pred_blend_val = _blend(alpha0, beta0, alpha1, beta1, pred_lstm_val, pred_ssm_val)

    score_linear = _weighted_pearson_two_targets(
        np.clip(preds_lstm[idx_val] + preds_ssm[idx_val], PRED_CLIP_LOW, PRED_CLIP_HIGH),
        target_val,
    )
    score_blend = _weighted_pearson_two_targets(pred_blend_val, target_val)

    print("\n--- Коэффициенты (обучены на первой половине valid) ---")
    print("  alpha0=%.4f, beta0=%.4f, alpha1=%.4f, beta1=%.4f" % (alpha0, beta0, alpha1, beta1))
    print("\n--- Валидация на второй половине valid ---")
    print("  Линейная сумма (LSTM+SSM): %.6f" % score_linear)
    print("  Обучаемое смешивание:      %.6f" % score_blend)

    # Сохранение коэффициентов для использования в инференсе
    out_path = os.path.join(CURRENT_DIR, "blend_coefficients.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"alpha0": alpha0, "beta0": beta0, "alpha1": alpha1, "beta1": beta1},
            f,
            indent=2,
        )
    log.info("Коэффициенты сохранены: %s", out_path)


if __name__ == "__main__":
    main()
