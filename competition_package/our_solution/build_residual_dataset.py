"""
Создание датасетов остатков для обучения SSM по plan.txt.

LSTM обучена на первой половине train.parquet (gym). Для SSM:
1) Вторая половина train → parquet с остатками (train_residual.parquet).
2) Валидация valid → parquet с остатками (valid_residual.parquet), центровка по
   y_w_mean, f1_w_mean с train (сохраняются в residual_metadata.json).

Структура parquet та же: те же колонки, в t0/t1 записаны остатки r' = y' - f1'.

Использование:
  python build_residual_dataset.py                    # оба датасета
  python build_residual_dataset.py --train-only       # только train
  python build_residual_dataset.py --valid-only       # только valid (нужен residual_metadata.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
    TRAIN_PATH,
    VAL_PATH,
    SEQUENCE_BATCH_SIZE,
    SEQ_LEN,
    LSTM_RUN_DIR,
    WEIGHT_EPS,
    TRAIN_RESIDUAL_PATH,
    VALID_RESIDUAL_PATH,
    RESIDUAL_METADATA_PATH,
)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _second_half_indices(n_sequences: int):
    """Индексы второй половины последовательностей (для SSM)."""
    half = n_sequences // 2
    return list(range(half, n_sequences)), half, n_sequences


def load_lstm_config_and_model(run_dir: str):
    """Загружает конфиг из run_dir/config.json, патчит constants, создаёт и загружает FullModel."""
    config_path = os.path.join(run_dir, "config.json")
    weights_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Веса LSTM не найдены: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hp = data.get("model", data.get("best_hyperparameters", data))
    constants.convo_constants.clear()
    constants.convo_constants.update(hp)

    from model import FullModel

    model = FullModel()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"Ожидается state_dict, получено: {type(state)}")

    from test import _remap_legacy_state_dict

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        if "lstm0" in str(state) or "linear2" in list(state.keys()):
            state = _remap_legacy_state_dict(state, model)
            model.load_state_dict(state, strict=True)
        else:
            raise
    return model.to(DEVICE), hp


def _table_to_xy(table, feature_cols, target_cols, seq_len):
    """Из таблицы (все строки одной части) извлекает X (N_seq, SEQ_LEN, F), Y (N_seq, SEQ_LEN, 2)."""
    # Поддержка таблиц без части колонок (только features + targets)
    have_feat = all(c in table.column_names for c in feature_cols)
    have_tgt = all(c in table.column_names for c in target_cols)
    if not have_feat or not have_tgt:
        raise ValueError(
            f"В parquet нужны колонки {feature_cols} и {target_cols}, есть: {table.column_names}"
        )
    n_rows = table.num_rows
    n_seq = n_rows // seq_len
    X = np.column_stack([table.column(c).to_numpy() for c in feature_cols]).astype(np.float32)
    Y = np.column_stack([table.column(c).to_numpy() for c in target_cols]).astype(np.float32)
    X = X.reshape(n_seq, seq_len, -1)
    Y = Y.reshape(n_seq, seq_len, -1)
    return X, Y


def _residual_from_xy(X, Y, F1, weight_eps):
    """Считает взвешенно центрированные остатки и средние по плану."""
    F1 = np.clip(F1, constants.PRED_CLIP_LOW, constants.PRED_CLIP_HIGH)
    y_flat = Y.reshape(-1, 2)
    f1_flat = F1.reshape(-1, 2)
    w_flat = np.maximum(np.abs(y_flat), weight_eps)

    y_w_mean = np.zeros(2)
    f1_w_mean = np.zeros(2)
    for j in range(2):
        wj = w_flat[:, j]
        y_w_mean[j] = np.average(y_flat[:, j], weights=wj)
        f1_w_mean[j] = np.average(f1_flat[:, j], weights=wj)

    residual_flat = (y_flat - y_w_mean) - (f1_flat - f1_w_mean)
    return residual_flat.astype(np.float32), y_w_mean, f1_w_mean


def _write_residual_parquet(table_slice, column_names, residual_flat, target_cols, output_path):
    """Пишет parquet: те же колонки, что в table_slice, в target_cols подставлены остатки."""
    cols = {}
    t0_name, t1_name = target_cols[0], target_cols[1]
    for name in column_names:
        if name == t0_name:
            cols[name] = pa.array(residual_flat[:, 0])
        elif name == t1_name:
            cols[name] = pa.array(residual_flat[:, 1])
        else:
            cols[name] = table_slice.column(name)
    out_table = pa.table(cols)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pq.write_table(out_table, output_path)
    log.info("Сохранён parquet: %s", output_path)


def build_train_residual(
    train_path: str,
    lstm_run_dir: str,
    output_parquet: str,
    metadata_path: str,
    batch_size: int = SEQUENCE_BATCH_SIZE,
):
    """
    Строит датасет остатков для второй половины train.parquet.
    Сохраняет parquet и residual_metadata.json (y_w_mean, f1_w_mean для valid).
    """
    log.info("Загрузка LSTM из %s", lstm_run_dir)
    lstm, _ = load_lstm_config_and_model(lstm_run_dir)
    lstm.eval()

    log.info("Чтение train.parquet: %s", train_path)
    table = pq.read_table(train_path)
    n_rows = table.num_rows
    n_sequences = n_rows // SEQ_LEN
    half = n_sequences // 2
    start_row = half * SEQ_LEN
    slice_table = table.slice(start_row, n_rows - start_row)
    n_second = slice_table.num_rows // SEQ_LEN
    log.info("Вторая половина: последовательности [%d, %d), всего %d", half, n_sequences, n_second)

    X, Y = _table_to_xy(slice_table, FEATURE_COLUMNS, TARGET_COLUMNS, SEQ_LEN)
    X_t = torch.from_numpy(X)
    all_f1 = []
    with torch.no_grad():
        for i in tqdm(range(0, X_t.shape[0], batch_size), desc="LSTM (train)", unit="batch"):
            batch = X_t[i : i + batch_size].to(DEVICE)
            f1 = lstm.forward_sequence(batch)
            all_f1.append(f1.cpu().numpy())
    F1 = np.concatenate(all_f1, axis=0)

    residual_flat, y_w_mean, f1_w_mean = _residual_from_xy(X, Y, F1, WEIGHT_EPS)
    _write_residual_parquet(
        slice_table, slice_table.column_names, residual_flat, TARGET_COLUMNS, output_parquet
    )

    meta = {"y_w_mean": y_w_mean.tolist(), "f1_w_mean": f1_w_mean.tolist()}
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Метаданные центровки: %s", metadata_path)
    return meta


def build_valid_residual(
    valid_path: str,
    lstm_run_dir: str,
    output_parquet: str,
    metadata_path: str,
    batch_size: int = SEQUENCE_BATCH_SIZE,
):
    """
    Строит датасет остатков для valid.parquet.
    Использует y_w_mean, f1_w_mean из metadata_path (с train).
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Сначала постройте train residual (чтобы появился {metadata_path})"
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    y_w_mean = np.array(meta["y_w_mean"], dtype=np.float32)
    f1_w_mean = np.array(meta["f1_w_mean"], dtype=np.float32)

    log.info("Загрузка LSTM из %s", lstm_run_dir)
    lstm, _ = load_lstm_config_and_model(lstm_run_dir)
    lstm.eval()

    log.info("Чтение valid.parquet: %s", valid_path)
    table = pq.read_table(valid_path)
    n_rows = table.num_rows
    n_seq = n_rows // SEQ_LEN
    X, Y = _table_to_xy(table, FEATURE_COLUMNS, TARGET_COLUMNS, SEQ_LEN)
    log.info("Валидация: %d последовательностей", n_seq)

    X_t = torch.from_numpy(X)
    all_f1 = []
    with torch.no_grad():
        for i in tqdm(range(0, X_t.shape[0], batch_size), desc="LSTM (valid)", unit="batch"):
            batch = X_t[i : i + batch_size].to(DEVICE)
            f1 = lstm.forward_sequence(batch)
            all_f1.append(f1.cpu().numpy())
    F1 = np.concatenate(all_f1, axis=0)
    F1 = np.clip(F1, constants.PRED_CLIP_LOW, constants.PRED_CLIP_HIGH)

    residual_flat = ((Y.reshape(-1, 2) - y_w_mean) - (F1.reshape(-1, 2) - f1_w_mean)).astype(
        np.float32
    )
    _write_residual_parquet(table, table.column_names, residual_flat, TARGET_COLUMNS, output_parquet)
    log.info("Готово: valid residual в %s", output_parquet)


def main():
    parser = argparse.ArgumentParser(description="Построение parquet с остатками для SSM")
    parser.add_argument("--train", default=TRAIN_PATH, help="Путь к train.parquet")
    parser.add_argument("--valid", default=VAL_PATH, help="Путь к valid.parquet")
    parser.add_argument("--lstm-run", default=LSTM_RUN_DIR, help="Директория с весами LSTM")
    parser.add_argument(
        "--train-output",
        default=TRAIN_RESIDUAL_PATH,
        help="Путь к train_residual.parquet",
    )
    parser.add_argument(
        "--valid-output",
        default=VALID_RESIDUAL_PATH,
        help="Путь к valid_residual.parquet",
    )
    parser.add_argument(
        "--metadata",
        default=RESIDUAL_METADATA_PATH,
        help="Путь к residual_metadata.json",
    )
    parser.add_argument("--train-only", action="store_true", help="Только train residual")
    parser.add_argument("--valid-only", action="store_true", help="Только valid residual")
    args = parser.parse_args()

    do_train = not args.valid_only
    do_valid = not args.train_only

    if do_train and not os.path.isfile(args.train):
        log.error("Не найден train.parquet: %s", args.train)
        sys.exit(1)
    if do_valid and not os.path.isfile(args.valid):
        log.error("Не найден valid.parquet: %s", args.valid)
        sys.exit(1)

    if do_train:
        build_train_residual(
            train_path=args.train,
            lstm_run_dir=args.lstm_run,
            output_parquet=args.train_output,
            metadata_path=args.metadata,
        )
    if do_valid:
        build_valid_residual(
            valid_path=args.valid,
            lstm_run_dir=args.lstm_run,
            output_parquet=args.valid_output,
            metadata_path=args.metadata,
        )


if __name__ == "__main__":
    main()
