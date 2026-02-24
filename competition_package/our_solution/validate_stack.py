"""
Валидация стека (метамодели) по weighted Pearson на указанном parquet-файле.
Метрика совпадает с той, что используется при валидации в stack_trainer / stack_hyperparameter_search.
Запуск: python validate_stack.py --data путь/к/данным.parquet [--weights путь/к/весам.pt]
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(ROOT)
for path in (ROOT, PKG):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import weighted_pearson_correlation
from dataset import ParquetSequenceDataset

# Константы как в stack_trainer (warmup, колонки, batch size)
FEATURE_COLUMNS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)
TARGET_COLUMNS = ["t0", "t1"]
WARMUP_STEPS = 99
SEQUENCE_BATCH_SIZE = 16

STACK_DIR = os.path.join(ROOT, "stack")
STACK_CONFIG_PATH = os.path.join(STACK_DIR, "best_hyperparameters_stack.json")
DEFAULT_STACK_WEIGHTS_PATH = os.path.join(STACK_DIR, "weights_stack", "best_stack.pt")
DEFAULT_META_HIDDEN_DIMS = [64]


def _get_meta_hidden_dims(stack_config_path: str) -> list:
    if os.path.isfile(stack_config_path):
        with open(stack_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hp = data.get("best_hyperparameters", {})
        if "meta_hidden_dims" in hp:
            return hp["meta_hidden_dims"]
    return DEFAULT_META_HIDDEN_DIMS


def load_stack_model(weights_path: str, device: str):
    """Загружает FullModel + GRUModel + meta_head и веса стека."""
    import constants
    from model import FullModel, GRUModel
    from stack import StackModel

    # FullModel
    inference_dir = os.path.join(ROOT, "inference")
    config_path = os.path.join(inference_dir, "config.json")
    full_weights = os.path.join(inference_dir, "weights", "best_model.pt")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
    constants.convo_constants.clear()
    constants.convo_constants.update(config)
    full_model = FullModel()
    full_model.load_state_dict(torch.load(full_weights, map_location="cpu"), strict=True)

    # GRUModel
    gru_dir = os.path.join(ROOT, "inference_gru")
    gru_config = os.path.join(gru_dir, "config.json")
    gru_weights = os.path.join(gru_dir, "weights", "best_model.pt")
    if os.path.isfile(gru_config):
        with open(gru_config, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
        constants.gru_constants.clear()
        constants.gru_constants.update(config)
    gru_model = GRUModel()
    gru_model.load_state_dict(torch.load(gru_weights, map_location="cpu"), strict=True)

    meta_hidden_dims = _get_meta_hidden_dims(STACK_CONFIG_PATH)
    stack_model = StackModel(
        full_model=full_model,
        gru_model=gru_model,
        meta_hidden_dims=meta_hidden_dims,
    )
    stack_model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    stack_model = stack_model.to(device)
    stack_model.eval()
    return stack_model


def compute_pearson_on_path(data_path: str, model, device: str, batch_size: int = SEQUENCE_BATCH_SIZE) -> float:
    """Считает weighted Pearson (среднее по t0 и t1) на данных из data_path, без warmup шагов."""
    ds = ParquetSequenceDataset(
        data_path,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model.forward_sequence(x)
            p = pred[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            t = y[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy()
            preds_list.append(p)
            targets_list.append(t)
    preds_all = np.concatenate(preds_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    corr0 = weighted_pearson_correlation(targets_all[:, 0], preds_all[:, 0])
    corr1 = weighted_pearson_correlation(targets_all[:, 1], preds_all[:, 1])
    return (corr0 + corr1) / 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Валидация стека по weighted Pearson на parquet-данных (как в stack_hyperparameter_search)."
    )
    parser.add_argument(
        "data",
        type=str,
        help="Путь к parquet-файлу (например datasets/valid.parquet или datasets/train.parquet)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_STACK_WEIGHTS_PATH,
        help="Путь к весам стека (по умолчанию stack/weights_stack/best_stack.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство (cuda/cpu). По умолчанию — constants.DEVICE",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SEQUENCE_BATCH_SIZE,
        help="Размер батча при проходе по данным",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.weights):
        print(f"Error: weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(1)

    from constants import DEVICE
    device = args.device or DEVICE

    print(f"Loading stack from {args.weights} ...")
    model = load_stack_model(args.weights, device)
    print(f"Computing weighted Pearson on {args.data} (warmup steps excluded: {WARMUP_STEPS}) ...")
    pearson = compute_pearson_on_path(args.data, model, device, batch_size=args.batch_size)
    print(f"Weighted Pearson (avg t0, t1): {pearson:.6f}")


if __name__ == "__main__":
    main()
