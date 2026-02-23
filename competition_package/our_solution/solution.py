"""
Точка входа для сдачи решения (README).
Класс PredictionModel с методом predict(data_point) -> np.ndarray | None.
Использует FullModel (сброс состояния по seq_ix — внутри FullModel.forward).
"""
import os
import sys
import json
from typing import Optional

import numpy as np
import torch

# Корень решения и пакет с utils
ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(ROOT)
for path in (ROOT, PKG):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import DataPoint

# Пути к конфигу и весам (inference/)
CONFIG_PATH = os.path.join(ROOT, "inference", "config.json")
WEIGHTS_PATH = os.path.join(ROOT, "inference", "weights", "best_model.pt")
PRED_CLIP_LOW, PRED_CLIP_HIGH = -6.0, 6.0


def _apply_config(config: dict) -> None:
    import constants
    constants.convo_constants.clear()
    constants.convo_constants.update(config)


def _load_full_model(config_path: str, weights_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
    _apply_config(config)
    from model import FullModel
    from constants import DEVICE
    model = FullModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


class PredictionModel:
    """
    Класс из README. Внутри — FullModel с загруженными весами.
    Смена состояния по последовательности (seq_ix) выполняется в FullModel.forward.
    """

    def __init__(self):
        self._full_model = _load_full_model(CONFIG_PATH, WEIGHTS_PATH)

    @property
    def full_model(self):
        return self._full_model

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        if not data_point.need_prediction:
            return None
        dev = next(self._full_model.parameters()).device
        x = torch.from_numpy(data_point.state).float().to(dev)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self._full_model.forward(x, seq_ix=data_point.seq_ix)
        if out.dim() == 2:
            out = out[-1]
        pred = out.cpu().numpy().flatten()
        return np.clip(pred, PRED_CLIP_LOW, PRED_CLIP_HIGH).astype(np.float64)


__all__ = ["PredictionModel", "DataPoint"]
