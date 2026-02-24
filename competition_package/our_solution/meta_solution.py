"""
Точка входа с метамоделью (stack): FullModel + GRUModel + meta_head.
Полный функционал как в solution.py: класс PredictionModel с predict(data_point).
Состояние обеих базовых моделей сбрасывается по seq_ix внутри StackModel.forward.
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

# Пути к конфигам и весам
INFERENCE_DIR = os.path.join(ROOT, "inference")
FULL_MODEL_CONFIG_PATH = os.path.join(INFERENCE_DIR, "config.json")
FULL_MODEL_WEIGHTS_PATH = os.path.join(INFERENCE_DIR, "weights", "best_model.pt")
GRU_WEIGHTS_PATH = os.path.join(ROOT, "inference_gru", "weights", "best_model.pt")
GRU_CONFIG_PATH = os.path.join(ROOT, "inference_gru", "config.json")
STACK_DIR = os.path.join(ROOT, "stack")
STACK_CONFIG_PATH = os.path.join(STACK_DIR, "best_hyperparameters_stack.json")
STACK_WEIGHTS_PATH = os.path.join(STACK_DIR, "weights_stack", "best_stack.pt")

PRED_CLIP_LOW, PRED_CLIP_HIGH = -6.0, 6.0

# Размеры meta_head по умолчанию, если нет best_hyperparameters_stack.json
DEFAULT_META_HIDDEN_DIMS = [64]


def _apply_convo_config(config: dict) -> None:
    import constants
    constants.convo_constants.clear()
    constants.convo_constants.update(config)


def _load_full_model():
    with open(FULL_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
    _apply_convo_config(config)
    from model import FullModel
    from constants import DEVICE
    model = FullModel()
    model.load_state_dict(torch.load(FULL_MODEL_WEIGHTS_PATH, map_location="cpu"), strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def _load_gru_model():
    import constants
    if os.path.isfile(GRU_CONFIG_PATH):
        with open(GRU_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("best_hyperparameters", data) if isinstance(data, dict) else data
        constants.gru_constants.clear()
        constants.gru_constants.update(config)
    from model import GRUModel
    from constants import DEVICE
    model = GRUModel()
    model.load_state_dict(torch.load(GRU_WEIGHTS_PATH, map_location="cpu"), strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def _get_meta_hidden_dims() -> list:
    if os.path.isfile(STACK_CONFIG_PATH):
        with open(STACK_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        hp = data.get("best_hyperparameters", {})
        if "meta_hidden_dims" in hp:
            return hp["meta_hidden_dims"]
    return DEFAULT_META_HIDDEN_DIMS


def _load_stack_model():
    from constants import DEVICE
    from stack import StackModel

    full_model = _load_full_model()
    gru_model = _load_gru_model()
    meta_hidden_dims = _get_meta_hidden_dims()

    stack_model = StackModel(
        full_model=full_model,
        gru_model=gru_model,
        meta_hidden_dims=meta_hidden_dims,
    )
    stack_model.load_state_dict(torch.load(STACK_WEIGHTS_PATH, map_location="cpu"), strict=True)
    stack_model = stack_model.to(DEVICE)
    stack_model.eval()
    return stack_model


class PredictionModel:
    """
    Класс из README. Внутри — StackModel (FullModel + GRUModel + meta_head).
    Смена состояния по seq_ix выполняется в forward базовых моделей.
    """

    def __init__(self):
        self._stack_model = _load_stack_model()

    @property
    def stack_model(self):
        return self._stack_model

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        if not data_point.need_prediction:
            return None
        dev = next(self._stack_model.parameters()).device
        x = torch.from_numpy(data_point.state).float().to(dev)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self._stack_model.forward(x, seq_ix=data_point.seq_ix)
        if out.dim() == 2:
            out = out[-1]
        pred = out.cpu().numpy().flatten()
        return np.clip(pred, PRED_CLIP_LOW, PRED_CLIP_HIGH).astype(np.float64)


__all__ = ["PredictionModel", "DataPoint"]
