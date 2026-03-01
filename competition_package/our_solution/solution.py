"""
Submission script: PredictionModel = LSTM (FullModel) + SSM в тандеме.
Веса загружаются из тех же путей, что и в check.py.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

# Корень our_solution (текущая директория solution.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import constants
from constants import (
    DEVICE,
    LSTM_RUN_DIR,
    WEIGHTS_SSM_DIR,
    SAVE_NAME_SSM,
    PRED_CLIP_LOW,
    PRED_CLIP_HIGH,
)


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


def _load_lstm(run_dir: str) -> torch.nn.Module:
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


def _load_ssm(weights_path: str) -> torch.nn.Module:
    """Загружает SSM по пути к весам (конфиг из constants.ssm_constants)."""
    from ssm_model import SSMModel

    model = SSMModel()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"Ожидается state_dict, получено: {type(state)}")
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


class PredictionModel:
    """
    Модель для сдачи: LSTM (FullModel) + SSM в тандеме.
    predict возвращает None, если need_prediction=False; иначе массив (2,) с предсказаниями t0, t1.
    """

    def __init__(self):
        lstm_run_dir = LSTM_RUN_DIR
        ssm_weights_path = os.path.join(WEIGHTS_SSM_DIR, SAVE_NAME_SSM)

        self._lstm = _load_lstm(lstm_run_dir)
        self._ssm = _load_ssm(ssm_weights_path)
        self._device = DEVICE
        # Буфер входа (1, 32) для переиспользования в predict
        self._state_np = np.empty(32, dtype=np.float32)
        self._x_buf = torch.empty(1, 32, dtype=torch.float32, device=self._device)

    def predict(self, data_point) -> np.ndarray | None:
        if not data_point.need_prediction:
            return None

        np.copyto(self._state_np, np.asarray(data_point.state, dtype=np.float32))
        self._x_buf[0].copy_(torch.from_numpy(self._state_np))

        with torch.inference_mode():
            out_lstm = self._lstm(self._x_buf, seq_ix=data_point.seq_ix)
            out_ssm = self._ssm(self._x_buf, seq_ix=data_point.seq_ix)

        if out_lstm.dim() == 2:
            out_lstm = out_lstm.squeeze(0)
        if out_ssm.dim() == 2:
            out_ssm = out_ssm.squeeze(0)

        combined = out_lstm.cpu().numpy() + out_ssm.cpu().numpy()
        prediction = np.clip(combined, PRED_CLIP_LOW, PRED_CLIP_HIGH)
        return prediction.astype(np.float64)
