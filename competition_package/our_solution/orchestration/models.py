from __future__ import annotations

import json
import os
from typing import Iterable

import torch
import torch.nn as nn

import constants
from constants import (
    DEFAULT_META_HIDDEN_DIMS,
    DEVICE,
    INFERENCE_GRU_CONFIG_PATH,
    INFERENCE_GRU_WEIGHTS_PATH,
    LSTM_RUN_DIR,
    SAVE_NAME_SSM,
    WEIGHTS_SSM_DIR,
)


def _apply_lstm_config(config: dict) -> None:
    constants.convo_constants.clear()
    constants.convo_constants.update(config)


def _apply_gru_config(config: dict) -> None:
    constants.gru_constants.clear()
    constants.gru_constants.update(config)


def _load_lstm_from_run(run_dir: str) -> nn.Module:
    config_path = os.path.join(run_dir, "config.json")
    weights_path = os.path.join(run_dir, "best_model.pt")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("model", data.get("best_hyperparameters", data))
    _apply_lstm_config(cfg)
    from model import FullModel

    model = FullModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model.to(DEVICE).eval()


def _load_gru_from_inference(
    config_path: str = INFERENCE_GRU_CONFIG_PATH,
    weights_path: str = INFERENCE_GRU_WEIGHTS_PATH,
) -> nn.Module:
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("best_hyperparameters", data)
        _apply_gru_config(cfg)
    from model import GRUModel

    model = GRUModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model.to(DEVICE).eval()


def _load_ssm(weights_path: str) -> nn.Module:
    from ssm_model import SSMModel

    model = SSMModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model.to(DEVICE).eval()


class PairOrchestrator(nn.Module):
    """
    Универсальный оркестратор двух базовых моделей.
    - mode=sum: итог = m1 + m2
    - mode=meta: итог = meta_head([m1, m2]), где concat размерности 4
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        mode: str = "sum",
        meta_hidden_dims: Iterable[int] | None = None,
        output_dim: int = 2,
    ):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.mode = mode
        self.output_dim = output_dim
        hidden = list(meta_hidden_dims) if meta_hidden_dims is not None else list(DEFAULT_META_HIDDEN_DIMS)

        if mode == "meta":
            self.meta_head = self._build_meta_head(input_dim=output_dim * 2, output_dim=output_dim, hidden=hidden)
        elif mode == "sum":
            self.meta_head = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _build_meta_head(input_dim: int, output_dim: int, hidden: list[int]) -> nn.Module:
        if not hidden:
            return nn.Linear(input_dim, output_dim)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, seq_ix: int | None = None) -> torch.Tensor:
        pa = self.model_a(x, seq_ix=seq_ix)
        pb = self.model_b(x, seq_ix=seq_ix)
        if self.mode == "sum":
            return pa + pb
        return self.meta_head(torch.cat([pa, pb], dim=-1))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        pa = self.model_a.forward_sequence(x)
        pb = self.model_b.forward_sequence(x)
        if self.mode == "sum":
            return pa + pb
        s = torch.cat([pa, pb], dim=-1)
        if s.dim() == 2:
            return self.meta_head(s)
        b, t, d = s.shape
        return self.meta_head(s.reshape(-1, d)).reshape(b, t, self.output_dim)

    def set_backbones_trainable(self, trainable: bool) -> None:
        for p in self.model_a.parameters():
            p.requires_grad = trainable
        for p in self.model_b.parameters():
            p.requires_grad = trainable


def build_base_pair(variant: str) -> PairOrchestrator:
    lstm = _load_lstm_from_run(LSTM_RUN_DIR)
    if variant == "lstm_gru":
        pair = PairOrchestrator(lstm, _load_gru_from_inference(), mode="sum")
    elif variant == "lstm_ssm":
        pair = PairOrchestrator(lstm, _load_ssm(os.path.join(WEIGHTS_SSM_DIR, SAVE_NAME_SSM)), mode="sum")
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return pair.to(DEVICE).eval()


def load_meta_head_state(pair: PairOrchestrator, weights_path: str) -> PairOrchestrator:
    state = torch.load(weights_path, map_location="cpu")
    pair.meta_head.load_state_dict(state, strict=True)
    pair.mode = "meta"
    return pair.to(DEVICE).eval()
