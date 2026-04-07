"""
Unified submission script with two orchestration variants:
- lstm_gru
- lstm_ssm
If meta-head weights exist, inference uses meta ensemble; otherwise raw sum.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from constants import (
    DEFAULT_ORCHESTRATION_VARIANT,
    DEVICE,
    ORCHESTRATION_VARIANTS,
    PRED_CLIP_HIGH,
    PRED_CLIP_LOW,
    STACK_LSTM_GRU_META_PATH,
    STACK_LSTM_SSM_META_PATH,
)
from orchestration.models import PairOrchestrator, build_base_pair


def _meta_path_for_variant(variant: str) -> str:
    if variant == "lstm_gru":
        return STACK_LSTM_GRU_META_PATH
    if variant == "lstm_ssm":
        return STACK_LSTM_SSM_META_PATH
    raise ValueError(f"Unknown variant: {variant}")


class PredictionModel:
    def __init__(self):
        variant = os.getenv("ORCHESTRATION_VARIANT", DEFAULT_ORCHESTRATION_VARIANT).strip().lower()
        if variant not in ORCHESTRATION_VARIANTS:
            raise ValueError(f"ORCHESTRATION_VARIANT must be one of {ORCHESTRATION_VARIANTS}, got: {variant}")
        self._variant = variant
        self._pair = build_base_pair(variant)
        self._device = DEVICE
        self._state_np = np.empty(32, dtype=np.float32)
        self._x_buf = torch.empty(1, 32, dtype=torch.float32, device=self._device)

        meta_path = _meta_path_for_variant(variant)
        self._use_meta = False
        if os.path.isfile(meta_path):
            # Build a meta-capable pair and load only meta-head from OOF training.
            meta_pair = PairOrchestrator(
                self._pair.model_a,
                self._pair.model_b,
                mode="meta",
            ).to(self._device)
            meta_pair.meta_head.load_state_dict(torch.load(meta_path, map_location="cpu"), strict=True)
            self._pair = meta_pair.eval()
            self._use_meta = True

    def predict(self, data_point) -> np.ndarray | None:
        if not data_point.need_prediction:
            return None
        np.copyto(self._state_np, np.asarray(data_point.state, dtype=np.float32))
        self._x_buf[0].copy_(torch.from_numpy(self._state_np))
        with torch.inference_mode():
            out = self._pair(self._x_buf, seq_ix=data_point.seq_ix)
        if out.dim() == 2:
            out = out.squeeze(0)
        prediction = np.clip(out.cpu().numpy(), PRED_CLIP_LOW, PRED_CLIP_HIGH)
        return prediction.astype(np.float64)
