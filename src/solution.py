from __future__ import annotations
import os
import numpy as np
import torch
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
    return (
        STACK_LSTM_GRU_META_PATH if variant == "lstm_gru" else STACK_LSTM_SSM_META_PATH
    )


class PredictionModel:

    def __init__(self):
        variant = (
            os.getenv("ORCHESTRATION_VARIANT", DEFAULT_ORCHESTRATION_VARIANT)
            .strip()
            .lower()
        )
        if variant not in ORCHESTRATION_VARIANTS:
            raise ValueError(
                f"ORCHESTRATION_VARIANT must be one of {ORCHESTRATION_VARIANTS}, got: {variant}"
            )
        self._pair = build_base_pair(variant)
        self._state_np = np.empty(32, dtype=np.float32)
        self._x_buf = torch.empty(1, 32, dtype=torch.float32, device=DEVICE)
        meta_path = _meta_path_for_variant(variant)
        if os.path.isfile(meta_path):
            meta_pair = PairOrchestrator(
                self._pair.model_a, self._pair.model_b, mode="meta"
            ).to(DEVICE)
            meta_pair.meta_head.load_state_dict(
                torch.load(meta_path, map_location="cpu"), strict=True
            )
            self._pair = meta_pair.eval()

    def predict(self, data_point):
        if not data_point.need_prediction:
            return None
        np.copyto(self._state_np, np.asarray(data_point.state, dtype=np.float32))
        self._x_buf[0].copy_(torch.from_numpy(self._state_np))
        with torch.inference_mode():
            out = self._pair(self._x_buf, seq_ix=data_point.seq_ix)
        if out.dim() == 2:
            out = out.squeeze(0)
        return np.clip(out.cpu().numpy(), PRED_CLIP_LOW, PRED_CLIP_HIGH).astype(
            np.float64
        )
