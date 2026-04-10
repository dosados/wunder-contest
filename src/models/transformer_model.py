from __future__ import annotations
from typing import Any, Optional
import torch
import torch.nn as nn
from constants import DEVICE, transformer_constants
from models.model_state import RingBuffer


def _get_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    defaults = dict(transformer_constants)
    if config:
        defaults.update(config)
    return defaults


def _sliding_band_mask(
    length: int, window: int, device: torch.device
) -> torch.Tensor:
    """Bool attention mask: True blocks. Causal attention with at most ``window`` positions."""
    i = torch.arange(length, device=device).unsqueeze(1)
    j = torch.arange(length, device=device).unsqueeze(0)
    causal_ok = j <= i
    band_ok = j >= (i - (window - 1)).clamp(min=0)
    return ~(causal_ok & band_ok)


def _streaming_window_masks(
    seq_len: int, window: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Left-pad to ``window``; block padded keys and future positions."""
    pad = window - seq_len
    i = torch.arange(window, device=device).unsqueeze(1)
    j = torch.arange(window, device=device).unsqueeze(0)
    causal_block = j > i
    pad_key_block = j < pad
    attn_mask = causal_block | pad_key_block
    if pad > 0:
        key_padding = torch.zeros(1, window, dtype=torch.bool, device=device)
        key_padding[0, :pad] = True
    else:
        key_padding = None
    return attn_mask, key_padding


class WindowTransformerModel(nn.Module):

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = _get_config(config)
        self.input_dim = cfg["input_dim"]
        self.d_model = cfg["d_model"]
        self.window = int(cfg["context_window"])
        self.output_dim = cfg["output_dim"]
        n_heads = cfg["n_heads"]
        if self.d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg.get("dropout", 0.0),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg["n_layers"])
        self.head0 = nn.Linear(self.d_model, 1)
        self.head1 = nn.Linear(self.d_model, 1)
        self.residual_proj = nn.Linear(self.input_dim, self.output_dim)
        self._buffer = RingBuffer(self.window, self.d_model, DEVICE)
        self._last_seq_ix: int | None = None
        self._band_mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def reset_state(self) -> None:
        self._buffer.reset()
        self._last_seq_ix = None

    def _band_mask_for_length(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, self.window, str(device))
        cached = self._band_mask_cache.get(key)
        if cached is None or cached.device != device:
            self._band_mask_cache[key] = _sliding_band_mask(
                length, self.window, device
            )
        return self._band_mask_cache[key]

    def _encode_padded_window(self) -> torch.Tensor:
        seq = self._buffer.get_sequence()
        L = seq.shape[0]
        attn_mask, key_pad = _streaming_window_masks(L, self.window, seq.device)
        if L < self.window:
            pad_rows = self.window - L
            pad = torch.zeros(
                pad_rows, self.d_model, device=seq.device, dtype=seq.dtype
            )
            seq = torch.cat([pad, seq], dim=0)
        seq_b = seq.unsqueeze(0)
        h = self.encoder(seq_b, mask=attn_mask, src_key_padding_mask=key_pad)
        return h[0, -1]

    def forward(self, x: torch.Tensor, seq_ix: int | None = None) -> torch.Tensor:
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix
        residual = self.residual_proj(x)
        if x.dim() == 1:
            self._buffer.add(self.input_proj(x))
            h = self._encode_padded_window()
            return (
                torch.cat([self.head0(h), self.head1(h)], dim=-1) + residual
            )
        outs = []
        for i in range(x.shape[0]):
            self._buffer.add(self.input_proj(x[i]))
            h = self._encode_padded_window()
            outs.append(
                torch.cat([self.head0(h), self.head1(h)], dim=-1) + residual[i]
            )
        return torch.stack(outs, dim=0)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        _, t, _ = x.shape
        residual = self.residual_proj(x)
        e = self.input_proj(x)
        mask = self._band_mask_for_length(t, e.device)
        out = self.encoder(e, mask=mask)
        pred = torch.cat([self.head0(out), self.head1(out)], dim=-1) + residual
        return pred.squeeze(0) if squeeze else pred
