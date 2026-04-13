from __future__ import annotations
from typing import Any, Optional
import torch
import torch.nn as nn
from constants import INPUT_DIM, TARGET_DIM, ssm_constants
from models.configurable_blocks import build_input_projection, merge_block_cfg
from models.configurable_blocks import ConfigurableLinearBlock
from .ssm_mamba_block import MambaBlock
from .ssm_model_state import MambaBlockState, SSMModelState


def _get_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    defaults = {
        "input_dim": INPUT_DIM,
        "d_model": 64,
        "n_layers": 2,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": None,
        "output_dim": TARGET_DIM,
    }
    defaults.update(ssm_constants)
    if config:
        defaults.update(config)
    return defaults


def _build_post_trunk(
    in_dim: int,
    post_stack: dict[str, Any] | None,
    global_block: dict[str, Any] | None,
) -> tuple[nn.Module, int]:
    if not post_stack:
        return nn.Identity(), in_dim
    layer_dims = list(post_stack.get("hidden_dims") or [])
    if not layer_dims:
        return nn.Identity(), in_dim
    cfg = merge_block_cfg(post_stack, global_block)
    dims = [in_dim] + layer_dims
    blocks = [
        ConfigurableLinearBlock(dims[i], dims[i + 1], cfg)
        for i in range(len(dims) - 1)
    ]
    trunk = nn.Sequential(*blocks) if len(blocks) > 1 else blocks[0]
    return trunk, layer_dims[-1]


class SSMModel(nn.Module):

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = _get_config(config)
        self.input_dim = cfg["input_dim"]
        self.d_model = cfg["d_model"]
        self.n_layers = cfg["n_layers"]
        self.output_dim = cfg["output_dim"]
        gb = cfg.get("block")
        self.input_proj = build_input_projection(
            cfg["input_dim"], self.d_model, cfg.get("input_stack"), gb
        )
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=self.d_model,
                    d_state=cfg["d_state"],
                    d_conv=cfg["d_conv"],
                    expand=cfg["expand"],
                    dt_rank=cfg.get("dt_rank"),
                    layer_idx=i,
                )
                for i in range(self.n_layers)
            ]
        )
        self.post_trunk, head_in = _build_post_trunk(
            self.d_model, cfg.get("post_stack"), gb
        )
        self.head0 = nn.Linear(head_in, 1)
        self.head1 = nn.Linear(head_in, 1)
        self.residual_proj = nn.Linear(cfg["input_dim"], self.output_dim)
        self._d_conv = cfg["d_conv"]
        self._d_inner = int(cfg["expand"] * self.d_model)
        self._d_state = cfg["d_state"]
        self._streaming_state: Optional[SSMModelState] = None
        self._last_seq_ix: Optional[int] = None

    def _get_state(self, device: torch.device) -> SSMModelState:
        if self._streaming_state is None:
            self._streaming_state = SSMModelState(
                [
                    MambaBlockState(self._d_conv, self._d_inner, self._d_state, device)
                    for _ in range(self.n_layers)
                ]
            )
        return self._streaming_state

    def reset_state(self) -> None:
        if self._streaming_state is not None:
            self._streaming_state.reset()
        self._last_seq_ix = None

    def forward(self, x: torch.Tensor, seq_ix: Optional[int] = None) -> torch.Tensor:
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix
        state = self._get_state(x.device)
        residual = self.residual_proj(x)
        if x.dim() == 1:
            x_proj = self.input_proj(x.unsqueeze(0)).squeeze(0)
            for block, block_state in zip(self.blocks, state.block_states):
                x_proj = block.forward_step(x_proj.unsqueeze(0), block_state).squeeze(0)
            x_proj = self.post_trunk(x_proj)
            out = torch.stack(
                [
                    self.head0(x_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
                    self.head1(x_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
                ],
                dim=-1,
            )
            return out + residual
        outs = []
        for i in range(x.shape[0]):
            xi = x[i]
            ri = residual[i]
            xi_proj = self.input_proj(xi.unsqueeze(0)).squeeze(0)
            for block, block_state in zip(self.blocks, state.block_states):
                xi_proj = block.forward_step(xi_proj.unsqueeze(0), block_state).squeeze(
                    0
                )
            xi_proj = self.post_trunk(xi_proj)
            out_i = torch.stack(
                [
                    self.head0(xi_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
                    self.head1(xi_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
                ],
                dim=-1,
            )
            outs.append(out_i + ri)
        return torch.stack(outs, dim=0)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        residual = self.residual_proj(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_trunk(x)
        out = torch.cat([self.head0(x), self.head1(x)], dim=-1) + residual
        return out.squeeze(0) if squeeze else out
