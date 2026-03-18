"""
Блок Mamba: проекция входа, causal conv1d, selective SSM, gate и residual.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm_core import SelectiveSSM, selective_scan_step
from .model_state import MambaBlockState


class MambaBlock(nn.Module):
    """
    Один блок Mamba: in_proj -> split (x, z) -> conv1d(x) -> SiLU -> x_proj(delta,B,C) -> SSM -> * silu(z) -> out_proj.
    Causal conv: padding = kernel_size - 1, затем срез [..., :seqlen].
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else max(1, self.d_inner // 16)
        self.layer_idx = layer_idx

        factory = {"device": device, "dtype": dtype}

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory,
        )

        # x_proj: из d_inner в dt_rank + d_state*2 (для delta_raw, B, C)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory,
        )
        dt_scale = self.dt_rank**-0.5
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory)
        nn.init.uniform_(self.dt_proj.weight, -dt_scale, dt_scale)
        # bias для dt уже инициализирован в SelectiveSSM; здесь мы передаём delta = dt_proj(x_dbl[:,:dt_rank])
        # и в SSM добавляется delta_bias. Поэтому dt_proj.bias можно оставить как в SelectiveSSM.
        # В оригинале dt_proj.bias задаётся как inv_dt. Чтобы не дублировать, вынесем инициализацию
        # в один раз: в SelectiveSSM храним dt_bias, а dt_proj делаем без bias и добавляем bias в блоке?
        # Проще: в блоке dt_proj с bias=True, инициализируем bias как inv_dt (как в mamba_simple).
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner, **factory)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        self.ssm = SelectiveSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            dt_rank=self.dt_rank,
            device=device,
            dtype=dtype,
        )

        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, L, d_model)
        Возвращает (B, L, d_model).
        """
        B, L, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # Causal conv: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :L]
        x = x.transpose(1, 2)
        x = self.act(x)

        # x_proj: (B, L, d_inner) -> (B, L, dt_rank + 2*d_state)
        x_dbl = self.x_proj(x)
        dt_raw, B_C = x_dbl.split([self.dt_rank, self.d_state * 2], dim=-1)
        dt = self.dt_proj(dt_raw)
        B, C = B_C.chunk(2, dim=-1)

        # SSM ожидает B, C (B, L, N)
        y = self.ssm(u=x, delta=dt, B=B, C=C, delta_bias=None)

        y = y * self.act(z)
        out = self.out_proj(y)
        return out + hidden_states

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        state: MambaBlockState,
    ) -> torch.Tensor:
        """
        Один шаг (streaming): hidden_states (d_model,) или (1, d_model).
        state обновляется in-place. Возвращает (d_model,) или (1, d_model).
        """
        squeeze = hidden_states.dim() == 1
        if squeeze:
            hidden_states = hidden_states.unsqueeze(0)

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        state.conv_buffer.add(x.squeeze(0))
        seq = state.conv_buffer.get_sequence()
        if seq.dim() == 2 and seq.shape[0] < self.d_conv:
            pad_len = self.d_conv - seq.shape[0]
            pad = torch.zeros(
                pad_len, self.d_inner, device=seq.device, dtype=seq.dtype
            )
            seq = torch.cat([pad, seq], dim=0)
        seq = seq.unsqueeze(0).transpose(1, 2)
        x_conv = self.conv1d(seq)[..., -1:]
        x_conv = x_conv.squeeze(-1).squeeze(0)
        x_conv = self.act(x_conv)

        x_dbl = self.x_proj(x_conv.unsqueeze(0))
        dt_raw, B_C = x_dbl.split([self.dt_rank, self.d_state * 2], dim=-1)
        dt = self.dt_proj(dt_raw).squeeze(0)
        B_t, C_t = B_C.squeeze(0).chunk(2, dim=-1)

        A = -torch.exp(self.ssm.A_log.to(x_conv.dtype))
        D = self.ssm.D.to(x_conv.dtype)
        h = state.get_or_create_h(x_conv.dtype)
        h_new, y_ssm = selective_scan_step(
            h, x_conv, dt, B_t, C_t, A, D, delta_softplus=True
        )
        state.h = h_new.detach()

        y = y_ssm * self.act(z.squeeze(0))
        out = self.out_proj(y.unsqueeze(0)) + hidden_states
        if squeeze:
            out = out.squeeze(0)
        return out
