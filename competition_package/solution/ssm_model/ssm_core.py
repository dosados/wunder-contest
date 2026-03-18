"""
Ядро Selective State Space Model (Mamba-style).
Чистый PyTorch: selective scan через параллельный prefix scan (без Python-цикла по L).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _selective_scan_parallel(
    dA: torch.Tensor,
    dB_u: torch.Tensor,
) -> torch.Tensor:
    """
    Параллельный inclusive scan: h_l = sum_{i=0..l} (prod_{j=i+1..l} dA_j)*dB_u_i.
    Ассоциативная операция: (A1, b1) * (A2, b2) = (A2*A1, A2*b1 + b2).
    Без in-place операций, чтобы autograd работал корректно.
    dA, dB_u: (B, L, D, N). Возвращает h: (B, L, D, N).
    """
    _, L, _, _ = dA.shape
    scan_A = dA.clone()
    scan_b = dB_u.clone()
    s = 1
    while s < L:
        left_A = scan_A[:, : L - s, :, :]
        left_b = scan_b[:, : L - s, :, :]
        right_A = scan_A[:, s:, :, :]
        right_b = scan_b[:, s:, :, :]
        # Новые тензоры вместо in-place, чтобы не ломать граф градиентов
        new_right_A = right_A * left_A
        new_right_b = right_A * left_b + right_b
        scan_A = torch.cat([scan_A[:, :s, :, :], new_right_A], dim=1)
        scan_b = torch.cat([scan_b[:, :s, :, :], new_right_b], dim=1)
        s *= 2
    return scan_b


def selective_scan_naive(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Selective scan: параллельный prefix scan по времени (без Python-цикла по L).
    u: (B, L, D), delta: (B, L, D), A: (D, N), B: (B, L, N), C: (B, L, N).
    Возвращает y: (B, L, D).
    """
    device = u.device
    dtype = u.dtype
    B_batch, L, D_dim = u.shape
    N = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.to(dtype).unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        delta = F.softplus(delta)

    # (B, L, D, N)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).to(dtype))
    dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

    h = _selective_scan_parallel(dA, dB_u)  # (B, L, D, N)

    # y(b,l,d) = sum_n C(b,l,n)*h(b,l,d,n) + D(d)*u(b,l,d)
    y = (h * C.unsqueeze(2)).sum(-1)  # (B, L, D)
    if D is not None:
        y = y + (D.to(dtype).unsqueeze(0).unsqueeze(0) * u)
    return y


def selective_scan_step(
    h: torch.Tensor,
    u_t: torch.Tensor,
    delta_t: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    delta_softplus: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Один шаг selective scan (для streaming).
    h: (B, D, N) или (D, N) — текущее состояние
    u_t: (B, D) или (D,) — вход на шаге t
    delta_t, B_t, C_t — на шаге t (соответствующих размеров)
    A: (D, N), D: (D,)
    Возвращает (h_new, y_t): h_new (B, D, N), y_t (B, D) или (D,).
    """
    if h.dim() == 2:
        h = h.unsqueeze(0)
        u_t = u_t.unsqueeze(0)
        delta_t = delta_t.unsqueeze(0)
        B_t = B_t.unsqueeze(0)
        C_t = C_t.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if delta_softplus:
        delta_t = F.softplus(delta_t)

    dA = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0).to(h.dtype))
    dB_u = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)
    h_new = dA * h + dB_u
    y_t = (h_new * C_t.unsqueeze(1)).sum(-1) + D.to(h.dtype) * u_t

    if squeeze:
        h_new = h_new.squeeze(0)
        y_t = y_t.squeeze(0)
    return h_new, y_t


class SelectiveSSM(nn.Module):
    """
    Один слой Selective SSM: по входам u (B, L, D) и проекциям delta, B, C
    выполняет selective scan и возвращает (B, L, D).
    A и D — параметры модуля; delta, B, C вычисляются снаружи из входа.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)
        factory = {"device": device, "dtype": dtype}

        # S4D-style A: диагональ как log(A), A = -exp(A_log)
        A = torch.arange(1, d_state + 1, device=device, dtype=torch.float32).float()
        A_log = torch.log(A).unsqueeze(0).expand(d_inner, -1).contiguous()
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(d_inner, device=device, dtype=torch.float32))

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # A = -exp(A_log) для устойчивой дискретизации
        A = -torch.exp(self.A_log.to(u.dtype))
        return selective_scan_naive(
            u,
            delta,
            A,
            B,
            C,
            D=self.D.to(u.dtype),
            delta_bias=delta_bias,
            delta_softplus=True,
        )
