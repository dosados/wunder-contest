from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _selective_scan_parallel(dA: torch.Tensor, dB_u: torch.Tensor) -> torch.Tensor:
    _, L, _, _ = dA.shape
    scan_A = dA.clone()
    scan_b = dB_u.clone()
    s = 1
    while s < L:
        left_A = scan_A[:, : L - s, :, :]
        left_b = scan_b[:, : L - s, :, :]
        right_A = scan_A[:, s:, :, :]
        right_b = scan_b[:, s:, :, :]
        scan_A = torch.cat([scan_A[:, :s, :, :], right_A * left_A], dim=1)
        scan_b = torch.cat([scan_b[:, :s, :, :], right_A * left_b + right_b], dim=1)
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
    dtype = u.dtype
    if delta_bias is not None:
        delta = delta + delta_bias.to(dtype).unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        delta = F.softplus(delta)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).to(dtype))
    dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    h = _selective_scan_parallel(dA, dB_u)
    y = (h * C.unsqueeze(2)).sum(-1)
    if D is not None:
        y = y + D.to(dtype).unsqueeze(0).unsqueeze(0) * u
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
    if h.dim() == 2:
        h, u_t, delta_t, B_t, C_t = (
            h.unsqueeze(0),
            u_t.unsqueeze(0),
            delta_t.unsqueeze(0),
            B_t.unsqueeze(0),
            C_t.unsqueeze(0),
        )
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
        return (h_new.squeeze(0), y_t.squeeze(0))
    return (h_new, y_t)


class SelectiveSSM(nn.Module):

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
