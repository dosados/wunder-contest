import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_state import ModelState
from constants import convo_constants, DEVICE

current_constants = convo_constants


class StreamingTemporalConvModel(nn.Module):
    """
    Один causal Conv1D слой.
    Streaming: один вектор (D,) за раз, состояние в ModelState.
    Обучение: forward_sequence(B, T, D) — один проход по всей последовательности.
    """

    def __init__(self, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.GELU()

        linear1_dim = current_constants.get("linear1_dim")
        conv_dim = current_constants.get("conv_dim")
        kernel = current_constants.get("conv_window")

        self.conv = nn.Conv1d(linear1_dim, conv_dim, kernel, bias=True)
        self.activation = activation
        self.kernel = kernel
        self.state = ModelState(kernel, linear1_dim, DEVICE)

    def _forward_conv_step(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (T, D). Causal свёртка, возвращает один вектор (conv_dim,)."""
        T, D = seq.shape
        kernel_size = self.kernel
        if T < kernel_size:
            pad_len = kernel_size - T
            pad = torch.zeros(pad_len, D, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([pad, seq], dim=0)
        else:
            seq = seq[-kernel_size:]
        x = seq.transpose(0, 1).unsqueeze(0)  # (1, D, K)
        out = self.conv(x)
        return out.squeeze(0).squeeze(-1)  # (conv_dim,)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Один шаг (streaming). x_t: (linear1_dim,).
        Обновляет state, возвращает (conv_dim,).
        """
        self.state.add_conv(x_t)
        seq = self.state.get_conv_sequence()
        y = self._forward_conv_step(seq)
        return self.activation(y)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, C) или (B, T, C), C = linear1_dim.
        Возвращает (T, conv_dim) или (B, T, conv_dim). Causal padding.
        """
        squeeze = False
        if x.dim() == 2:
            squeeze = True
            x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel - 1, 0), mode="constant", value=0)
        x = self.conv(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        if squeeze:
            x = x.squeeze(0)
        return x

    def reset_state(self):
        self.state.reset()
