import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_state import ModelState
from constants import convo_constants, DEVICE

current_constants = convo_constants


class StreamingTemporalConvModel(nn.Module):
    """
    3 causal Conv1D слоя.
    Работает в streaming-режиме: на вход получает один вектор (D,).
    Если окно не заполнено — применяется zero-padding слева.
    """

    def __init__(
        self,
        activation=nn.GELU()
    ):
        super().__init__()

        linear1_dim = current_constants.get("linear1_dim")
        hidden_dim1 = current_constants.get("conv1_dim")
        hidden_dim2 = current_constants.get("conv2_dim")
        hidden_dim3 = current_constants.get("conv3_dim")
        kernel1 = current_constants.get("conv1_window")
        kernel2 = current_constants.get("conv2_window")
        kernel3 = current_constants.get("conv3_window")

        self.conv1 = nn.Conv1d(linear1_dim, hidden_dim1, kernel1, bias=True)
        self.conv2 = nn.Conv1d(hidden_dim1, hidden_dim2, kernel2, bias=True)
        self.conv3 = nn.Conv1d(hidden_dim2, hidden_dim3, kernel3, bias=True)

        self.activation = activation

        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3

        self.state = ModelState(kernel1, kernel2, kernel3, hidden_dim1, hidden_dim2, hidden_dim3, DEVICE)

    def _forward_conv(self, conv, seq, kernel_size):
        """
        seq: (T, D)
        Делает causal свёртку с zero-padding слева при необходимости.
        Возвращает последний timestep.
        """

        T, D = seq.shape

        if T < kernel_size:
            pad_len = kernel_size - T
            pad = torch.zeros(pad_len, D, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([pad, seq], dim=0)
        else:
            seq = seq[-kernel_size:]

        # (K, D) -> (1, D, K)
        x = seq.transpose(0, 1).unsqueeze(0)

        # Conv1d без встроенного padding
        out = conv(x)

        # (1, C, 1) -> (C,)
        return out.squeeze(0).squeeze(-1)

    def forward(self, x_t: torch.Tensor):
        """
        x_t: (input_dim,)
        state: ModelState

        Возвращает: (hidden_dim3,)
        """

        # ---- Conv1 ----
        self.state.add_conv1(x_t)
        seq1 = self.state.get_conv1_sequence()
        y1 = self._forward_conv(self.conv1, seq1, self.kernel1)
        y1 = self.activation(y1)

        # ---- Conv2 ----
        self.state.add_conv2(y1)
        seq2 = self.state.get_conv2_sequence()
        y2 = self._forward_conv(self.conv2, seq2, self.kernel2)
        y2 = self.activation(y2)

        # ---- Conv3 ----
        self.state.add_conv3(y2)
        seq3 = self.state.get_conv3_sequence()
        y3 = self._forward_conv(self.conv3, seq3, self.kernel3)
        y3 = self.activation(y3)

        return y3

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Один проход по всей последовательности (для обучения).
        x: (T, C) или (B, T, C), C = linear1_dim.
        Возвращает: (T, conv3_dim) или (B, T, conv3_dim).
        Causal padding — те же веса, что и в streaming forward.
        """
        squeeze = False
        if x.dim() == 2:
            squeeze = True
            x = x.unsqueeze(0)  # (1, T, C)
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel1 - 1, 0), mode="constant", value=0)
        x = self.conv1(x)
        x = self.activation(x)
        x = F.pad(x, (self.kernel2 - 1, 0), mode="constant", value=0)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.pad(x, (self.kernel3 - 1, 0), mode="constant", value=0)
        x = self.conv3(x)
        x = self.activation(x)
        # (B, conv3_dim, T) -> (B, T, conv3_dim)
        x = x.transpose(1, 2)
        if squeeze:
            x = x.squeeze(0)
        return x