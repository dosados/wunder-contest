import torch
import torch.nn as nn

from constants import convo_constants

current_constants = convo_constants


class StreamingLSTM(nn.Module):
    """
    LSTM с поддержкой двух режимов:
    - forward(x_t, h, c): один шаг, обновляет и возвращает (h, c) для следующего вызова.
    - forward_sequence(x): батч (B, T, C) — один проход без сохранения состояния между вызовами (для обучения).
    """

    def __init__(self):
        super().__init__()
        input_size = current_constants.get("conv_dim")
        hidden_size = current_constants.get("lstm_hidden")
        num_layers = current_constants.get("lstm_num_layers", 1)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
        )

    def _get_zero_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        h = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=dtype
        )
        c = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=dtype
        )
        return h, c

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Один шаг (streaming). x_t: (conv_dim,) или (batch, conv_dim).
        h, c: (num_layers, batch, hidden) или None (тогда нулевое состояние).
        Возвращает (out, h_new, c_new), out: (hidden,) или (batch, hidden).
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0).unsqueeze(0)  # (1, 1, conv_dim)
            batch_size = 1
            squeeze_out = True
        else:
            batch_size = x_t.shape[0]
            x_t = x_t.unsqueeze(1)  # (batch, 1, conv_dim)
            squeeze_out = False

        device = x_t.device
        dtype = x_t.dtype
        if h is None or c is None:
            h, c = self._get_zero_state(batch_size, device, dtype)

        out, (h_new, c_new) = self.lstm(x_t, (h, c))
        if squeeze_out:
            out = out.squeeze(0).squeeze(0)  # (hidden,)
        else:
            out = out.squeeze(1)  # (batch, hidden)
        return out, h_new, c_new

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Один проход по всей последовательности (для обучения).
        x: (T, conv_dim) или (B, T, conv_dim).
        Возвращает (T, hidden) или (B, T, hidden). Состояние не сохраняется между вызовами.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        out, _ = self.lstm(x)
        if squeeze:
            out = out.squeeze(0)
        return out
