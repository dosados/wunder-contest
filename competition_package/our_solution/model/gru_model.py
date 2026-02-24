"""
Модель на основе GRU (PyTorch) для обучения на последовательностях.
Поддерживает forward_sequence для обучения и (опционально) streaming forward для инференса.
"""
import torch
import torch.nn as nn

from constants import gru_constants

current_constants = gru_constants


class GRUModel(nn.Module):
    """
    Архитектура: input_proj (опционально) -> GRU -> linear -> output.
    Обучение: forward_sequence(x) с x (B, T, D).
    Инференс: forward(x, seq_ix) по одному шагу с сохранением состояния.
    """

    def __init__(self):
        super().__init__()
        input_dim = current_constants.get("input_dim")
        linear_dim = current_constants.get("linear_dim", input_dim)
        gru_hidden = current_constants.get("gru_hidden")
        gru_num_layers = current_constants.get("gru_num_layers", 1)
        output_dim = current_constants.get("output_dim")

        self.input_dim = input_dim
        self.linear_dim = linear_dim
        self.gru_hidden = gru_hidden
        self.gru_num_layers = gru_num_layers
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, linear_dim) if linear_dim != input_dim else nn.Identity()
        gru_input_size = linear_dim if linear_dim != input_dim else input_dim
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_num_layers,
            batch_first=True,
            bias=True,
        )
        self.linear_out = nn.Linear(gru_hidden, output_dim)

        self._h: torch.Tensor | None = None
        self._last_seq_ix: int | None = None

    def reset_state(self):
        """Сброс состояния при смене последовательности (новый seq_ix)."""
        self._h = None

    def forward(self, x: torch.Tensor, seq_ix: int | None = None) -> torch.Tensor:
        """
        Один или несколько шагов (streaming). Обновляет внутреннее состояние.
        x: (D,) один шаг или (B, D) — B шагов подряд.
        seq_ix: при смене значения состояние сбрасывается (новая последовательность).
        """
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix
        if x.dim() == 1:
            x = self.input_proj(x)
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, linear_dim)
            batch_size = 1
            squeeze_out = True
        else:
            x = self.input_proj(x)
            x = x.unsqueeze(1)  # (B, 1, linear_dim)
            batch_size = x.shape[0]
            squeeze_out = False

        device = x.device
        dtype = x.dtype
        if self._h is None:
            self._h = torch.zeros(
                self.gru_num_layers, batch_size, self.gru_hidden,
                device=device, dtype=dtype,
            )

        out, h_new = self.gru(x, self._h)
        self._h = h_new.detach().clone()

        out = self.linear_out(out)
        if squeeze_out:
            out = out.squeeze(0).squeeze(0)  # (output_dim,)
        else:
            out = out.squeeze(1)  # (B, output_dim)
        return out

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Целая последовательность для обучения. Состояние не сохраняется между вызовами.
        x: (B, T, D) или (T, D). Возвращает (B, T, output_dim) или (T, output_dim).
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        x = self.input_proj(x)
        out, _ = self.gru(x)
        out = self.linear_out(out)
        if squeeze:
            out = out.squeeze(0)
        return out
