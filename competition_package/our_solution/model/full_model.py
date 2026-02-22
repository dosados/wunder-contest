import torch
import torch.nn as nn

from model.conv_blocks import StreamingTemporalConvModel
from model.lstm_blocks import StreamingLSTM
from constants import convo_constants

current_constants = convo_constants


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = current_constants.get("input_dim")
        linear1_dim = current_constants.get("linear1_dim")
        lstm_hidden = current_constants.get("lstm_hidden")
        output_dim = current_constants.get("output_dim")

        self.input_dim = input_dim
        self.linear1_dim = linear1_dim
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, linear1_dim)
        self.conv_model = StreamingTemporalConvModel()
        self.lstm = StreamingLSTM()
        self.linear2 = nn.Linear(lstm_hidden, output_dim)

        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None
        self._last_seq_ix: int | None = None

    def reset_state(self):
        """Сброс состояния при смене последовательности (новый seq_ix)."""
        self.conv_model.reset_state()
        self._h = None
        self._c = None

    def forward(self, x: torch.Tensor, seq_ix: int | None = None) -> torch.Tensor:
        """
        Один или несколько шагов (streaming): обновляет внутреннее состояние.
        x: (D,) один шаг или (B, D) — B шагов подряд (состояние обновляется после каждого).
        seq_ix: при передаче — состояние сбрасывается при смене значения (новая последовательность).
        """
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix
        if x.dim() == 1:
            x = self.input_proj(x)
            x = self.conv_model(x)
            out, h_new, c_new = self.lstm(x, self._h, self._c)
            self._h = h_new.detach().clone()
            self._c = c_new.detach().clone()
            return self.linear2(out)
        # (B, D): несколько шагов подряд
        outs = []
        for i in range(x.shape[0]):
            xi = self.input_proj(x[i])
            xi = self.conv_model(xi)
            out, h_new, c_new = self.lstm(xi, self._h, self._c)
            self._h = h_new.detach().clone()
            self._c = c_new.detach().clone()
            outs.append(self.linear2(out))
        return torch.stack(outs, dim=0)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Целая последовательность (B, T, D) или (T, D) — для обучения, без сохранения состояния между вызовами."""
        x = self.input_proj(x)
        x = self.conv_model.forward_sequence(x)
        x = self.lstm.forward_sequence(x)
        return self.linear2(x)
