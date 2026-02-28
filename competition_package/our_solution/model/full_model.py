import torch
import torch.nn as nn

from model.conv_blocks import StreamingTemporalConvModel
from model.lstm_blocks import StreamingLSTM
from constants import convo_constants

current_constants = convo_constants


class FullModel(nn.Module):
    """
    Conv + два LSTM (один на таргет t0, один на t1) + residual skip от входа к выходу.
    output = residual_proj(x_raw) + [lstm0->linear0, lstm1->linear1].
    """

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
        self.lstm0 = StreamingLSTM()
        self.lstm1 = StreamingLSTM()
        self.linear0 = nn.Linear(lstm_hidden, 1)
        self.linear1 = nn.Linear(lstm_hidden, 1)
        self.residual_proj = nn.Linear(input_dim, output_dim)

        self._h0: torch.Tensor | None = None
        self._c0: torch.Tensor | None = None
        self._h1: torch.Tensor | None = None
        self._c1: torch.Tensor | None = None
        self._last_seq_ix: int | None = None

    def reset_state(self):
        """Сброс состояния при смене последовательности (новый seq_ix)."""
        self.conv_model.reset_state()
        self._h0 = None
        self._c0 = None
        self._h1 = None
        self._c1 = None

    def forward(self, x: torch.Tensor, seq_ix: int | None = None) -> torch.Tensor:
        """
        Один или несколько шагов (streaming): обновляет внутреннее состояние.
        x: (D,) один шаг или (B, D) — B шагов подряд (состояние обновляется после каждого).
        seq_ix: при передаче — состояние сбрасывается при смене значения (новая последовательность).
        """
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix

        residual = self.residual_proj(x)  # (D,) -> (2,) или (B, D) -> (B, 2)

        if x.dim() == 1:
            x_proj = self.input_proj(x)
            x_conv = self.conv_model(x_proj)
            out0, self._h0, self._c0 = self.lstm0(x_conv, self._h0, self._c0)
            out1, self._h1, self._c1 = self.lstm1(x_conv, self._h1, self._c1)
            head = torch.cat([self.linear0(out0), self.linear1(out1)], dim=-1)
            return head + residual
        # (B, D): несколько шагов подряд
        outs = []
        for i in range(x.shape[0]):
            xi = x[i]
            ri = residual[i] if residual.dim() > 1 else residual
            xi_proj = self.input_proj(xi)
            xi_conv = self.conv_model(xi_proj)
            out0, self._h0, self._c0 = self.lstm0(xi_conv, self._h0, self._c0)
            out1, self._h1, self._c1 = self.lstm1(xi_conv, self._h1, self._c1)
            head = torch.cat([self.linear0(out0), self.linear1(out1)], dim=-1)
            outs.append(head + ri)
        return torch.stack(outs, dim=0)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Целая последовательность (B, T, D) или (T, D) — для обучения, без сохранения состояния между вызовами."""
        residual = self.residual_proj(x)  # (B, T, 2) или (T, 2)
        x = self.input_proj(x)
        x = self.conv_model.forward_sequence(x)
        out0 = self.linear0(self.lstm0.forward_sequence(x))   # (B, T, 1) или (T, 1)
        out1 = self.linear1(self.lstm1.forward_sequence(x))
        head = torch.cat([out0, out1], dim=-1)
        return head + residual
