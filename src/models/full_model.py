import torch
import torch.nn as nn
from constants import convo_constants
from models.conv_blocks import StreamingTemporalConvModel
from models.lstm_blocks import StreamingLSTM


class FullModel(nn.Module):

    def __init__(self):
        super().__init__()
        cfg = convo_constants
        self.input_dim = cfg["input_dim"]
        self.linear1_dim = cfg["linear1_dim"]
        self.lstm_hidden = cfg["lstm_hidden"]
        self.output_dim = cfg["output_dim"]
        self.input_proj = nn.Linear(self.input_dim, self.linear1_dim)
        self.conv_model = StreamingTemporalConvModel()
        self.lstm0 = StreamingLSTM()
        self.lstm1 = StreamingLSTM()
        self.linear0 = nn.Linear(self.lstm_hidden, 1)
        self.linear1 = nn.Linear(self.lstm_hidden, 1)
        self.residual_proj = nn.Linear(self.input_dim, self.output_dim)
        self._h0 = self._c0 = self._h1 = self._c1 = None
        self._last_seq_ix = None

    def reset_state(self):
        self.conv_model.reset_state()
        self._h0 = self._c0 = self._h1 = self._c1 = None

    def forward(self, x: torch.Tensor, seq_ix: int | None = None):
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix
        residual = self.residual_proj(x)
        if x.dim() == 1:
            x_conv = self.conv_model(self.input_proj(x))
            o0, self._h0, self._c0 = self.lstm0(x_conv, self._h0, self._c0)
            o1, self._h1, self._c1 = self.lstm1(x_conv, self._h1, self._c1)
            return torch.cat([self.linear0(o0), self.linear1(o1)], dim=-1) + residual
        outs = []
        for i in range(x.shape[0]):
            xi = x[i]
            ri = residual[i]
            x_conv = self.conv_model(self.input_proj(xi))
            o0, self._h0, self._c0 = self.lstm0(x_conv, self._h0, self._c0)
            o1, self._h1, self._c1 = self.lstm1(x_conv, self._h1, self._c1)
            outs.append(torch.cat([self.linear0(o0), self.linear1(o1)], dim=-1) + ri)
        return torch.stack(outs, dim=0)

    def forward_sequence(self, x: torch.Tensor):
        residual = self.residual_proj(x)
        x = self.conv_model.forward_sequence(self.input_proj(x))
        out0 = self.linear0(self.lstm0.forward_sequence(x))
        out1 = self.linear1(self.lstm1.forward_sequence(x))
        return torch.cat([out0, out1], dim=-1) + residual
