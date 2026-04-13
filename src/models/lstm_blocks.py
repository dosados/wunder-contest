import torch
import torch.nn as nn
from constants import convo_constants


class StreamingLSTM(nn.Module):

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = dict(convo_constants) if cfg is None else {**convo_constants, **cfg}
        self.hidden_size = cfg["lstm_hidden"]
        self.num_layers = cfg.get("lstm_num_layers", 1)
        input_size = cfg.get("lstm_input_size", cfg["conv_dim"])
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
        )

    def _get_zero_state(self, batch_size, device, dtype):
        h = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype
        )
        c = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype
        )
        return (h, c)

    def forward(self, x_t, h=None, c=None):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0).unsqueeze(0)
            batch_size, squeeze = (1, True)
        else:
            x_t = x_t.unsqueeze(1)
            batch_size, squeeze = (x_t.shape[0], False)
        if h is None or c is None:
            h, c = self._get_zero_state(batch_size, x_t.device, x_t.dtype)
        out, (h_new, c_new) = self.lstm(x_t, (h, c))
        out = out.squeeze(1)
        if squeeze:
            out = out.squeeze(0)
        return (out, h_new, c_new)

    def forward_sequence(self, x):
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        out, _ = self.lstm(x)
        return out.squeeze(0) if squeeze else out
