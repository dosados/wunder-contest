import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import DEVICE, convo_constants
from models.model_state import ModelState


class StreamingTemporalConvModel(nn.Module):

    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation or nn.GELU()
        linear1_dim = convo_constants.get("linear1_dim")
        conv_dim = convo_constants.get("conv_dim")
        self.kernel = convo_constants.get("conv_window")
        self.conv = nn.Conv1d(linear1_dim, conv_dim, self.kernel, bias=True)
        self.state = ModelState(self.kernel, linear1_dim, DEVICE)

    def _forward_conv_step(self, seq: torch.Tensor):
        if seq.shape[0] < self.kernel:
            pad = torch.zeros(
                self.kernel - seq.shape[0],
                seq.shape[1],
                device=seq.device,
                dtype=seq.dtype,
            )
            seq = torch.cat([pad, seq], dim=0)
        else:
            seq = seq[-self.kernel :]
        return self.conv(seq.transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(-1)

    def forward(self, x_t):
        self.state.add_conv(x_t)
        return self.activation(self._forward_conv_step(self.state.get_conv_sequence()))

    def forward_sequence(self, x):
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel - 1, 0), mode="constant", value=0)
        x = self.activation(self.conv(x)).transpose(1, 2)
        return x.squeeze(0) if squeeze else x

    def reset_state(self):
        self.state.reset()
