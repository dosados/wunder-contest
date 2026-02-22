import torch
import torch.nn as nn

from model.conv_blocks import StreamingTemporalConvModel
from constants import convo_constants

current_constants = convo_constants


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = current_constants.get("input_dim")
        linear1_dim = current_constants.get("linear1_dim")
        conv3_dim = current_constants.get("conv3_dim")
        output_dim = current_constants.get("output_dim")

        self.input_dim = input_dim
        self.linear1_dim = linear1_dim
        self.conv3_dim = conv3_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, linear1_dim)
        self.conv_model = StreamingTemporalConvModel()
        self.linear2 = nn.Linear(conv3_dim, output_dim)


    def forward(self, x):
        """Один шаг (streaming): x (D,) или (B, D). Для инференса."""
        x = self.input_proj(x)
        x = self.conv_model(x)
        x = self.linear2(x)
        return x

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Целая последовательность: x (T, D) или (B, T, D). Для обучения."""
        x = self.input_proj(x)
        x = self.conv_model.forward_sequence(x)
        x = self.linear2(x)
        return x