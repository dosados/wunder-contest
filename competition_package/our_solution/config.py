"""
Конфигурация модели и данных для предсказания состояния рынка.
Соответствует архитектуре из description.md: Hybrid Conv + Transformer.
"""

from dataclasses import dataclass
from typing import Optional

# Размерность данных из data_description.md и utils.py (колонки 3:35)
FEATURE_DIM = 32
SEQ_LENGTH = 1000
TARGET_DIM = 2


@dataclass
class ModelConfig:
    """Гиперпараметры модели."""
    # Вход
    input_dim: int = FEATURE_DIM
    seq_length: int = SEQ_LENGTH

    # Feature projection
    d_model: int = 128

    # Temporal convolution
    n_conv_blocks: int = 4
    conv_kernel: int = 3
    conv_dilations: tuple = (1, 2, 4, 8)  # по одному на блок

    # Transformer
    n_transformer_layers: int = 4
    n_heads: int = 4
    ff_dim: Optional[int] = None  # по умолчанию 4 * d_model

    # Pooling: "mean" | "attention"
    pooling: str = "mean"

    # Head
    head_hidden: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = 4 * self.d_model
        if len(self.conv_dilations) < self.n_conv_blocks:
            # повторить последнее dilation при нехватке
            last = self.conv_dilations[-1]
            self.conv_dilations = tuple(
                list(self.conv_dilations) + [last * 2 ** (i + 1) for i in range(self.n_conv_blocks - len(self.conv_dilations))]
            )[: self.n_conv_blocks]


# Конфиг по умолчанию для инференса/обучения
DEFAULT_CONFIG = ModelConfig()
