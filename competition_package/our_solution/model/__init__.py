"""
Hybrid Conv + Transformer модель для предсказания состояния рынка.
Архитектура: description.md.
"""

from .hybrid_model import HybridConvTransformer
from ..config import ModelConfig

__all__ = ["HybridConvTransformer", "ModelConfig"]
