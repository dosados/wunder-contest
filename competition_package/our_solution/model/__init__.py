"""
Hybrid Conv + Transformer модель для предсказания состояния рынка.
Архитектура: description.md.
"""

from .full_model import FullModel

try:
    from .gru_model import GRUModel
except ModuleNotFoundError:
    GRUModel = None

__all__ = ["FullModel", "GRUModel"]
