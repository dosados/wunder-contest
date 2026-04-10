from __future__ import annotations
from typing import Any
from models import FullModel, GRUModel, SSMModel, WindowTransformerModel


def create_model(model_name: str, model_config: dict[str, Any] | None = None):
    model_name = model_name.lower()
    if model_name == "conv_lstm":
        return FullModel()
    if model_name == "gru":
        return GRUModel()
    if model_name == "ssm":
        return SSMModel(config=model_config)
    if model_name in ("transformer", "window_transformer"):
        return WindowTransformerModel(config=model_config)
    raise ValueError(f"Unknown model: {model_name}")
