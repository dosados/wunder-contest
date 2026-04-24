from __future__ import annotations
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs"

_DEFAULT_TRAIN_CONFIGS = {
    "conv_lstm": _CONFIGS_DIR / "train_conv_lstm.json",
    "gru": _CONFIGS_DIR / "train_gru.json",
    "ssm": _CONFIGS_DIR / "train_ssm.json",
    "transformer": _CONFIGS_DIR / "train_transformer.json",
    "window_transformer": _CONFIGS_DIR / "train_transformer.json",
}


def load_json_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    return "transformer" if normalized == "window_transformer" else normalized


def default_train_config_path(model_name: str) -> Path:
    key = normalize_model_name(model_name)
    if key not in _DEFAULT_TRAIN_CONFIGS:
        raise ValueError(f"Unsupported model name: {model_name}")
    return _DEFAULT_TRAIN_CONFIGS[key]


def optuna_best_config_path(model_name: str) -> Path:
    key = normalize_model_name(model_name)
    return _CONFIGS_DIR / f"optuna_best_{key}.json"


def resolve_train_config_path(
    *,
    model_name: str,
    explicit_path: str | Path | None = None,
    prefer_optuna_best: bool = True,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    if prefer_optuna_best:
        best_path = optuna_best_config_path(model_name)
        if best_path.is_file():
            return best_path
    return default_train_config_path(model_name)
