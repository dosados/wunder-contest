from __future__ import annotations
from typing import Any, Callable
import optuna
from constants import (
    convo_constants,
    gru_constants,
    ssm_constants,
    transformer_constants,
)


def _block_from_trial(trial: optuna.Trial, prefix: str = "") -> dict[str, Any]:
    pfx = f"{prefix}_" if prefix else ""
    return {
        "activation": trial.suggest_categorical(f"{pfx}act", ["relu", "gelu", "tanh"]),
        "use_batchnorm": trial.suggest_categorical(f"{pfx}bn", [True, False]),
        "use_dropout": trial.suggest_categorical(f"{pfx}do", [True, False]),
        "dropout": trial.suggest_float(f"{pfx}dropout", 0.0, 0.25),
        "skip_mode": trial.suggest_categorical(f"{pfx}skip", ["none", "sum", "concat"]),
    }


def sample_gru_params(trial: optuna.Trial) -> dict[str, Any]:
    p = dict(gru_constants)
    p["gru_hidden"] = trial.suggest_int("gru_hidden", 128, 384, step=64)
    p["gru_num_layers"] = trial.suggest_int("gru_num_layers", 1, 3)
    p["linear_dim"] = trial.suggest_int("linear_dim", 32, 128, step=32)
    p["block"] = _block_from_trial(trial, "blk")
    post = trial.suggest_categorical("gru_post", ["none", "one", "two"])
    if post == "none":
        p.pop("post_stack", None)
    elif post == "one":
        p["post_stack"] = {
            "hidden_dims": [trial.suggest_int("gru_post_h", 64, 256, step=64)],
            **_block_from_trial(trial, "post"),
        }
    else:
        h1 = trial.suggest_int("gru_post_h1", 128, 320, step=64)
        h2 = trial.suggest_int("gru_post_h2", 32, min(h1, 192), step=32)
        p["post_stack"] = {
            "hidden_dims": [h1, h2],
            **_block_from_trial(trial, "post"),
        }
    use_in = trial.suggest_categorical("gru_input_stack", [False, True])
    if use_in:
        p["input_stack"] = {
            "hidden_dims": [trial.suggest_int("gru_in_h", 32, 128, step=32)],
            **_block_from_trial(trial, "in"),
        }
    else:
        p.pop("input_stack", None)
    return p


def sample_conv_lstm_params(trial: optuna.Trial) -> dict[str, Any]:
    p = dict(convo_constants)
    p["linear1_dim"] = trial.suggest_int("cl_linear1", 32, 128, step=32)
    p["conv_dim"] = trial.suggest_int("cl_conv_dim", 64, 256, step=64)
    p["lstm_hidden"] = trial.suggest_int("cl_lstm_h", 128, 384, step=64)
    p["conv_window"] = trial.suggest_int("cl_kw", 3, 9, step=2)
    p["lstm_num_layers"] = trial.suggest_int("cl_lstm_layers", 1, 3)
    p["conv_activation"] = trial.suggest_categorical("cl_conv_act", ["gelu", "relu"])
    p["block"] = _block_from_trial(trial, "cl_blk")
    if trial.suggest_categorical("cl_input_stack", [False, True]):
        p["input_stack"] = {
            "hidden_dims": [trial.suggest_int("cl_in_h", 32, 96, step=32)],
            **_block_from_trial(trial, "cl_in"),
        }
    else:
        p.pop("input_stack", None)
    pre = trial.suggest_categorical("cl_pre_lstm", ["none", "one"])
    if pre == "one":
        p["pre_lstm_stack"] = {
            "hidden_dims": [trial.suggest_int("cl_pre_h", 64, min(p["conv_dim"], 256), step=32)],
            **_block_from_trial(trial, "cl_pre"),
        }
    else:
        p.pop("pre_lstm_stack", None)
    post = trial.suggest_categorical("cl_post_lstm", ["none", "one"])
    if post == "one":
        p["post_lstm_stack"] = {
            "hidden_dims": [trial.suggest_int("cl_post_h", 64, min(p["lstm_hidden"], 256), step=32)],
            **_block_from_trial(trial, "cl_post"),
        }
    else:
        p.pop("post_lstm_stack", None)
    return p


def sample_ssm_params(trial: optuna.Trial) -> dict[str, Any]:
    p = dict(ssm_constants)
    p["d_model"] = trial.suggest_categorical("ssm_d_model", [32, 48, 64])
    p["n_layers"] = trial.suggest_int("ssm_n_layers", 1, 4)
    p["d_state"] = trial.suggest_categorical("ssm_d_state", [8, 16, 24])
    p["d_conv"] = trial.suggest_categorical("ssm_d_conv", [2, 4, 8])
    p["expand"] = trial.suggest_categorical("ssm_expand", [1, 2])
    p["block"] = _block_from_trial(trial, "ssm_blk")
    if trial.suggest_categorical("ssm_input_stack", [False, True]):
        hid = trial.suggest_int("ssm_in_h", 32, max(p["d_model"], 32), step=16)
        p["input_stack"] = {
            "hidden_dims": [hid],
            **_block_from_trial(trial, "ssm_in"),
        }
    else:
        p.pop("input_stack", None)
    if trial.suggest_categorical("ssm_post_stack", [False, True]):
        ph = trial.suggest_int("ssm_post_h", 32, max(p["d_model"], 64), step=16)
        p["post_stack"] = {
            "hidden_dims": [ph],
            **_block_from_trial(trial, "ssm_post"),
        }
    else:
        p.pop("post_stack", None)
    return p


def sample_transformer_params(trial: optuna.Trial) -> dict[str, Any]:
    p = dict(transformer_constants)
    p["d_model"] = trial.suggest_categorical("tf_d", [48, 64, 96])
    p["n_heads"] = trial.suggest_categorical("tf_heads", [2, 4])
    p["n_layers"] = trial.suggest_int("tf_layers", 1, 4)
    p["context_window"] = trial.suggest_categorical("tf_win", [32, 64, 96])
    p["dim_feedforward"] = trial.suggest_categorical("tf_ff", [128, 256, 384])
    p["encoder_dropout"] = trial.suggest_float("tf_enc_do", 0.0, 0.2)
    p["encoder_activation"] = trial.suggest_categorical("tf_enc_act", ["gelu", "relu"])
    p["encoder_norm_first"] = trial.suggest_categorical("tf_norm_first", [True, False])
    if p["d_model"] % p["n_heads"] != 0:
        p["d_model"] = ((p["d_model"] // p["n_heads"]) + 1) * p["n_heads"]
    p["block"] = _block_from_trial(trial, "tf_blk")
    if trial.suggest_categorical("tf_input_stack", [False, True]):
        p["input_stack"] = {
            "hidden_dims": [trial.suggest_int("tf_in_h", 32, p["d_model"], step=16)],
            **_block_from_trial(trial, "tf_in"),
        }
    else:
        p.pop("input_stack", None)
    if trial.suggest_categorical("tf_post_stack", [False, True]):
        p["post_stack"] = {
            "hidden_dims": [trial.suggest_int("tf_post_h", 32, p["d_model"], step=16)],
            **_block_from_trial(trial, "tf_post"),
        }
    else:
        p.pop("post_stack", None)
    return p


MODEL_PARAM_SAMPLERS: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "gru": sample_gru_params,
    "conv_lstm": sample_conv_lstm_params,
    "ssm": sample_ssm_params,
}


def normalize_model_name(name: str) -> str:
    n = name.lower().strip()
    if n == "window_transformer":
        return "transformer"
    return n


def sample_model_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    key = normalize_model_name(model_name)
    if key == "transformer":
        return sample_transformer_params(trial)
    fn = MODEL_PARAM_SAMPLERS.get(key)
    if fn is None:
        raise ValueError(f"No Optuna search space for model {model_name!r}")
    return fn(trial)
