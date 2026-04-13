from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}

DEFAULT_MLP_BLOCK: dict[str, Any] = {
    "activation": "gelu",
    "use_batchnorm": False,
    "use_dropout": False,
    "dropout": 0.1,
    "skip_mode": "none",
}


def get_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise KeyError(f"Unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


def merge_block_cfg(
    stack: dict[str, Any] | None, global_block: dict[str, Any] | None = None
) -> dict[str, Any]:
    out = dict(DEFAULT_MLP_BLOCK)
    if global_block:
        out.update(global_block)
    if stack:
        for k, v in stack.items():
            if k != "hidden_dims":
                out[k] = v
    return out


class ConfigurableLinearBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, cfg: dict[str, Any]):
        super().__init__()
        self.skip_mode = cfg.get("skip_mode", "none")
        if self.skip_mode not in ("none", "sum", "concat"):
            raise ValueError(f"skip_mode must be none|sum|concat, got {self.skip_mode!r}")
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = (
            nn.BatchNorm1d(out_dim) if cfg.get("use_batchnorm") else None
        )
        self.act = get_activation(cfg.get("activation", "gelu"))
        self.dropout = (
            nn.Dropout(float(cfg.get("dropout", 0.1)))
            if cfg.get("use_dropout")
            else None
        )
        if self.skip_mode == "sum":
            self.skip_proj = (
                nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            )
            self.merge = None
        elif self.skip_mode == "concat":
            self.skip_proj = nn.Linear(in_dim, out_dim)
            self.merge = nn.Linear(out_dim * 2, out_dim)
        else:
            self.skip_proj = None
            self.merge = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *leading, d_in = x.shape
        flat = x.reshape(-1, d_in)
        z = self.linear(flat)
        if self.bn is not None:
            z = self.bn(z)
        z = self.act(z)
        if self.dropout is not None:
            z = self.dropout(z)
        d_out = z.shape[-1]
        z = z.view(*leading, d_out)

        if self.skip_mode == "none":
            return z

        x_flat = x.reshape(-1, d_in)
        if self.skip_mode == "sum":
            skip = self.skip_proj(x_flat).view(*leading, d_out)
            return z + skip

        skip = self.skip_proj(x_flat).view(*leading, d_out)
        merged = torch.cat([z, skip], dim=-1)
        mf = merged.reshape(-1, d_out * 2)
        out = self.merge(mf).view(*leading, d_out)
        return out


def build_mlp_stack(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int] | None,
    stack_cfg: dict[str, Any] | None,
    global_block: dict[str, Any] | None = None,
) -> nn.Module:
    hidden_dims = list(hidden_dims or [])
    block_cfg = merge_block_cfg(stack_cfg, global_block)
    dims = [in_dim] + hidden_dims + [out_dim]
    if len(dims) < 2:
        raise ValueError("invalid dims")
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(ConfigurableLinearBlock(dims[i], dims[i + 1], block_cfg))
    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)


def build_input_projection(
    in_dim: int,
    out_dim: int,
    input_stack: dict[str, Any] | None,
    global_block: dict[str, Any] | None = None,
) -> nn.Module:
    if input_stack is None:
        if in_dim == out_dim:
            return nn.Identity()
        return nn.Linear(in_dim, out_dim)
    return build_mlp_stack(
        in_dim,
        out_dim,
        input_stack.get("hidden_dims"),
        input_stack,
        global_block,
    )


def transformer_activation_name(cfg_activation: str) -> str:
    a = cfg_activation.lower()
    if a in ("relu", "gelu"):
        return a
    if a == "silu":
        return "gelu"
    return "gelu"
