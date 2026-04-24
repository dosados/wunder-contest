from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from metrics import contest_metric, mae, mse
from utils import get_logger


@dataclass
class StackFeatureSpec:
    model_names: list[str]
    feature_order: list[str]
    target_names: list[str]
    warmup_steps: int
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_names": self.model_names,
            "feature_order": self.feature_order,
            "target_names": self.target_names,
            "warmup_steps": self.warmup_steps,
            "version": self.version,
        }


def _read_oof(path: str, selected_models: list[str] | None = None):
    tbl = pq.read_table(path)
    cols = tbl.column_names
    x_cols = [c for c in cols if c.endswith("_pred_t0") or c.endswith("_pred_t1")]
    if selected_models:
        allowed = set(selected_models)
        x_cols = [c for c in x_cols if c.rsplit("_pred_", 1)[0] in allowed]
        if not x_cols:
            raise ValueError("selected_models produced empty feature set")
    X = np.column_stack([tbl[c].to_numpy() for c in x_cols]).astype(np.float32)
    y = np.column_stack(
        [tbl["target_t0"].to_numpy(), tbl["target_t1"].to_numpy()]
    ).astype(np.float32)
    model_names = sorted({c.rsplit("_pred_", 1)[0] for c in x_cols})
    return (X, y, x_cols, model_names)


class MLPStack(nn.Module):

    def __init__(self, in_dim: int, hidden_layers: list[int]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_stack(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    logger = get_logger("stack_trainer")
    X, y, feature_order, model_names = _read_oof(
        config["oof_path"], config.get("selected_models")
    )
    feature_spec = StackFeatureSpec(
        model_names=model_names,
        feature_order=feature_order,
        target_names=["target_t0", "target_t1"],
        warmup_steps=int(config.get("warmup_steps", 99)),
    )
    mode = config.get("mode", "mlp")
    out_weights = Path(run_dir) / "weights"
    out_weights.mkdir(parents=True, exist_ok=True)
    history = {
        "train": {"contest_metric": [], "mse": [], "mae": []},
        "val": {"contest_metric": [], "mse": [], "mae": []},
    }
    if mode == "ridge":
        alpha = float(config.get("ridge_alpha", 1.0))
        model = Ridge(alpha=alpha, random_state=int(config.get("seed", 42)))
        model.fit(X, y)
        pred = model.predict(X)
        history["train"]["contest_metric"].append(contest_metric(y, pred))
        history["train"]["mse"].append(mse(y, pred))
        history["train"]["mae"].append(mae(y, pred))
        joblib.dump(model, out_weights / "ridge.joblib")
        return {
            "mode": "ridge",
            "history": history,
            "weights_path": str(out_weights / "ridge.joblib"),
            "feature_spec": feature_spec.to_dict(),
        }
    if mode == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError(
                "xgboost mode requires xgboost to be installed"
            ) from exc
        split = float(config.get("val_split", 0.2))
        n = len(X)
        n_val = max(1, int(n * split))
        X_train, X_val = (X[:-n_val], X[-n_val:])
        y_train, y_val = (y[:-n_val], y[-n_val:])
        params = {
            "n_estimators": int(config.get("xgb_n_estimators", 300)),
            "max_depth": int(config.get("xgb_max_depth", 6)),
            "learning_rate": float(config.get("xgb_learning_rate", 0.05)),
            "subsample": float(config.get("xgb_subsample", 0.9)),
            "colsample_bytree": float(config.get("xgb_colsample_bytree", 0.9)),
            "reg_alpha": float(config.get("xgb_reg_alpha", 0.0)),
            "reg_lambda": float(config.get("xgb_reg_lambda", 1.0)),
            "random_state": int(config.get("seed", 42)),
            "objective": "reg:squarederror",
            "n_jobs": int(config.get("xgb_n_jobs", -1)),
            "tree_method": str(config.get("xgb_tree_method", "auto")),
        }
        base_model = XGBRegressor(**params)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        history["train"]["contest_metric"].append(contest_metric(y_train, pred_train))
        history["train"]["mse"].append(mse(y_train, pred_train))
        history["train"]["mae"].append(mae(y_train, pred_train))
        history["val"]["contest_metric"].append(contest_metric(y_val, pred_val))
        history["val"]["mse"].append(mse(y_val, pred_val))
        history["val"]["mae"].append(mae(y_val, pred_val))
        logger.info(
            "XGBoost stack val: contest=%.6f mse=%.6f mae=%.6f",
            history["val"]["contest_metric"][-1],
            history["val"]["mse"][-1],
            history["val"]["mae"][-1],
        )
        joblib.dump(model, out_weights / "xgboost.joblib")
        return {
            "mode": "xgboost",
            "history": history,
            "weights_path": str(out_weights / "xgboost.joblib"),
            "feature_spec": feature_spec.to_dict(),
            "params": params,
        }
    device = config.get("device", "cpu")
    split = float(config.get("val_split", 0.2))
    n = len(X)
    n_val = max(1, int(n * split))
    X_train, X_val = (X[:-n_val], X[-n_val:])
    y_train, y_val = (y[:-n_val], y[-n_val:])
    model = MLPStack(X.shape[1], config.get("hidden_layers", [64]))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 0.001)))
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=int(config.get("batch_size", 1024)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=int(config.get("batch_size", 1024)),
        shuffle=False,
    )
    epochs = int(config.get("epochs", 20))
    best_metric = -1e18
    best_path = out_weights / "stack_mlp.pt"
    for ep in range(1, epochs + 1):
        model.train()
        p_train, t_train = ([], [])
        for xb, yb in tqdm(
            train_loader, desc=f"Stack train {ep}/{epochs}", leave=False
        ):
            xb, yb = (xb.to(device), yb.to(device))
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            p_train.append(pred.detach().cpu().numpy())
            t_train.append(yb.detach().cpu().numpy())
        tr_p = np.concatenate(p_train, axis=0)
        tr_t = np.concatenate(t_train, axis=0)
        history["train"]["contest_metric"].append(contest_metric(tr_t, tr_p))
        history["train"]["mse"].append(mse(tr_t, tr_p))
        history["train"]["mae"].append(mae(tr_t, tr_p))
        model.eval()
        p_val, t_val = ([], [])
        with torch.no_grad():
            for xb, yb in tqdm(
                val_loader, desc=f"Stack val {ep}/{epochs}", leave=False
            ):
                pred = model(xb.to(device)).cpu().numpy()
                p_val.append(pred)
                t_val.append(yb.numpy())
        va_p = np.concatenate(p_val, axis=0)
        va_t = np.concatenate(t_val, axis=0)
        v_metric = contest_metric(va_t, va_p)
        history["val"]["contest_metric"].append(v_metric)
        history["val"]["mse"].append(mse(va_t, va_p))
        history["val"]["mae"].append(mae(va_t, va_p))
        logger.info(
            "Epoch %d stack val: contest=%.6f mse=%.6f mae=%.6f",
            ep,
            history["val"]["contest_metric"][-1],
            history["val"]["mse"][-1],
            history["val"]["mae"][-1],
        )
        if v_metric > best_metric:
            best_metric = v_metric
            torch.save(model.state_dict(), best_path)
    return {
        "mode": "mlp",
        "history": history,
        "weights_path": str(best_path),
        "feature_spec": feature_spec.to_dict(),
    }
