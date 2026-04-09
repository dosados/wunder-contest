from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from metrics import contest_metric, mae, mse
from utils import get_logger


@dataclass
class TrainResult:
    best_epoch: int
    best_metric: float
    history: dict
    best_weights_path: str


def _collect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "contest_metric": contest_metric(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: str,
    epochs: int,
    lr: float,
    run_dir: str | Path,
    warmup_steps: int = 99,
    loss_name: str = "mse",
) -> TrainResult:
    logger = get_logger("training")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss() if loss_name == "mse" else torch.nn.L1Loss()
    history = {
        "train": {"contest_metric": [], "mse": [], "mae": []},
        "val": {"contest_metric": [], "mse": [], "mae": []},
    }
    best_metric = -1e18
    best_epoch = -1
    best_path = str(Path(run_dir) / "weights" / "best_model.pt")
    logger.info("Start training for %d epochs", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        tr_y, tr_p = ([], [])
        tr_bar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for x, y in tr_bar:
            x, y = (x.to(device), y.to(device))
            pred = model.forward_sequence(x)
            loss = criterion(pred[:, warmup_steps:], y[:, warmup_steps:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_y.append(y[:, warmup_steps:].reshape(-1, 2).detach().cpu().numpy())
            tr_p.append(pred[:, warmup_steps:].reshape(-1, 2).detach().cpu().numpy())
            tr_bar.set_postfix(loss=f"{loss.item():.5f}")
        tr_m = _collect_metrics(
            np.concatenate(tr_y, axis=0), np.concatenate(tr_p, axis=0)
        )
        for k, v in tr_m.items():
            history["train"][k].append(v)
        logger.info(
            "Epoch %d train: contest=%.6f mse=%.6f mae=%.6f",
            epoch,
            tr_m["contest_metric"],
            tr_m["mse"],
            tr_m["mae"],
        )
        if val_loader is not None:
            model.eval()
            va_y, va_p = ([], [])
            va_bar = tqdm(val_loader, desc=f"Val {epoch}/{epochs}", leave=False)
            with torch.no_grad():
                for x, y in va_bar:
                    x, y = (x.to(device), y.to(device))
                    pred = model.forward_sequence(x)
                    va_y.append(y[:, warmup_steps:].reshape(-1, 2).cpu().numpy())
                    va_p.append(pred[:, warmup_steps:].reshape(-1, 2).cpu().numpy())
            va_m = _collect_metrics(
                np.concatenate(va_y, axis=0), np.concatenate(va_p, axis=0)
            )
            for k, v in va_m.items():
                history["val"][k].append(v)
            logger.info(
                "Epoch %d val: contest=%.6f mse=%.6f mae=%.6f",
                epoch,
                va_m["contest_metric"],
                va_m["mse"],
                va_m["mae"],
            )
            metric = va_m["contest_metric"]
        else:
            metric = tr_m["contest_metric"]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved: epoch=%d metric=%.6f", epoch, metric)
    return TrainResult(
        best_epoch=best_epoch,
        best_metric=best_metric,
        history=history,
        best_weights_path=best_path,
    )
