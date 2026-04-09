from __future__ import annotations
import numpy as np


def weighted_pearson(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-08
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_pred = np.clip(y_pred, -6.0, 6.0)
    w = np.clip(np.abs(y_true), eps, None)
    sw = w.sum()
    if sw <= 0:
        return 0.0
    mt = (y_true * w).sum() / sw
    mp = (y_pred * w).sum() / sw
    dt = y_true - mt
    dp = y_pred - mp
    cov = (w * dt * dp).sum() / sw
    vt = (w * dt * dt).sum() / sw
    vp = (w * dp * dp).sum() / sw
    den = max((vt * vp) ** 0.5, eps)
    return float(max(min(cov / den, 1.0), -1.0))


def contest_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    c0 = weighted_pearson(y_true[:, 0], y_pred[:, 0])
    c1 = weighted_pearson(y_true[:, 1], y_pred[:, 1])
    return (c0 + c1) * 0.5


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(e * e))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.abs(e)))
