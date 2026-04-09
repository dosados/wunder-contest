from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt


def plot_history(history: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = ["contest_metric", "mse", "mae"]
    for m in metrics:
        tr = history.get("train", {}).get(m, [])
        va = history.get("val", {}).get(m, [])
        if not tr and (not va):
            continue
        plt.figure(figsize=(8, 4))
        if tr:
            plt.plot(tr, label="train")
        if va:
            plt.plot(va, label="val")
        plt.title(m)
        plt.xlabel("epoch")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"{m}.png")
        plt.close()
