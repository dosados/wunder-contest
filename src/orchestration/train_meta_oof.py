from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from constants import (
    DEVICE,
    FEATURE_COLUMNS,
    LR_STACK,
    PRED_CLIP_HIGH,
    PRED_CLIP_LOW,
    SEQUENCE_BATCH_SIZE,
    STACK_DIR,
    TARGET_COLUMNS,
    TRAIN_PATH,
    WARMUP_STEPS,
)
from dataset import ParquetSequenceDataset
from orchestration.models import PairOrchestrator, build_base_pair
from metrics import weighted_pearson


def _build_meta_head(
    input_dim: int = 4, output_dim: int = 2, hidden: int = 64
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, output_dim),
    )


def _collect_fold_predictions(
    pair: PairOrchestrator, dataset: ParquetSequenceDataset, indices: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    x_meta, y_meta = ([], [])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pa = pair.model_a.forward_sequence(x)
            pb = pair.model_b.forward_sequence(x)
            x_meta.append(
                torch.cat([pa, pb], dim=-1)[:, WARMUP_STEPS:]
                .reshape(-1, 4)
                .cpu()
                .numpy()
            )
            y_meta.append(y[:, WARMUP_STEPS:].reshape(-1, 2).cpu().numpy())
    return (np.concatenate(x_meta, axis=0), np.concatenate(y_meta, axis=0))


def _kfold_indices(n: int, folds: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = np.array_split(idx, folds)
    return [
        (
            np.concatenate([split[j] for j in range(folds) if j != i]).tolist(),
            split[i].tolist(),
        )
        for i in range(folds)
    ]


def train_meta_oof(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    variant = config["variant"]
    folds = int(config.get("folds", 5))
    epochs = int(config.get("epochs", 10))
    seed = int(config.get("seed", 42))
    hidden = int(config.get("hidden", 64))
    ds = ParquetSequenceDataset(
        TRAIN_PATH, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS
    )
    pair = build_base_pair(variant)
    pair.set_backbones_trainable(False)
    oof_x, oof_y = ([], [])
    for _, valid_idx in _kfold_indices(len(ds), folds, seed):
        xv, yv = _collect_fold_predictions(pair, ds, valid_idx)
        oof_x.append(xv)
        oof_y.append(yv)
    X = torch.from_numpy(np.concatenate(oof_x, axis=0)).float().to(DEVICE)
    Y = torch.from_numpy(np.concatenate(oof_y, axis=0)).float().to(DEVICE)
    meta = _build_meta_head(hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(meta.parameters(), lr=LR_STACK)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        pred = meta(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = meta(X).clamp(PRED_CLIP_LOW, PRED_CLIP_HIGH).cpu().numpy()
        t = Y.cpu().numpy()
    score = (weighted_pearson(t[:, 0], p[:, 0]) + weighted_pearson(t[:, 1], p[:, 1])) * 0.5
    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "meta_head_oof.pt"
    torch.save(meta.state_dict(), weights_path)
    # Keep historical location for inference compatibility.
    legacy_path = Path(STACK_DIR) / variant / "meta_head_oof.pt"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta.state_dict(), legacy_path)
    return {
        "variant": variant,
        "score": float(score),
        "weights_path": str(weights_path),
        "legacy_weights_path": str(legacy_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train meta-head on OOF predictions")
    parser.add_argument("--variant", choices=["lstm_gru", "lstm_ssm"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path(STACK_DIR) / args.variant
    result = train_meta_oof(
        {
            "variant": args.variant,
            "folds": args.folds,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        run_dir=run_dir,
    )
    print(f"OOF meta score ({args.variant}): {result['score']:.6f}")


if __name__ == "__main__":
    main()
