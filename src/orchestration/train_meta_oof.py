from __future__ import annotations
import argparse
import os
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


def main():
    parser = argparse.ArgumentParser(description="Train meta-head on OOF predictions")
    parser.add_argument("--variant", choices=["lstm_gru", "lstm_ssm"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ds = ParquetSequenceDataset(
        TRAIN_PATH, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS
    )
    pair = build_base_pair(args.variant)
    pair.set_backbones_trainable(False)
    oof_x, oof_y = ([], [])
    for _, valid_idx in _kfold_indices(len(ds), args.folds, args.seed):
        xv, yv = _collect_fold_predictions(pair, ds, valid_idx)
        oof_x.append(xv)
        oof_y.append(yv)
    X = torch.from_numpy(np.concatenate(oof_x, axis=0)).float().to(DEVICE)
    Y = torch.from_numpy(np.concatenate(oof_y, axis=0)).float().to(DEVICE)
    meta = _build_meta_head().to(DEVICE)
    opt = torch.optim.Adam(meta.parameters(), lr=LR_STACK)
    loss_fn = nn.MSELoss()
    for _ in range(args.epochs):
        pred = meta(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = meta(X).clamp(PRED_CLIP_LOW, PRED_CLIP_HIGH).cpu().numpy()
        t = Y.cpu().numpy()
    print(
        f"OOF meta score ({args.variant}): {(weighted_pearson(t[:, 0], p[:, 0]) + weighted_pearson(t[:, 1], p[:, 1])) * 0.5:.6f}"
    )
    out_dir = os.path.join(STACK_DIR, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(meta.state_dict(), os.path.join(out_dir, "meta_head_oof.pt"))


if __name__ == "__main__":
    main()
