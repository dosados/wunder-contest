from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from constants import (
    DEVICE,
    FEATURE_COLUMNS,
    SEQUENCE_BATCH_SIZE,
    TARGET_COLUMNS,
    TRAIN_PATH,
    WARMUP_STEPS,
)
from dataset import ParquetSequenceDataset
from models import create_model
from utils import get_logger


def _kfold_indices(n: int, folds: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = np.array_split(idx, folds)
    for i in range(folds):
        yield (
            np.concatenate([split[j] for j in range(folds) if j != i]).tolist(),
            split[i].tolist(),
        )


def build_oof_dataset(config: dict[str, Any], run_dir: str | Path) -> dict[str, str]:
    logger = get_logger("oof_builder")
    ds = ParquetSequenceDataset(
        TRAIN_PATH, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS
    )
    folds = int(config.get("folds", 5))
    seed = int(config.get("seed", 42))
    models_cfg = config["models"]
    out_dir = Path(run_dir) / "oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_paths = {}
    merged_features = []
    merged_target = None
    for mcfg in models_cfg:
        name = mcfg["name"]
        logger.info("Build OOF for model: %s", name)
        oof_pred = np.zeros((len(ds), ds.seq_len - WARMUP_STEPS, 2), dtype=np.float32)
        oof_target = np.zeros((len(ds), ds.seq_len - WARMUP_STEPS, 2), dtype=np.float32)
        for _, valid_idx in _kfold_indices(len(ds), folds, seed):
            model = create_model(name, mcfg.get("model_config"))
            weights_path = mcfg["weights_path"]
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            model = model.to(DEVICE).eval()
            loader = DataLoader(
                Subset(ds, valid_idx),
                batch_size=SEQUENCE_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            with torch.no_grad():
                for batch_i, (x, y) in enumerate(
                    tqdm(loader, desc=f"OOF {name}", leave=False)
                ):
                    x = x.to(DEVICE)
                    pred = model.forward_sequence(x).cpu().numpy()[:, WARMUP_STEPS:]
                    tgt = y.numpy()[:, WARMUP_STEPS:]
                    start = batch_i * SEQUENCE_BATCH_SIZE
                    for j in range(pred.shape[0]):
                        seq_idx = valid_idx[start + j]
                        oof_pred[seq_idx] = pred[j]
                        oof_target[seq_idx] = tgt[j]
        flat_pred = oof_pred.reshape(-1, 2)
        flat_tgt = oof_target.reshape(-1, 2)
        table = pa.table(
            {
                f"{name}_pred_t0": flat_pred[:, 0],
                f"{name}_pred_t1": flat_pred[:, 1],
                "target_t0": flat_tgt[:, 0],
                "target_t1": flat_tgt[:, 1],
            }
        )
        path = out_dir / f"oof_{name}.parquet"
        pq.write_table(table, path)
        per_model_paths[name] = str(path)
        merged_features.append(flat_pred)
        merged_target = flat_tgt
    merged = np.concatenate(merged_features, axis=1)
    merged_cols = {}
    for i, mcfg in enumerate(models_cfg):
        merged_cols[f"{mcfg['name']}_pred_t0"] = merged[:, i * 2 + 0]
        merged_cols[f"{mcfg['name']}_pred_t1"] = merged[:, i * 2 + 1]
    merged_cols["target_t0"] = merged_target[:, 0]
    merged_cols["target_t1"] = merged_target[:, 1]
    merged_path = out_dir / "oof_stack_train.parquet"
    pq.write_table(pa.table(merged_cols), merged_path)
    return {"merged": str(merged_path), "per_model": per_model_paths}
