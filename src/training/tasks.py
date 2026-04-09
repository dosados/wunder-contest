from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from constants import (
    ARTIFACTS_ROOT,
    DEVICE,
    FEATURE_COLUMNS,
    SEQUENCE_BATCH_SIZE,
    TARGET_COLUMNS,
    TRAIN_PATH,
    VAL_PATH,
    WARMUP_STEPS,
)
from dataset import ParquetSequenceDataset
from models import create_model
from training import train_model
from utils import (
    get_logger,
    make_run_dir,
    plot_history,
    save_json,
    snapshot_config,
    update_latest_link,
)


def run_base_training(
    config: dict[str, Any], config_path: str | Path, process_name: str
) -> dict[str, Any]:
    logger = get_logger(process_name)
    run_dir = make_run_dir(ARTIFACTS_ROOT, process_name)
    snapshot_config(config_path, run_dir)
    logger.info("Run dir: %s", run_dir)
    model_name = config["model"]["name"]
    model_cfg = config["model"].get("params", {})
    model = create_model(model_name, model_cfg)
    train_ds = ParquetSequenceDataset(
        config.get("train_path", TRAIN_PATH),
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.get("batch_size", SEQUENCE_BATCH_SIZE)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = None
    val_path = config.get("val_path", VAL_PATH)
    if val_path:
        val_ds = ParquetSequenceDataset(
            val_path, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(config.get("batch_size", SEQUENCE_BATCH_SIZE)),
            shuffle=False,
            num_workers=0,
        )
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.get("device", DEVICE),
        epochs=int(config.get("epochs", 20)),
        lr=float(config.get("lr", 0.001)),
        run_dir=run_dir,
        warmup_steps=int(config.get("warmup_steps", WARMUP_STEPS)),
        loss_name=config.get("loss", "mse"),
    )
    plot_history(result.history, Path(run_dir) / "plots")
    save_json(result.history, Path(run_dir) / "metrics_history.json")
    save_json(
        {
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
            "weights_path": result.best_weights_path,
            "process": process_name,
        },
        Path(run_dir) / "manifest.json",
    )
    update_latest_link(run_dir)
    logger.info(
        "Done. Best epoch=%d metric=%.6f", result.best_epoch, result.best_metric
    )
    return {"run_dir": str(run_dir), "weights_path": result.best_weights_path}
