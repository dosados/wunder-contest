from __future__ import annotations
from pathlib import Path
from typing import Any
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
from training.engine import TrainResult
from utils import (
    get_logger,
    make_run_dir,
    plot_history,
    save_json,
    snapshot_config,
    update_latest_link,
)


def execute_training(
    config: dict[str, Any], process_name: str, run_dir: str | Path
) -> TrainResult:
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
    return train_model(
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


def run_training_job(
    config: dict[str, Any],
    process_name: str,
    run_dir: str | Path | None = None,
    *,
    snapshot_original_config: str | Path | None = None,
) -> tuple[TrainResult, Path]:
    run_dir = Path(
        run_dir if run_dir is not None else make_run_dir(ARTIFACTS_ROOT, process_name)
    )
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    if snapshot_original_config is not None:
        (run_dir / "config_snapshot").mkdir(parents=True, exist_ok=True)
        snapshot_config(snapshot_original_config, run_dir)
    else:
        save_json(config, run_dir / "config.json")
    result = execute_training(config, process_name, run_dir)
    return result, run_dir


def run_base_training(
    config: dict[str, Any], config_path: str | Path, process_name: str
) -> dict[str, Any]:
    logger = get_logger(process_name)
    run_dir = make_run_dir(ARTIFACTS_ROOT, process_name)
    snapshot_config(config_path, run_dir)
    logger.info("Run dir: %s", run_dir)
    result, _ = run_training_job(
        config,
        process_name,
        run_dir=run_dir,
        snapshot_original_config=config_path,
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
