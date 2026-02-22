"""
Простой трейнер: обучение FullModel на ParquetDataset с валидацией и сохранением весов.
"""
import os
import sys
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# импорты из our_solution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from constants import DEVICE
from model import FullModel
from dataset import ParquetDataset, MIN_BATCH_MULTIPLE

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# колонки: seq_ix (для сброса state при смене последовательности), 32 признака, t0, t1
FEATURE_COLUMNS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)
TARGET_COLUMNS = ["t0", "t1"]
SEQ_IX_COL = "seq_ix"
TRAIN_COLUMNS = [SEQ_IX_COL] + FEATURE_COLUMNS + TARGET_COLUMNS
INPUT_DIM = len(FEATURE_COLUMNS)
TARGET_DIM = len(TARGET_COLUMNS)
# в батче: [seq_ix, ...features..., t0, t1]
OFFSET_FEATURES = 1
OFFSET_TARGETS = OFFSET_FEATURES + INPUT_DIM

# папка для сохранения весов
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")

# пути к данным и параметры обучения (вместо аргументов командной строки)
TRAIN_PATH = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
VAL_PATH = os.path.join(CURRENT_DIR, "..", "datasets", "valid.parquet")
VAL_FROM_TRAIN_BATCHES = 50  # первых N батчей train как val, если VAL_PATH недоступен; 0 = без val
# Батч загрузки датасета — сколько строк за раз читаем из parquet. Должен быть кратен MIN_BATCH_MULTIPLE.
LOADER_BATCH_SIZE = MIN_BATCH_MULTIPLE * 50
# Батч обучения — размер одного шага optimizer.step(); должен делить LOADER_BATCH_SIZE.
TRAIN_BATCH_SIZE = MIN_BATCH_MULTIPLE * 2
assert LOADER_BATCH_SIZE % TRAIN_BATCH_SIZE == 0, "LOADER_BATCH_SIZE должен делиться на TRAIN_BATCH_SIZE"
EPOCHS = 3
LR = 1e-3
SAVE_NAME = "model.pt"


def _forward_batch_with_seq_ix(model: FullModel, x: torch.Tensor, seq_ix: torch.Tensor, eval_mode: bool) -> torch.Tensor:
    """Прогон батча: сбрасываем state только при смене seq_ix (как в README)."""
    if eval_mode:
        model.eval()
    outs = []
    for i in range(x.size(0)):
        if i == 0 or seq_ix[i].item() != seq_ix[i - 1].item():
            model.conv_model.state.reset()
        if eval_mode:
            with torch.no_grad():
                out = model(x[i])
        else:
            out = model(x[i])
        outs.append(out)
    return torch.stack(outs)


def train_epoch(model, loader, optimizer, criterion, device, epoch=None, total_epochs=None, total_train_batches=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    desc = f"Train"
    if epoch is not None and total_epochs is not None:
        desc = f"Train Epoch {epoch}/{total_epochs}"
    pbar = tqdm(total=total_train_batches, desc=desc, unit="batch", leave=True)
    for batch in loader:
        batch = batch.to(device)
        for start in range(0, batch.size(0), TRAIN_BATCH_SIZE):
            end = start + TRAIN_BATCH_SIZE
            sub_batch = batch[start:end]
            seq_ix = sub_batch[:, 0]
            x = sub_batch[:, OFFSET_FEATURES:OFFSET_TARGETS]
            y = sub_batch[:, OFFSET_TARGETS : OFFSET_TARGETS + TARGET_DIM]
            pred = _forward_batch_with_seq_ix(model, x, seq_ix, eval_mode=False)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            avg_loss = total_loss / n_batches
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
            model.conv_model.state.detach()
    pbar.close()
    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device, max_batches=None, epoch=None, total_epochs=None, total_val_batches=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    desc = "Val"
    if epoch is not None and total_epochs is not None:
        desc = f"Val Epoch {epoch}/{total_epochs}"
    pbar = tqdm(total=total_val_batches, desc=desc, unit="batch", leave=True)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for start in range(0, batch.size(0), TRAIN_BATCH_SIZE):
                end = start + TRAIN_BATCH_SIZE
                sub_batch = batch[start:end]
                seq_ix = sub_batch[:, 0]
                x = sub_batch[:, OFFSET_FEATURES:OFFSET_TARGETS]
                y = sub_batch[:, OFFSET_TARGETS : OFFSET_TARGETS + TARGET_DIM]
                pred = _forward_batch_with_seq_ix(model, x, seq_ix, eval_mode=True)
                loss = criterion(pred, y)
                total_loss += loss.item()
                n_batches += 1
                avg_loss = total_loss / n_batches
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
                model.conv_model.state.detach()
                if max_batches is not None and n_batches >= max_batches:
                    break
            if max_batches is not None and n_batches >= max_batches:
                break
    pbar.close()
    return total_loss / max(n_batches, 1)


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, SAVE_NAME)

    log.info("Device: %s", DEVICE)
    model = FullModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    if not os.path.isfile(TRAIN_PATH):
        log.error("Train file not found: %s", TRAIN_PATH)
        sys.exit(1)

    train_ds = ParquetDataset(TRAIN_PATH, batch_size=LOADER_BATCH_SIZE, columns=TRAIN_COLUMNS)
    # DataLoader с batch_size=1 т.к. ParquetDataset уже отдаёт батчи
    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=lambda x: x[0])
    train_batches_per_loader_batch = LOADER_BATCH_SIZE // TRAIN_BATCH_SIZE
    total_train_batches = len(train_ds) * train_batches_per_loader_batch

    val_loader = None
    val_max_batches = None  # в батчах обучения; None = все батчи
    total_val_batches = None
    if VAL_PATH and os.path.isfile(VAL_PATH):
        val_ds = ParquetDataset(VAL_PATH, batch_size=LOADER_BATCH_SIZE, columns=TRAIN_COLUMNS)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=lambda x: x[0])
        total_val_batches = len(val_ds) * train_batches_per_loader_batch
        log.info("Validation from %s", VAL_PATH)
    elif VAL_FROM_TRAIN_BATCHES > 0:
        val_ds = ParquetDataset(TRAIN_PATH, batch_size=LOADER_BATCH_SIZE, columns=TRAIN_COLUMNS)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=lambda x: x[0])
        val_max_batches = VAL_FROM_TRAIN_BATCHES * train_batches_per_loader_batch
        total_val_batches = val_max_batches
        log.info("Validation from first %s batches of train (val file not found or disabled)", VAL_FROM_TRAIN_BATCHES)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            epoch=epoch, total_epochs=EPOCHS,
            total_train_batches=total_train_batches,
        )
        log.info("Epoch %d train loss: %.6f", epoch, train_loss)

        if val_loader is not None:
            val_loss = validate(
                model, val_loader, criterion, DEVICE,
                max_batches=val_max_batches,
                epoch=epoch, total_epochs=EPOCHS,
                total_val_batches=total_val_batches,
            )
            log.info("Epoch %d val loss: %.6f", epoch, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                log.info("Saved best weights to %s", save_path)

    if val_loader is None:
        torch.save(model.state_dict(), save_path)
        log.info("Saved weights to %s", save_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
