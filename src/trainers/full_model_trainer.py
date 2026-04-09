from dataset import ParquetSequenceDataset
from models import FullModel
from constants import (
    DEVICE,
    EPOCHS,
    FEATURE_COLUMNS,
    LR,
    SAVE_NAME,
    SEQUENCE_BATCH_SIZE,
    TARGET_COLUMNS,
    TRAIN_PATH,
    VAL_PATH,
    WEIGHTS_DIR,
)
import os
import torch
from torch.utils.data import DataLoader


def run_training_loop(save_path=None, epochs=None):
    if save_path is None:
        save_path = os.path.join(WEIGHTS_DIR, SAVE_NAME)
    if epochs is None:
        epochs = EPOCHS
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model = FullModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    train_ds = ParquetSequenceDataset(
        TRAIN_PATH, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS
    )
    train_loader = DataLoader(
        train_ds, batch_size=SEQUENCE_BATCH_SIZE, shuffle=True, num_workers=0
    )
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model.forward_sequence(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), save_path)
    return save_path if os.path.isfile(VAL_PATH) else save_path
