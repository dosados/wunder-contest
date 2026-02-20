"""
Обёртка над FullModel (Conv + Transformer) для соревнования.
Класс PredictionModel совместим с ScorerStepByStep.
В __main__ — пример обучения и валидации с учётом признаковых и таргетных полей датасета.
"""

import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Логгер для этапов работы (настраивается в __main__)
logger = logging.getLogger("our_solution")

# Компактный быстрый прогресс-бар: минимум обновлений, одна строка
PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:16}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
PROGRESS_MININTERVAL = 0.3  # сек — реже обновлять для скорости

# Путь к competition_package для импорта utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, ".."))
from utils import DataPoint, ScorerStepByStep

# Наша модель
from model import FullModel

# --- Признаковые и таргетные поля (по data_description.md и README) ---
# Признаки: p0–p11, v0–v11, dp0–dp3, dv0–dv3 (32 колонки, индексы 3:35 в parquet)
# Таргеты: t0, t1 (2 колонки, индексы 35:37)
INPUT_DIM = 32
TARGET_DIM = 2
CONV_WINDOW = 100
TRANS_WINDOW = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PredictionModel:
    """
    Обёртка над FullModel для пошагового инференса.
    Сохраняет состояние последовательности (ModelState), сбрасывает при смене seq_ix.
    """

    def __init__(self, model_path: str = ""):
        self.current_seq_ix = None
        self.model = FullModel(
            input_dim=INPUT_DIM,
            d_model=128,
            conv_window=CONV_WINDOW,
            trans_window=TRANS_WINDOW,
            n_heads=8,
            n_layers=4,
        ).to(DEVICE)
        self.model.eval()
        self.state = self.model.create_state(device=DEVICE)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            logger.info("Загружена модель: %s", model_path)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix is not None and self.current_seq_ix != data_point.seq_ix:
            self.state.reset()
        self.current_seq_ix = data_point.seq_ix

        x = torch.tensor(data_point.state, dtype=torch.float32, device=DEVICE)
        self.state.add_raw(x)

        if not data_point.need_prediction:
            return None

        with torch.no_grad():
            raw_seq = self.state.get_raw_sequence()  # (L, D), L <= conv_window
            if raw_seq.dim() == 1:
                raw_seq = raw_seq.unsqueeze(0)
            L = raw_seq.size(0)
            if L < CONV_WINDOW:
                pad = torch.zeros(CONV_WINDOW - L, INPUT_DIM, device=DEVICE, dtype=raw_seq.dtype)
                raw_seq = torch.cat([pad, raw_seq], dim=0)
            raw_batch = raw_seq.unsqueeze(0)  # (1, conv_window, D)

            proj = self.model.projection(raw_batch)
            conv_out = self.model.conv(proj)
            last_conv = conv_out[0, -1]
            self.state.add_conv(last_conv)

            conv_seq = self.state.get_conv_sequence()  # (L2, d_model)
            if conv_seq.dim() == 1:
                conv_seq = conv_seq.unsqueeze(0)
            L2 = conv_seq.size(0)
            if L2 < TRANS_WINDOW:
                pad = torch.zeros(TRANS_WINDOW - L2, self.model.d_model, device=DEVICE, dtype=conv_seq.dtype)
                conv_seq = torch.cat([pad, conv_seq], dim=0)
            conv_batch = conv_seq.unsqueeze(0)  # (1, trans_window, d_model)

            trans_out = self.model.transformer(conv_batch)
            last_token = trans_out[0, -1].unsqueeze(0)
            logits = self.model.head(last_token)
            pred = logits[0].cpu().numpy().astype(np.float32)

        pred = np.clip(pred, -6.0, 6.0)
        return pred


# --- Датасет для обучения: окна по последовательностям, таргеты только где need_prediction ---

class SequenceWindowDataset(Dataset):
    """
    По parquet строит сэмплы (окно признаков [conv_window, INPUT_DIM], таргет [2]).
    Окно — последние conv_window шагов перед шагом step; таргет берётся на step.
    Используются только шаги с need_prediction=True (с 99 по 999).
    """

    def __init__(self, parquet_path: str, conv_window: int = CONV_WINDOW):
        import pandas as pd
        logger.info("Чтение parquet: %s", parquet_path)
        self.df = pd.read_parquet(parquet_path)
        self.conv_window = conv_window
        feat_cols = self.df.columns[3:35].tolist()
        targ_cols = self.df.columns[35:37].tolist()
        self.sequences = []  # список (feat_arr, targ_arr) по последовательностям
        self.samples = []    # (seq_idx, end_step) — индекс последовательности и шаг предсказания
        by_seq = self.df.groupby("seq_ix", sort=False)
        for _, grp in by_seq:
            grp = grp.sort_values("step_in_seq")
            need = grp["need_prediction"].values
            feat = grp[feat_cols].values.astype(np.float32)
            targ = grp[targ_cols].values.astype(np.float32)
            seq_idx = len(self.sequences)
            self.sequences.append((feat, targ))
            for i in range(len(grp)):
                if need[i]:
                    self.samples.append((seq_idx, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, end = self.samples[idx]
        feat, targ = self.sequences[seq_idx]
        start = max(0, end - self.conv_window + 1)
        window = feat[start : end + 1]
        if len(window) < self.conv_window:
            pad = np.zeros((self.conv_window - len(window), INPUT_DIM), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        target = targ[end]
        return torch.from_numpy(window), torch.from_numpy(target)


def train_epoch(model, loader, optimizer, criterion, device, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0
    desc = f"Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs else "Train"
    pbar = tqdm(
        loader,
        desc=desc,
        leave=False,
        bar_format=PROGRESS_BAR_FORMAT,
        mininterval=PROGRESS_MININTERVAL,
        dynamic_ncols=True,
    )
    for window, target in pbar:
        window = window.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        out = model(window)
        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    return total_loss / len(loader)


def validate_scorer(model_class, valid_path, model_path=None):
    """Запуск ScorerStepByStep на valid.parquet с нашей моделью."""
    logger.info("Валидация: загрузка модели и скорера")
    model = model_class(model_path=model_path or "")
    scorer = ScorerStepByStep(valid_path)
    logger.info("Валидация: прогон по valid.parquet")
    return scorer.score(model)


def _setup_logging():
    """Настройка логгера: один раз в __main__."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


if __name__ == "__main__":
    _setup_logging()

    train_file = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
    valid_file = os.path.join(CURRENT_DIR, "..", "datasets", "valid.parquet")
    save_path = os.path.join(CURRENT_DIR, "fullmodel.pt")

    if not os.path.exists(train_file):
        logger.error("Train parquet не найден: %s", train_file)
        sys.exit(1)

    # Параметры обучения
    batch_size = 64
    epochs = 3
    lr = 1e-3

    logger.info("Этап: загрузка датасета train.parquet")
    dataset = SequenceWindowDataset(train_file, conv_window=CONV_WINDOW)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    logger.info("Этап: датасет загружен, сэмплов=%d, batch_size=%d", len(dataset), batch_size)

    logger.info("Этап: инициализация FullModel (Conv + Transformer)")
    model = FullModel(
        input_dim=INPUT_DIM,
        d_model=128,
        conv_window=CONV_WINDOW,
        trans_window=TRANS_WINDOW,
        n_heads=8,
        n_layers=4,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger.info("Этап: обучение, epochs=%d, lr=%s", epochs, lr)
    for ep in range(epochs):
        loss = train_epoch(
            model, loader, optimizer, criterion, DEVICE,
            epoch=ep + 1, total_epochs=epochs,
        )
        logger.info("Epoch %d/%d  loss=%.6f", ep + 1, epochs, loss)

    logger.info("Этап: сохранение модели в %s", save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info("Модель сохранена")

    if os.path.exists(valid_file):
        logger.info("Этап: валидация на valid.parquet")
        results = validate_scorer(PredictionModel, valid_file, model_path=save_path)
        logger.info("Результаты валидации:")
        logger.info("  Weighted Pearson: %.6f", results["weighted_pearson"])
        for k, v in results.items():
            if k != "weighted_pearson":
                logger.info("  %s: %.6f", k, v)
    else:
        logger.warning("Valid parquet не найден: %s", valid_file)
