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
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm

# Логгер для этапов работы (настраивается в __main__)
logger = logging.getLogger("our_solution")

# Компактный быстрый прогресс-бар: минимум обновлений, одна строка
PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:16}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
PROGRESS_MININTERVAL = 0.3  # сек — реже обновлять для скорости

# Путь к competition_package для импорта utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, ".."))
import pyarrow.parquet as pq
from utils import DataPoint, ScorerStepByStep

# Наша модель
from model import FullModel

# --- Признаковые и таргетные поля (по data_description.md и README) ---
# Признаки: p0–p11, v0–v11, dp0–dp3, dv0–dv3 (32 колонки, индексы 3:35 в parquet)
# Таргеты: t0, t1 (2 колонки, индексы 35:37)
INPUT_DIM = 32
TARGET_DIM = 2
CONV_WINDOW = 3
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


# --- Датасет для обучения: окна по последовательностям, чтение parquet по батчам ---

# Размер батча при чтении parquet и размер батча сэмплов должны быть кратны 1000
READ_BATCH_MULTIPLE = 1000


def _check_batch_multiple(value: int, name: str) -> None:
    if value <= 0 or value % READ_BATCH_MULTIPLE != 0:
        raise ValueError(
            f"{name} должен быть положительным и кратным {READ_BATCH_MULTIPLE}, получено: {value}"
        )


class SequenceWindowDataset(IterableDataset):
    """
    Строит сэмплы (окно [conv_window, INPUT_DIM], таргет [2]) из parquet без загрузки файла целиком.
    Чтение parquet — батчами (размер кратен 1000); выдаёт батчи сэмплов (размер кратен 1000).
    Окно — последние conv_window шагов перед шагом step; таргет только где need_prediction=True.
    """

    def __init__(
        self,
        parquet_path: str,
        conv_window: int = CONV_WINDOW,
        read_batch_size: int = 10_000,
        train_batch_size: int = 1000,
    ):
        _check_batch_multiple(read_batch_size, "read_batch_size")
        _check_batch_multiple(train_batch_size, "train_batch_size")
        self.parquet_path = parquet_path
        self.conv_window = conv_window
        self.read_batch_size = read_batch_size
        self.train_batch_size = train_batch_size
        self._total_samples = None

    def _total_samples_from_metadata(self) -> int:
        if self._total_samples is None:
            pf = pq.ParquetFile(self.parquet_path)
            total_rows = pf.metadata.num_rows
            num_sequences = total_rows // 1000
            self._total_samples = num_sequences * 901
        return self._total_samples

    def _process_sequence(self, feat: np.ndarray, targ: np.ndarray, need: np.ndarray):
        """Для одной последовательности (feat, targ, need) выдаёт (window, target) для каждого need_prediction."""
        windows, targets = [], []
        n = len(need)
        for i in range(n):
            if not need[i]:
                continue
            start = max(0, i - self.conv_window + 1)
            window = feat[start : i + 1]
            if len(window) < self.conv_window:
                pad = np.zeros((self.conv_window - len(window), INPUT_DIM), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)
            windows.append(window)
            targets.append(targ[i])
        if windows:
            return np.stack(windows, axis=0), np.stack(targets, axis=0)
        return None, None

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        # Буфер строк неполной последовательности (переход на следующую seq)
        buf_seq_ix, buf_step, buf_need, buf_feat, buf_targ = [], [], [], [], []
        # Накопленные сэмплы перед выдачей батча
        pending_windows, pending_targets = [], []

        for record_batch in pf.iter_batches(batch_size=self.read_batch_size):
            seq_ix = record_batch.column(0).to_numpy()
            step = record_batch.column(1).to_numpy()
            need = record_batch.column(2).to_numpy()
            feat = np.column_stack([
                record_batch.column(j).to_numpy() for j in range(3, 35)
            ]).astype(np.float32)
            targ = np.column_stack([
                record_batch.column(j).to_numpy() for j in range(35, 37)
            ]).astype(np.float32)

            # Добавляем к буферу
            if buf_seq_ix:
                buf_seq_ix.append(seq_ix)
                buf_step.append(step)
                buf_need.append(need)
                buf_feat.append(feat)
                buf_targ.append(targ)
                seq_ix = np.concatenate(buf_seq_ix)
                step = np.concatenate(buf_step)
                need = np.concatenate(buf_need)
                feat = np.concatenate(buf_feat, axis=0)
                targ = np.concatenate(buf_targ, axis=0)
                buf_seq_ix, buf_step, buf_need, buf_feat, buf_targ = [], [], [], [], []
            else:
                seq_ix = np.asarray(seq_ix)
                step = np.asarray(step)
                need = np.asarray(need)

            # Границы последовательностей
            change = np.diff(seq_ix, prepend=seq_ix[0] - 1) != 0
            group_starts = np.where(change)[0]
            group_ends = np.concatenate([group_starts[1:], [len(seq_ix)]])

            # Последняя группа может быть неполной — оставляем в буфере
            if group_starts.size == 0:
                continue
            last_start, last_end = group_starts[-1], group_ends[-1]
            if last_end - last_start < 1000 and record_batch.num_rows > 0:
                # Неполная последовательность в конце — в буфер
                buf_seq_ix = [seq_ix[last_start:last_end]]
                buf_step = [step[last_start:last_end]]
                buf_need = [need[last_start:last_end]]
                buf_feat = [feat[last_start:last_end]]
                buf_targ = [targ[last_start:last_end]]
                groups = list(zip(group_starts[:-1], group_ends[:-1]))
            else:
                buf_seq_ix, buf_step, buf_need, buf_feat, buf_targ = [], [], [], [], []
                groups = list(zip(group_starts, group_ends))

            for start, end in groups:
                step_slice = step[start:end]
                order = np.argsort(step_slice)
                need_s = need[start:end][order]
                feat_s = feat[start:end][order]
                targ_s = targ[start:end][order]
                w, t = self._process_sequence(feat_s, targ_s, need_s)
                if w is not None:
                    pending_windows.append(w)
                    pending_targets.append(t)

            # Собираем накопленные сэмплы и выдаём батчами по train_batch_size
            while pending_windows:
                all_w = np.concatenate(pending_windows, axis=0)
                all_t = np.concatenate(pending_targets, axis=0)
                pending_windows, pending_targets = [], []
                off = 0
                while off + self.train_batch_size <= len(all_w):
                    w_batch = torch.from_numpy(all_w[off : off + self.train_batch_size])
                    t_batch = torch.from_numpy(all_t[off : off + self.train_batch_size])
                    yield w_batch, t_batch
                    off += self.train_batch_size
                if off < len(all_w):
                    pending_windows = [all_w[off:]]
                    pending_targets = [all_t[off:]]

        # Обработать последнюю неполную последовательность из буфера
        if buf_seq_ix:
            seq_ix = np.concatenate(buf_seq_ix)
            step = np.concatenate(buf_step)
            need = np.concatenate(buf_need)
            feat = np.concatenate(buf_feat, axis=0)
            targ = np.concatenate(buf_targ, axis=0)
            order = np.argsort(step)
            need = need[order]
            feat = feat[order]
            targ = targ[order]
            w, t = self._process_sequence(feat, targ, need)
            if w is not None:
                pending_windows.append(w)
                pending_targets.append(t)

        # Выдать остаток
        if pending_windows:
            all_w = np.concatenate(pending_windows, axis=0)
            all_t = np.concatenate(pending_targets, axis=0)
            yield torch.from_numpy(all_w), torch.from_numpy(all_t)


def train_epoch(model, dataset, optimizer, criterion, device, epoch=None, total_epochs=None):
    """dataset — IterableDataset, уже выдаёт батчи (window, target)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    desc = f"Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs else "Train"
    total_batches = getattr(dataset, "_total_samples", None)
    if total_batches is not None:
        total_batches = (total_batches + dataset.train_batch_size - 1) // dataset.train_batch_size
    pbar = tqdm(
        dataset,
        desc=desc,
        leave=False,
        total=total_batches,
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
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    return total_loss / num_batches if num_batches else 0.0


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
    logger.info("Устройство: %s%s", DEVICE, f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")

    train_file = "/home/timofey/Documents/own/wunder-contest/competition_package/datasets/train.parquet"
    valid_file = "/home/timofey/Documents/own/wunder-contest/competition_package/datasets/valid.parquet"
    save_path = os.path.join(CURRENT_DIR, "fullmodel.pt")

    if not os.path.exists(train_file):
        logger.error("Train parquet не найден: %s", train_file)
        sys.exit(1)

    # Параметры обучения (размеры батчей кратны 1000)
    read_batch_size = 100_000
    train_batch_size = 1000
    epochs = 3
    lr = 1e-3

    logger.info("Этап: датасет train.parquet (чтение по батчам %d)", read_batch_size)
    dataset = SequenceWindowDataset(
        train_file,
        conv_window=CONV_WINDOW,
        read_batch_size=read_batch_size,
        train_batch_size=train_batch_size,
    )
    dataset._total_samples = dataset._total_samples_from_metadata()
    logger.info("Этап: сэмплов (по метаданным) ≈%d, train_batch_size=%d", dataset._total_samples, train_batch_size)

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
            model, dataset, optimizer, criterion, DEVICE,
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
