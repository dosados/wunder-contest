"""
Парсинг датасета и буфер последовательности для пошагового инференса.

- SequenceBuffer: накапливает шаги одной последовательности, сбрасывается при
  смене seq_ix; отдаёт тензор (1, T, D) с левым паддингом для модели.
- load_sequences_from_parquet: загрузка независимых последовательностей для обучения.
"""

import numpy as np
from typing import List, Iterator, Optional, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .config import FEATURE_DIM, SEQ_LENGTH, TARGET_DIM


class SequenceBuffer:
    """
    Буфер для одной последовательности. Накапливает векторы состояния по шагам.
    При переходе к новой последовательности (новый seq_ix) нужно вызвать reset().
    """

    __slots__ = ("_steps", "_max_len", "_feature_dim", "_dtype")

    def __init__(
        self,
        max_length: int = SEQ_LENGTH,
        feature_dim: int = FEATURE_DIM,
        dtype: type = np.float32,
    ):
        self._max_len = max_length
        self._feature_dim = feature_dim
        self._dtype = dtype
        self._steps: List[np.ndarray] = []

    def reset(self) -> None:
        """Сбросить состояние при переходе к новой последовательности."""
        self._steps = []

    def append(self, state: np.ndarray) -> None:
        """Добавить один снимок состояния рынка (вектор длины feature_dim)."""
        state = np.asarray(state, dtype=self._dtype)
        if state.ndim == 1:
            assert state.shape[0] == self._feature_dim
            self._steps.append(state)
        else:
            raise ValueError(f"state must be 1D, got shape {state.shape}")

    def __len__(self) -> int:
        return len(self._steps)

    def get_sequence_padded(self, pad_value: float = 0.0) -> np.ndarray:
        """
        Вернуть последовательность в виде (1, T, D) с левым паддингом:
        в начале нули, затем фактические шаги до текущего момента.
        T = max_length (SEQ_LENGTH).
        """
        n = len(self._steps)
        if n == 0:
            return np.zeros((1, self._max_len, self._feature_dim), dtype=self._dtype)
        arr = np.stack(self._steps, axis=0)  # (n, D)
        if n >= self._max_len:
            # обрезать слева, оставить последние max_length шагов
            arr = arr[-self._max_len :]
            return np.expand_dims(arr, axis=0)
        pad_len = self._max_len - n
        padding = np.full((pad_len, self._feature_dim), pad_value, dtype=self._dtype)
        full = np.concatenate([padding, arr], axis=0)  # (max_len, D)
        return np.expand_dims(full, axis=0)


def load_sequences_from_parquet(
    path: str,
    feature_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    seq_id_col: str = "seq_ix",
    step_col: str = "step_in_seq",
) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Загружает parquet и отдаёт по одной последовательности: (X, y).
    X: (T, D), y: (T, 2) или None если колонок целей нет в файле.

    feature_columns / target_columns — списки имён колонок. Если None,
    берутся по соглашению: после seq_ix, step_in_seq, need_prediction идут
    32 признака, затем t0, t1.
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for load_sequences_from_parquet")
    df = pd.read_parquet(path)
    cols = list(df.columns)

    if feature_columns is None or target_columns is None:
        # По data_description и utils: колонки 0,1,2 — meta, 3:35 — 32 фичи, 35:37 — t0, t1
        feature_columns = cols[3 : 3 + FEATURE_DIM]
        target_columns = cols[3 + FEATURE_DIM : 3 + FEATURE_DIM + TARGET_DIM] if len(cols) >= 3 + FEATURE_DIM + TARGET_DIM else None

    grouped = df.groupby(seq_id_col, sort=False)
    for _seq_ix, grp in grouped:
        grp = grp.sort_values(step_col)
        X = grp[feature_columns].values.astype(np.float32)
        if target_columns and all(c in grp.columns for c in target_columns):
            y = grp[target_columns].values.astype(np.float32)
        else:
            y = None
        yield X, y


def yield_training_batches(
    path: str,
    batch_size: int,
    seq_length: int = SEQ_LENGTH,
    feature_dim: int = FEATURE_DIM,
    pad_value: float = 0.0,
    shuffle_seqs: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Для обучения: выдаёт батчи (X, y).
    X: (B, seq_length, feature_dim) — левый паддинг по последовательности до текущего шага.
    y: (B, 2) — целевые t0, t1 на шагах, где need_prediction=True (шаги 99..999).
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required")
    df = pd.read_parquet(path)
    feature_cols = df.columns[3 : 3 + FEATURE_DIM].tolist()
    target_cols = df.columns[3 + FEATURE_DIM : 3 + FEATURE_DIM + TARGET_DIM].tolist()
    seq_ids = df["seq_ix"].unique()
    if shuffle_seqs:
        np.random.shuffle(seq_ids)

    buffer_X: List[np.ndarray] = []
    buffer_y: List[np.ndarray] = []
    for seq_ix in seq_ids:
        grp = df[df["seq_ix"] == seq_ix].sort_values("step_in_seq")
        X_seq = grp[feature_cols].values.astype(np.float32)
        y_seq = grp[target_cols].values.astype(np.float32)
        need_pred = grp["need_prediction"].values
        for t in range(seq_length):
            if not need_pred[t]:
                continue
            # Вход: последовательность от 0 до t включительно, паддинг слева до seq_length
            x_t = X_seq[: t + 1]
            if x_t.shape[0] >= seq_length:
                x_t = x_t[-seq_length:]
            else:
                pad = np.full((seq_length - x_t.shape[0], feature_dim), pad_value, dtype=np.float32)
                x_t = np.concatenate([pad, x_t], axis=0)
            buffer_X.append(x_t)
            buffer_y.append(y_seq[t])
        while len(buffer_X) >= batch_size:
            yield (
                np.stack(buffer_X[:batch_size], axis=0),
                np.stack(buffer_y[:batch_size], axis=0),
            )
            buffer_X = buffer_X[batch_size:]
            buffer_y = buffer_y[batch_size:]
    if buffer_X:
        yield np.stack(buffer_X, axis=0), np.stack(buffer_y, axis=0)


def get_feature_and_target_columns_from_parquet(path: str) -> Tuple[List[str], List[str]]:
    """Возвращает (feature_columns, target_columns) по структуре файла."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required")
    df = pd.read_parquet(path)
    cols = list(df.columns)
    # первые 3: seq_ix, step_in_seq, need_prediction; затем 32 фичи; затем t0, t1
    feature_columns = cols[3 : 3 + FEATURE_DIM]
    target_columns = cols[3 + FEATURE_DIM : 3 + FEATURE_DIM + TARGET_DIM]
    return feature_columns, target_columns
