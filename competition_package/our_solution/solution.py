"""
Предсказание (t0, t1) по последовательности состояний рынка.
Архитектура: Hybrid Conv + Transformer (см. description.md).
При смене seq_ix состояние буфера сбрасывается.
"""

import os
import sys
import numpy as np

# Пути: our_solution — для data/config/model, родитель — для utils
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CURRENT_DIR)
sys.path.insert(0, os.path.dirname(_CURRENT_DIR))

from utils import DataPoint

from data import SequenceBuffer
from config import DEFAULT_CONFIG, SEQ_LENGTH, FEATURE_DIM
from model import HybridConvTransformer

try:
    import torch
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    _DEVICE = None


class PredictionModel:
    """
    Модель предсказания следующего состояния рынка.
    Накапливает последовательность в буфере; при переходе к новой последовательности
    (новый seq_ix) состояние буфера сбрасывается.
    """

    def __init__(self, config=None, checkpoint_path: str | None = None):
        self.config = config or DEFAULT_CONFIG
        self._buffer = SequenceBuffer(
            max_length=SEQ_LENGTH,
            feature_dim=FEATURE_DIM,
        )
        self._current_seq_ix: int | None = None
        self._model = None
        if torch is not None:
            self._model = HybridConvTransformer(self.config).to(_DEVICE)
            self._model.eval()
            if checkpoint_path and os.path.isfile(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=_DEVICE)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                elif hasattr(state, "state_dict"):
                    state = state.state_dict()
                self._model.load_state_dict(state, strict=False)

    def _reset_if_new_sequence(self, seq_ix: int) -> None:
        """Сбросить буфер при переходе к новой последовательности."""
        if self._current_seq_ix is not None and self._current_seq_ix != seq_ix:
            self._buffer.reset()
        self._current_seq_ix = seq_ix

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if not data_point.need_prediction:
            # Всё равно накапливаем историю для контекста
            self._reset_if_new_sequence(data_point.seq_ix)
            self._buffer.append(data_point.state)
            return None

        self._reset_if_new_sequence(data_point.seq_ix)
        self._buffer.append(data_point.state)

        if self._model is None:
            return np.zeros(2, dtype=np.float32)

        with torch.no_grad():
            x = self._buffer.get_sequence_padded()  # (1, T, D)
            x_t = torch.from_numpy(x).to(_DEVICE)
            out = self._model(x_t)
            pred = out[0].cpu().numpy()
        return pred.astype(np.float64)


if __name__ == "__main__":
    from utils import ScorerStepByStep
    valid_path = os.path.join(os.path.dirname(_CURRENT_DIR), "datasets", "valid.parquet")
    if os.path.isfile(valid_path):
        model = PredictionModel()
        scorer = ScorerStepByStep(valid_path)
        print("Running Hybrid Conv+Transformer on valid.parquet...")
        results = scorer.score(model)
        print("Results:", results)
    else:
        print("Valid parquet not found:", valid_path)
