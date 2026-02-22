import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Компактный быстрый прогресс-бар: минимум обновлений, одна строка
PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:16}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
PROGRESS_MININTERVAL = 0.3  # сек — реже обновлять для скорости

# Путь к competition_package для импорта utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, ".."))


import pyarrow.parquet as pq
from utils import DataPoint, ScorerStepByStep

from model import FullModel, ModelState
from constants import convo_constants, DEVICE

current_constants = convo_constants


class PredictionModel:
    """Обёртка над FullModel для пошагового инференса. Сбрасывает состояние при смене seq_ix."""

    def __init__(self):
        self.model = FullModel.from_dict(convo_constants).to(DEVICE)
        self.state = ModelState(
            conv1_window=self.model.conv1_window,
            conv2_window=self.model.conv2_window,
            conv3_window=self.model.conv3_window,
            conv1_feat_dim=self.model.linear1_dim,
            conv2_feat_dim=self.model.conv1_dim,
            conv3_feat_dim=self.model.conv2_dim,
            device=DEVICE,
        )
        self._current_seq_ix: int | None = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if not data_point.need_prediction:
            return None
        if self._current_seq_ix is not None and self._current_seq_ix != data_point.seq_ix:
            self.state.reset()
        self._current_seq_ix = data_point.seq_ix

        x = torch.from_numpy(data_point.state).float().to(DEVICE)
        with torch.no_grad():
            x = self.model.forward(x)
        return x.cpu().numpy().astype(np.float64)



from dataset import ParquetDataset, _check_batch_size, MIN_BATCH_MULTIPLE


