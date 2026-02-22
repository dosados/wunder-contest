import logging
import os
import sys

import torch

# чтобы импорты model, constants, dataset работали
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

from constants import convo_constants, DEVICE
from model import FullModel
from dataset import ParquetDataset

INPUT_DIM = convo_constants["input_dim"]
DEFAULT_DATA_PATH = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
MIN_BATCH_MULTIPLE = 1000

# Колонки признаков по README (вход модели); таргеты t0, t1 в модель не подаём
FEATURE_COLUMNS = (
    [f"p{i}" for i in range(12)]  # p0–p11: bid/ask prices
    + [f"v{i}" for i in range(12)]  # v0–v11: bid/ask volumes
    + [f"dp{i}" for i in range(4)]  # dp0–dp3: trade prices
    + [f"dv{i}" for i in range(4)]   # dv0–dv3: trade volumes
)
assert len(FEATURE_COLUMNS) == INPUT_DIM, "FEATURE_COLUMNS должен иметь длину input_dim"


def main():
    log.info("Этап 1: инициализация FullModel")
    model = FullModel().to(DEVICE)
    model.eval()
    log.info("FullModel создана и переведена в режим eval")

    log.info("Этап 2: загрузка одного батча из датасета")
    if not os.path.isfile(DEFAULT_DATA_PATH):
        log.warning("Файл %s не найден, используем случайный батч [%s, %s]", DEFAULT_DATA_PATH, MIN_BATCH_MULTIPLE, INPUT_DIM)
        batch = torch.randn(MIN_BATCH_MULTIPLE, INPUT_DIM, device=DEVICE, dtype=torch.float32)
    else:
        dataset = ParquetDataset(
            DEFAULT_DATA_PATH,
            batch_size=MIN_BATCH_MULTIPLE,
            columns=FEATURE_COLUMNS,
        )
        batch = next(iter(dataset))
        batch = batch.to(DEVICE)
    log.info("Батч загружен, shape: %s", tuple(batch.shape))

    log.info("Этап 3: прогон без обучения (forward)")
    with torch.no_grad():
        # Модель стриминговая: один вектор за раз. Берём первый сэмпл из батча.
        sample = batch[0]
        out = model(sample)
    log.info("Входной сэмпл shape: %s, выход модели shape: %s", tuple(sample.shape), tuple(out.shape))

    log.info("Этап 4: завершение")
    log.info("Тест пройден успешно")


if __name__ == "__main__":
    main()
