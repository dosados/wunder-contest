"""
Перебор комбинаций гиперпараметров для GRU-модели (constants.gru_constants).
Лучшая комбинация по val weighted Pearson сохраняется в best_hyperparameters_gru.json.
Веса — в weights_gru/.
"""
import os
import sys
import json
import itertools
import logging
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import constants
from constants import BEST_PARAMS_PATH_GRU as BEST_PARAMS_PATH, WEIGHTS_GRU_DIR
from gru_trainer import run_training_loop, SAVE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

WEIGHTS_DIR = WEIGHTS_GRU_DIR
BEST_WEIGHTS_NAME = "gru_best_search.pt"


def get_search_grid():
    """Сетка для GRU: ключи как в constants.gru_constants."""
    return {
        "input_dim": [32],
        "linear_dim": [64],
        "gru_hidden": [128, 256, 320],
        "gru_num_layers": [1, 2, 3],
        "output_dim": [2],
    }


def dict_product(grid):
    """Все комбинации: список словарей."""
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def run_search(epochs_per_run=None, grid=None):
    """
    Перебор комбинаций. Для каждой обновляет constants.gru_constants,
    обучает GRUModel, считает val Pearson. Лучшая комбинация и веса сохраняются.
    """
    if grid is None:
        grid = get_search_grid()

    combinations = list(dict_product(grid))
    total = len(combinations)
    log.info("Всего комбинаций: %d", total)

    best_val_pearson = -float("inf")
    best_combo = None
    best_weights_path = None
    run_weights_path = os.path.join(WEIGHTS_DIR, "gru_hp_run_best.pt")

    for idx, combo in enumerate(combinations, start=1):
        constants.gru_constants.clear()
        constants.gru_constants.update(combo)

        log.info("--- Комбинация %d / %d ---", idx, total)
        print("Текущая комбинация:", json.dumps(combo, indent=2))

        val_pearson = run_training_loop(
            save_path=run_weights_path,
            epochs=epochs_per_run,
        )

        log.info("Комбинация %d / %d: val_pearson = %.6f", idx, total, val_pearson)

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_combo = combo.copy()
            best_weights_path = os.path.join(WEIGHTS_DIR, BEST_WEIGHTS_NAME)
            shutil.copy(run_weights_path, best_weights_path)
            log.info(
                "Новый лучший результат: val_pearson = %.6f, сохранены веса в %s",
                best_val_pearson,
                best_weights_path,
            )

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    result = {
        "best_val_pearson": best_val_pearson,
        "best_hyperparameters": best_combo,
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info("Поиск завершён. Лучший val_pearson: %.6f", best_val_pearson)
    log.info("Лучшая комбинация сохранена в %s", BEST_PARAMS_PATH)
    log.info("Веса лучшей модели: %s", best_weights_path)
    print("Лучшая комбинация:", json.dumps(result, indent=2, ensure_ascii=False))

    return best_val_pearson, best_combo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Перебор гиперпараметров GRU по val Pearson")
    parser.add_argument("--epochs", type=int, default=None, help="Эпох на одну комбинацию (по умолчанию из gru_trainer)")
    args = parser.parse_args()
    run_search(epochs_per_run=args.epochs)
