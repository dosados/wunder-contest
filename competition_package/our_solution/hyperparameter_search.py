"""
Перебор комбинаций гиперпараметров из constants.convo_constants.
Лучшая комбинация выбирается по метрике на валидации (weighted Pearson, как в trainer).
Лучший набор сохраняется в файл, текущая комбинация выводится в консоль.
"""
import os
import sys
import json
import itertools
import logging
import shutil

# чтобы импорты our_solution работали
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import constants
from trainer import run_training_loop, WEIGHTS_DIR, SAVE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Файл, куда сохраняем лучшую комбинацию гиперпараметров
BEST_PARAMS_PATH = os.path.join(CURRENT_DIR, "best_hyperparameters.json")
# Имя весов для лучшей модели по итогам поиска
BEST_WEIGHTS_NAME = "model4_best_search.pt"


def get_search_grid():
    """
    Сетка значений для перебора. Ключи — как в constants.convo_constants.
    input_dim и output_dim заданы данными (32 и 2), их можно не перебирать или оставить один вариант.
    """
    return {
        "input_dim": [32],
        "linear1_dim": [64],
        "conv_window": [2],
        "conv_dim": [64],
        "lstm_hidden": [320],
        "lstm_num_layers": [3],
        "output_dim": [2],
    }


def dict_product(grid):
    """Генерирует все комбинации: список словарей с полным набором ключей."""
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def run_search(epochs_per_run=None, grid=None):
    """
    Запускает перебор комбинаций. Для каждой обновляет constants.convo_constants,
    обучает модель, считает val Pearson. Лучшая комбинация и её веса сохраняются.
    """
    if grid is None:
        grid = get_search_grid()

    combinations = list(dict_product(grid))
    total = len(combinations)
    log.info("Всего комбинаций: %d", total)

    best_val_pearson = -float("inf")
    best_combo = None
    best_weights_path = None
    run_weights_path = os.path.join(WEIGHTS_DIR, "hp4_run_best.pt")

    for idx, combo in enumerate(combinations, start=1):
        # Подставляем текущую комбинацию (модули model читают convo_constants по ссылке)
        constants.convo_constants.clear()
        constants.convo_constants.update(combo)

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
            log.info("Новый лучший результат: val_pearson = %.6f, сохранены веса в %s", best_val_pearson, best_weights_path)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Сохраняем лучшую комбинацию
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
    parser = argparse.ArgumentParser(description="Перебор гиперпараметров по val Pearson")
    parser.add_argument("--epochs", type=int, default=None, help="Эпох на одну комбинацию (по умолчанию из trainer)")
    args = parser.parse_args()
    run_search(epochs_per_run=args.epochs)
