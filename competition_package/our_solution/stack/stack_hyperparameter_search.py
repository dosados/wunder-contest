"""
Перебор гиперпараметров метамодели: размеры скрытых слоёв meta_head.
Обучение только на валидации (базовые модели заморожены). Лучшая конфигурация сохраняется.
"""
import os
import sys
import json
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stack.stack_trainer import run_training_loop, WEIGHTS_STACK_DIR, SAVE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BEST_PARAMS_PATH = os.path.join(CURRENT_DIR, "best_hyperparameters_stack.json")


def get_search_grid():
    """Сетка для meta_head: meta_hidden_dims — список списков."""
    return {
        "meta_hidden_dims": [
            [64]
        ],
    }


def run_search(epochs_per_run=None, grid=None):
    """
    Перебор комбинаций. Для каждой обучает метамодель на valid, считает val Pearson.
    Лучшая конфигурация и веса сохраняются.
    """
    if grid is None:
        grid = get_search_grid()

    # Один ключ — meta_hidden_dims, значение — список вариантов (каждый вариант — список)
    meta_options = grid["meta_hidden_dims"]
    total = len(meta_options)
    log.info("Всего конфигураций meta_head: %d", total)

    best_val_pearson = -float("inf")
    best_meta_hidden_dims = None
    run_weights_path = os.path.join(WEIGHTS_STACK_DIR, "stack_hp_run_best.pt")

    for idx, meta_hidden_dims in enumerate(meta_options, start=1):
        log.info("--- Конфигурация %d / %d: meta_hidden_dims = %s ---", idx, total, meta_hidden_dims)

        val_pearson = run_training_loop(
            meta_hidden_dims=meta_hidden_dims,
            save_path=run_weights_path,
            epochs=epochs_per_run,
        )

        log.info("Конфигурация %d / %d: val_pearson = %.6f", idx, total, val_pearson)

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_meta_hidden_dims = meta_hidden_dims
            best_weights_path = os.path.join(WEIGHTS_STACK_DIR, SAVE_NAME)
            import shutil
            shutil.copy(run_weights_path, best_weights_path)
            log.info(
                "Новый лучший результат: val_pearson = %.6f, сохранены веса в %s",
                best_val_pearson,
                best_weights_path,
            )

    os.makedirs(WEIGHTS_STACK_DIR, exist_ok=True)

    result = {
        "best_val_pearson": best_val_pearson,
        "best_hyperparameters": {
            "meta_hidden_dims": best_meta_hidden_dims,
        },
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info("Поиск завершён. Лучший val_pearson: %.6f", best_val_pearson)
    log.info("Лучшая конфигурация сохранена в %s", BEST_PARAMS_PATH)
    log.info("meta_hidden_dims: %s", best_meta_hidden_dims)
    print("Лучшая конфигурация:", json.dumps(result, indent=2, ensure_ascii=False))

    return best_val_pearson, result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Перебор гиперпараметров метамодели (размеры слоёв meta_head) по val Pearson"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Эпох на одну конфигурацию (по умолчанию из stack_trainer)",
    )
    args = parser.parse_args()
    run_search(epochs_per_run=args.epochs)
