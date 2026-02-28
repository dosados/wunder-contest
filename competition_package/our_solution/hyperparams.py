"""
Перебор конфигураций FullModel (Conv + LSTM) и выбор лучшей по метрике weighted Pearson на валидации.
Обновляет constants.convo_constants для каждой конфигурации, вызывает trainer.run_training_loop.
Сохраняет лучший конфиг в best_hyperparameters.json и лучшие веса в saves/ или weights/.
"""
import os
import sys
import json
import logging
import itertools
import shutil
from copy import deepcopy
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import constants
from trainer import run_training_loop
from constants import (
    DEVICE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TRAIN_PATH,
    VAL_PATH,
    SEQUENCE_BATCH_SIZE,
    EPOCHS,
    ROOT,
    BEST_PARAMS_PATH,
    BEST_VAL_PEARSON_KEY,
    SAVES_DIR,
    WEIGHTS_DIR,
    SAVE_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def get_config_grid():
    """
    Сетка конфигов для FullModel (ключи как в constants.convo_constants).
    Можно заменить на случайный поиск или свои варианты.
    """
    return {
        "input_dim": [32],
        "linear1_dim": [64],
        "conv_window": [7],
        "conv_dim": [64],
        "lstm_hidden": [256],
        "lstm_num_layers": [1],
        "output_dim": [2],
    }


def _config_slug(config: dict) -> str:
    """Короткое читаемое имя конфига для имени папки."""
    return "conv{}_dim{}_lstm{}_l{}".format(
        config.get("conv_window", ""),
        config.get("conv_dim", ""),
        config.get("lstm_hidden", ""),
        config.get("lstm_num_layers", ""),
    )


def _dict_product(grid: dict):
    """Все комбинации: список словарей."""
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def run_single_config(
    config: dict,
    save_path: str,
    epochs: int,
) -> float:
    """
    Выставляет constants.convo_constants, обучает FullModel через run_training_loop.
    Возвращает лучший val weighted Pearson.
    """
    constants.convo_constants.clear()
    constants.convo_constants.update(deepcopy(config))
    return run_training_loop(save_path=save_path, epochs=epochs)


def run_hyperparameter_search(
    configs: list[dict] | None = None,
    epochs_per_config: int | None = None,
    max_configs: int | None = None,
    saves_dir: str | None = None,
) -> dict:
    """
    Перебирает конфиги FullModel, обучает для каждого, определяет лучший по val weighted Pearson.
    Для каждого конфига сохраняет в saves: конфиг с метрикой и лучшие веса по этому конфигу.
    Итоговый лучший конфиг (с best_val_pearson) сохраняется в BEST_PARAMS_PATH.
    Копия лучших весов — в WEIGHTS_DIR/SAVE_NAME при желании можно добавить отдельно.
    """
    if configs is None:
        configs = list(_dict_product(get_config_grid()))
    if epochs_per_config is None:
        epochs_per_config = min(EPOCHS, 10)
    if max_configs is not None:
        configs = configs[:max_configs]
    if saves_dir is None:
        saves_dir = SAVES_DIR

    if not os.path.isfile(TRAIN_PATH):
        log.error("Train file not found: %s", TRAIN_PATH)
        return {}

    run_name = "run_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = os.path.join(saves_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log.info("Saving configs and weights to %s", run_dir)

    results: list[tuple[dict, float]] = []
    for i, config in enumerate(configs):
        log.info("Config %d/%d: %s", i + 1, len(configs), config)
        slug = _config_slug(config)
        config_dir_name = "config_{:04d}_{}".format(i, slug)
        config_dir = os.path.join(run_dir, config_dir_name)
        os.makedirs(config_dir, exist_ok=True)
        save_path = os.path.join(config_dir, "best_weights.pt")

        best_pearson = run_single_config(
            deepcopy(config),
            save_path=save_path,
            epochs=epochs_per_config,
        )
        results.append((config, best_pearson))

        config_with_metric = deepcopy(config)
        config_with_metric[BEST_VAL_PEARSON_KEY] = best_pearson
        config_json_path = os.path.join(config_dir, "config.json")
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_with_metric, f, indent=2, ensure_ascii=False)
        log.info("Config %d best val Pearson: %.6f (saved to %s)", i + 1, best_pearson, config_dir)

    best_pair = max(results, key=lambda x: x[1])
    best_config, best_pearson = best_pair
    best_config_out = deepcopy(best_config)
    best_config_out[BEST_VAL_PEARSON_KEY] = best_pearson

    summary_path = os.path.join(run_dir, "best_config_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_config_out, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(BEST_PARAMS_PATH) or ".", exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_hyperparameters": best_config, BEST_VAL_PEARSON_KEY: best_pearson}, f, indent=2, ensure_ascii=False)

    best_weights_src = os.path.join(run_dir, "config_{:04d}_{}".format(results.index(best_pair), _config_slug(best_config)), "best_weights.pt")
    if os.path.isfile(best_weights_src):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        best_weights_dst = os.path.join(WEIGHTS_DIR, SAVE_NAME)
        shutil.copy(best_weights_src, best_weights_dst)
        log.info("Copied best weights to %s", best_weights_dst)

    log.info(
        "Best config: %s with %s = %.6f (summary in %s)",
        best_config_out,
        BEST_VAL_PEARSON_KEY,
        best_pearson,
        summary_path,
    )
    return best_config_out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter search for FullModel (Conv + LSTM)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per config (default: min(EPOCHS, 10))")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs to try")
    parser.add_argument("--saves-dir", type=str, default=SAVES_DIR, help="Directory for saves")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to copy best config JSON",
    )
    args = parser.parse_args()

    best = run_hyperparameter_search(
        epochs_per_config=args.epochs,
        max_configs=args.max_configs,
        saves_dir=args.saves_dir,
    )
    if best and args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        log.info("Saved best config copy to %s", args.save)


if __name__ == "__main__":
    main()
