"""
Скрипт: несколько раз с нуля обучает GRU-модель с заданным конфигом и сохраняет
лучшие веса (и конфиг) в inference_gru/ для дальнейшего использования.
"""
import os
import sys
import json
import shutil
import logging
import argparse

# our_solution в path до импорта gru_trainer (gru_trainer импортирует model и constants)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUR_SOLUTION_DIR = os.path.dirname(CURRENT_DIR)
if OUR_SOLUTION_DIR not in sys.path:
    sys.path.insert(0, OUR_SOLUTION_DIR)

PKG_DIR = os.path.dirname(OUR_SOLUTION_DIR)
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

import constants
from gru_trainer import run_training_loop, WEIGHTS_DIR

# Куда кладём лучшие веса и конфиг для inference GRU
INFERENCE_DIR = CURRENT_DIR
INFERENCE_WEIGHTS_DIR = os.path.join(INFERENCE_DIR, "weights")
INFERENCE_CONFIG_PATH = os.path.join(INFERENCE_DIR, "config.json")
INFERENCE_BEST_WEIGHTS_PATH = os.path.join(INFERENCE_WEIGHTS_DIR, "best_model.pt")

# Файл весов для текущего запуска обучения (перезаписывается каждым run)
RUN_WEIGHTS_NAME = "train_best_gru_run.pt"
RUN_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, RUN_WEIGHTS_NAME)

# Конфиг по умолчанию: из best_hyperparameters_gru.json или из constants.gru_constants
BEST_HP_PATH = os.path.join(OUR_SOLUTION_DIR, "best_hyperparameters_gru.json")
BEST_VAL_PEARSON_KEY = "best_val_pearson"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_config(config_path: str | None) -> dict:
    """Загружает конфиг из JSON. Если путь к файлу с best_hyperparameters — берёт best_hyperparameters."""
    path = config_path or BEST_HP_PATH
    if not os.path.isfile(path):
        log.warning("Config file not found %s, using constants.gru_constants", path)
        return dict(constants.gru_constants)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "best_hyperparameters" in data:
        return data["best_hyperparameters"]
    return data if isinstance(data, dict) else {}


def load_inference_config() -> tuple[dict, float]:
    """
    Загружает inference_gru config.json если есть.
    Возвращает (config_dict, all_time_best_pearson).
    all_time_best_pearson = -inf если файла нет или ключа best_val_pearson нет.
    """
    if not os.path.isfile(INFERENCE_CONFIG_PATH):
        return {}, -float("inf")
    with open(INFERENCE_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get(BEST_VAL_PEARSON_KEY)
    if raw is None:
        return data, -float("inf")
    try:
        best = float(raw)
    except (TypeError, ValueError):
        best = -float("inf")
    return data, best


def save_inference_config(best_hyperparameters: dict, best_val_pearson: float | None) -> None:
    """Пишет config.json с best_hyperparameters и опционально best_val_pearson."""
    payload = {"best_hyperparameters": best_hyperparameters}
    if best_val_pearson is not None and best_val_pearson > -float("inf"):
        payload[BEST_VAL_PEARSON_KEY] = best_val_pearson
    with open(INFERENCE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_multiple_trains(
    num_runs: int = 3,
    config_path: str | None = None,
    epochs_per_run: int | None = None,
) -> float:
    """
    Запускает num_runs обучений GRU с нуля с одним и тем же конфигом.
    Лучшие по val Pearson веса сохраняются в inference_gru/ только при перебитии
    all-time best метрики (хранится в inference_gru/config.json).
    Конфиг записывается с best_hyperparameters и best_val_pearson.
    Возвращает лучший val_pearson за текущую сессию.
    """
    config = load_config(config_path)
    constants.gru_constants.clear()
    constants.gru_constants.update(config)

    os.makedirs(INFERENCE_WEIGHTS_DIR, exist_ok=True)
    _, all_time_best = load_inference_config()
    save_inference_config(config, all_time_best if all_time_best > -float("inf") else None)
    log.info("Config applied: %s", json.dumps(config, indent=2))
    if all_time_best > -float("inf"):
        log.info("All-time best val_pearson (from config): %.6f", all_time_best)
    else:
        log.info("All-time best: not set (no previous config)")
    log.info("Inference GRU config path: %s", INFERENCE_CONFIG_PATH)

    for run in range(1, num_runs + 1):
        log.info("=== Run %d / %d ===", run, num_runs)
        val_pearson = run_training_loop(
            save_path=RUN_WEIGHTS_PATH,
            epochs=epochs_per_run,
        )
        log.info("Run %d val_pearson: %.6f", run, val_pearson)

        if val_pearson > all_time_best:
            all_time_best = val_pearson
            shutil.copy(RUN_WEIGHTS_PATH, INFERENCE_BEST_WEIGHTS_PATH)
            save_inference_config(config, all_time_best)
            log.info("New all-time best. Weights and config saved (best_val_pearson=%.6f)", all_time_best)

    log.info("Done. Best val_pearson this session: %.6f", all_time_best)
    log.info("Use inference_gru with: config=%s, weights=%s", INFERENCE_CONFIG_PATH, INFERENCE_BEST_WEIGHTS_PATH)
    return all_time_best


def main():
    parser = argparse.ArgumentParser(
        description="Обучить GRU-модель несколько раз с одним конфигом и сохранить лучшие веса в inference_gru/"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Количество запусков обучения с нуля (по умолчанию 3)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к JSON с конфигом (по умолчанию best_hyperparameters_gru.json в our_solution)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Эпох за один запуск (по умолчанию из gru_trainer)",
    )
    args = parser.parse_args()
    run_multiple_trains(
        num_runs=args.runs,
        config_path=args.config,
        epochs_per_run=args.epochs,
    )


if __name__ == "__main__":
    main()
