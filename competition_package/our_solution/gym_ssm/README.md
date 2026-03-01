# Gym SSM — обучение второй модели стека на остатках

SSM (Mamba-style) обучается предсказывать **остаток** первой модели (LSTM). Данные — parquet с остатками, подготовленные `build_residual_dataset.py`. Loss: отрицательная взвешенная корреляция Пирсона между предсказанием SSM и остатком (см. plan.txt).

## Что нужно до запуска

- Файлы `datasets/train_residual.parquet` и `datasets/valid_residual.parquet` (и при необходимости `datasets/residual_metadata.json`). Их создаёт `build_residual_dataset.py`.

## Запуск

Из каталога **`our_solution`**:

```bash
python -m gym_ssm.trainer
```

- Обучаются 3 прогона по 20 эпох (константы `NUM_RUNS`, `EPOCHS_SSM` в `trainer.py` и `constants.py`).
- Лучшие веса по валидации сохраняются в:
  - `weights_ssm/ssm_model.pt` — сюда по умолчанию смотрит `check.py`;
  - `weights_ssm/run_DDMMYY_HHMMSS/ssm_model.pt` — копия в папку текущего run.

## Конфиг

Параметры SSM задаются в `constants.ssm_constants`. Эпохи и learning rate: `EPOCHS_SSM`, `LR_SSM` в `constants.py`.
