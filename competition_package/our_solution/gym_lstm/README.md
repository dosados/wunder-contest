# Gym LSTM — обучение второй LSTM на остатках первой

Вторая LSTM (ResidualLSTM) обучается предсказывать **остаток** первой модели (Conv+LSTM). Данные — те же parquet с остатками, что и для SSM: `build_residual_dataset.py`. Loss: комбинированный (Pearson + MSE), как в gym_ssm.

## Что нужно до запуска

- Файлы `datasets/train_residual.parquet` и `datasets/valid_residual.parquet` (создаёт `build_residual_dataset.py`).

## Запуск

Из каталога **`our_solution`**:

```bash
python -m gym_lstm.trainer
```

- Обучаются 3 прогона по 20 эпох (константы `NUM_RUNS`, `EPOCHS_LSTM2` в `trainer.py` и `constants.py`).
- Лучшие веса сохраняются в:
  - `weights_lstm2/lstm2_model.pt` — по умолчанию смотрит `lstm_check.py`;
  - `weights_lstm2/run_DDMMYY_HHMMSS/lstm2_model.pt` — копия в папку run.

## Конфиг

Параметры второй LSTM задаются в `constants.lstm2_constants`. Эпохи и learning rate: `EPOCHS_LSTM2`, `LR_LSTM2` в `constants.py`.
