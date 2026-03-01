# Pipeline: LSTM + SSM (остатки)

Краткий гайд по порядку запуска: датасет остатков → обучение SSM → проверка связки на valid.

---

## 1. Предусловия

- **LSTM** уже обучена на первой половине train (через `gym/trainer.py`, `TRAIN_DATA_PART = "first_half"`).
- В каталоге **`inference/weights/run_280226_235333/`** есть:
  - `config.json`
  - `best_model.pt`

Иначе шаг 2 не выполнится.

---

## 2. Создать датасеты остатков

Из каталога **`our_solution`**:

```bash
python build_residual_dataset.py
```

Без флагов строятся оба датасета. Результат:

- `../datasets/train_residual.parquet` — вторая половина train, в `t0`/`t1` остатки;
- `../datasets/valid_residual.parquet` — valid с остатками (центровка с train);
- `../datasets/residual_metadata.json` — метаданные центровки (для повторного построения valid).

Опции:

- `--train-only` — только train residual и metadata;
- `--valid-only` — только valid residual (нужен уже созданный `residual_metadata.json`).

---

## 3. Обучить SSM на остатках

Из каталога **`our_solution`**:

```bash
python -m gym_ssm.trainer
```

- Читает `TRAIN_RESIDUAL_PATH` и `VALID_RESIDUAL_PATH` из `constants.py`.
- Обучает SSM максимизировать Weighted Pearson между предсказанием и остатком (plan.txt).
- Сохраняет лучшие веса в **`weights_ssm/ssm_model.pt`** (и копию в `weights_ssm/run_*`).

Параметры: `EPOCHS_SSM`, `LR_SSM`, `ssm_constants` в `constants.py`; число прогонов — в `gym_ssm/trainer.py` (`NUM_RUNS`).

---

## 4. Проверить связку LSTM + SSM

Из каталога **`our_solution`**:

```bash
python check.py
```

- Загружает LSTM из `inference/weights/run_280226_235333/` и SSM из `weights_ssm/ssm_model.pt`.
- Гоняет **исходный** `../datasets/valid.parquet` по шагам: `pred = LSTM(x) + SSM(x)`, клип [-6, 6].
- Выводит Weighted Pearson: только LSTM, только SSM, **LSTM + SSM** (итог).

Опции: `--lstm-run`, `--ssm-weights`, `--valid`.

---

## 5. (Опционально) Обучаемые коэффициенты смешивания

Из каталога **`our_solution`**:

```bash
python blend_coefficients.py
```

- Загружает LSTM и SSM, прогоняет `valid.parquet`.
- Первую **половину последовательностей** использует для обучения 4 коэффициентов:  
  `pred = alpha_k * LSTM + beta_k * SSM` по каждому таргету (k=0,1), максимизация Weighted Pearson.
- Вторую половину — для валидации. Сохраняет коэффициенты в **`blend_coefficients.json`** для использования в инференсе вместо линейной суммы.

Опции: `--valid`, `--lstm-run`, `--ssm-weights`, `--batched`.

---

## Порядок и пути (итого)

| Шаг | Команда | Результат |
|-----|--------|-----------|
| 1 | (уже есть) LSTM в `inference/weights/run_280226_235333/` | — |
| 2 | `python build_residual_dataset.py` | `train_residual.parquet`, `valid_residual.parquet`, `residual_metadata.json` |
| 3 | `python -m gym_ssm.trainer` | `weights_ssm/ssm_model.pt` |
| 4 | `python check.py` | Метрики на valid: LSTM, SSM, LSTM+SSM |
| 5 | `python blend_coefficients.py` | `blend_coefficients.json` (коэффициенты для инференса) |

Все команды запускаются из **`competition_package/our_solution`**.
