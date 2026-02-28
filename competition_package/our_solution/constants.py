"""
Все конфиги и константы решения в одном месте.
"""
import os

# Корень our_solution (папка, в которой лежит constants.py)
ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Данные: колонки и размерности ---
FEATURE_COLUMNS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)
TARGET_COLUMNS = ["t0", "t1"]
INPUT_DIM = len(FEATURE_COLUMNS)  # 32
TARGET_DIM = len(TARGET_COLUMNS)  # 2

# --- Последовательности и обучение ---
WARMUP_STEPS = 99
SEQUENCE_BATCH_SIZE = 16
SEQ_LEN = 1000
MIN_BATCH_MULTIPLE = 1000

# --- Пути к данным (относительно competition_package) ---
PKG_DIR = os.path.dirname(ROOT)
TRAIN_PATH = os.path.join(PKG_DIR, "datasets", "train.parquet")
VAL_PATH = os.path.join(PKG_DIR, "datasets", "valid.parquet")

# --- FullModel (Conv + LSTM): конфиг модели ---
convo_constants = {
    "input_dim": 32,
    "linear1_dim": 64,
    "conv_window": 5,
    "conv_dim": 128,
    "lstm_hidden": 256,
    "lstm_num_layers": 2,
    "output_dim": 2,
}

# --- Обучение FullModel ---
WEIGHTS_DIR = os.path.join(ROOT, "weights")
EPOCHS = 20
LR = 1e-3
SAVE_NAME = "model.pt"

# --- Inference (сдача): папка и имена файлов ---
INFERENCE_DIR = os.path.join(ROOT, "inference")
INFERENCE_CONFIG_PATH = os.path.join(INFERENCE_DIR, "config.json")
INFERENCE_WEIGHTS_PATH = os.path.join(INFERENCE_DIR, "weights", "best_model.pt")

# --- Метрика Pearson ---
PRED_CLIP_LOW = -6.0
PRED_CLIP_HIGH = 6.0
WEIGHT_EPS = 1e-8
VAR_EPS = 1e-8

# --- Гиперпоиск FullModel ---
BEST_PARAMS_PATH = os.path.join(ROOT, "best_hyperparameters.json")
BEST_VAL_PEARSON_KEY = "best_val_pearson"
SAVES_DIR = os.path.join(ROOT, "saves")

# --- GRU-модель ---
gru_constants = {
    "input_dim": 32,
    "linear_dim": 64,
    "gru_hidden": 256,
    "gru_num_layers": 3,
    "output_dim": 2,
}
WEIGHTS_GRU_DIR = os.path.join(ROOT, "weights_gru")
EPOCHS_GRU = 12
LR_GRU = 1e-3
SAVE_NAME_GRU = "gru_model.pt"
BEST_PARAMS_PATH_GRU = os.path.join(ROOT, "best_hyperparameters_gru.json")
INFERENCE_GRU_DIR = os.path.join(ROOT, "inference_gru")
INFERENCE_GRU_CONFIG_PATH = os.path.join(INFERENCE_GRU_DIR, "config.json")
INFERENCE_GRU_WEIGHTS_PATH = os.path.join(INFERENCE_GRU_DIR, "weights", "best_model.pt")

# --- Stack (метамодель) ---
STACK_DIR = os.path.join(ROOT, "stack")
WEIGHTS_STACK_DIR = os.path.join(STACK_DIR, "weights_stack")
SAVE_NAME_STACK = "best_stack.pt"
STACK_CONFIG_PATH = os.path.join(STACK_DIR, "best_hyperparameters_stack.json")
DEFAULT_STACK_WEIGHTS_PATH = os.path.join(WEIGHTS_STACK_DIR, SAVE_NAME_STACK)
DEFAULT_META_HIDDEN_DIMS = [64]
EPOCHS_STACK = 20
LR_STACK = 1e-3
TRAIN_VAL_FRACTION = 0.1

# --- Устройство ---
from torch.cuda import is_available

DEVICE = "cuda" if is_available() else "cpu"
