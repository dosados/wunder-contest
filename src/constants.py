import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
FEATURE_COLUMNS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)
TARGET_COLUMNS = ["t0", "t1"]
INPUT_DIM = len(FEATURE_COLUMNS)
TARGET_DIM = len(TARGET_COLUMNS)
WARMUP_STEPS = 99
SEQUENCE_BATCH_SIZE = 16
SEQ_LEN = 1000
MIN_BATCH_MULTIPLE = 1000
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
TRAIN_PATH = os.path.join(DATASETS_DIR, "train.parquet")
VAL_PATH = os.path.join(DATASETS_DIR, "valid.parquet")
convo_constants = {
    "input_dim": 32,
    "linear1_dim": 64,
    "conv_window": 5,
    "conv_dim": 128,
    "lstm_hidden": 256,
    "lstm_num_layers": 2,
    "output_dim": 2,
}
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
WEIGHTS_DIR = os.path.join(ARTIFACTS_ROOT, "weights")
EPOCHS = 20
LR = 0.001
SAVE_NAME = "model.pt"
INFERENCE_DIR = os.path.join(ARTIFACTS_ROOT, "inference")
INFERENCE_CONFIG_PATH = os.path.join(INFERENCE_DIR, "config.json")
INFERENCE_WEIGHTS_PATH = os.path.join(INFERENCE_DIR, "weights", "best_model.pt")
LSTM_RUN_DIR = os.path.join(INFERENCE_DIR, "weights", "run_280226_235333")
PRED_CLIP_LOW = -6.0
PRED_CLIP_HIGH = 6.0
WEIGHT_EPS = 1e-08
VAR_EPS = 1e-08
gru_constants = {
    "input_dim": 32,
    "linear_dim": 64,
    "gru_hidden": 256,
    "gru_num_layers": 3,
    "output_dim": 2,
}
WEIGHTS_GRU_DIR = os.path.join(ARTIFACTS_ROOT, "weights_gru")
EPOCHS_GRU = 12
LR_GRU = 0.001
SAVE_NAME_GRU = "gru_model.pt"
INFERENCE_GRU_DIR = os.path.join(ARTIFACTS_ROOT, "inference_gru")
INFERENCE_GRU_CONFIG_PATH = os.path.join(INFERENCE_GRU_DIR, "config.json")
INFERENCE_GRU_WEIGHTS_PATH = os.path.join(INFERENCE_GRU_DIR, "weights", "best_model.pt")
STACK_DIR = os.path.join(ARTIFACTS_ROOT, "stack")
DEFAULT_META_HIDDEN_DIMS = [64]
LR_STACK = 0.001
ORCHESTRATION_VARIANTS = ("lstm_gru", "lstm_ssm")
DEFAULT_ORCHESTRATION_VARIANT = "lstm_ssm"
STACK_LSTM_GRU_META_PATH = os.path.join(STACK_DIR, "lstm_gru", "meta_head_oof.pt")
STACK_LSTM_SSM_META_PATH = os.path.join(STACK_DIR, "lstm_ssm", "meta_head_oof.pt")
ssm_constants = {
    "input_dim": INPUT_DIM,
    "d_model": 32,
    "n_layers": 2,
    "d_state": 8,
    "d_conv": 4,
    "expand": 1,
    "dt_rank": None,
    "output_dim": TARGET_DIM,
}
transformer_constants = {
    "input_dim": INPUT_DIM,
    "d_model": 64,
    "context_window": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.0,
    "output_dim": TARGET_DIM,
}
WEIGHTS_TRANSFORMER_DIR = os.path.join(ARTIFACTS_ROOT, "weights_transformer")
SAVE_NAME_TRANSFORMER = "transformer_model.pt"
EPOCHS_TRANSFORMER = 20
LR_TRANSFORMER = 0.001
WEIGHTS_SSM_DIR = os.path.join(ARTIFACTS_ROOT, "weights_ssm")
SAVE_NAME_SSM = "ssm_model.pt"
TRAIN_RESIDUAL_PATH = os.path.join(DATASETS_DIR, "train_residual.parquet")
VALID_RESIDUAL_PATH = os.path.join(DATASETS_DIR, "valid_residual.parquet")
RESIDUAL_METADATA_PATH = os.path.join(DATASETS_DIR, "residual_metadata.json")
from torch.cuda import is_available

DEVICE = "cuda" if is_available() else "cpu"
