convo_constants = {
    "input_dim": 32,
    "linear1_dim": 64,
    "conv_window": 5,
    "conv_dim": 128,
    "lstm_hidden": 256,
    "lstm_num_layers": 1,
    "output_dim": 2,
}

# Гиперпараметры для GRU-модели (model.gru_model.GRUModel)
gru_constants = {
    "input_dim": 32,
    "linear_dim": 64,
    "gru_hidden": 256,
    "gru_num_layers": 3,
    "output_dim": 2,
}

from torch.cuda import is_available

DEVICE = "cuda" if is_available() else "cpu"