convo_constants = {
    "input_dim": 32,
    "linear1_dim": 64,
    "conv1_window": 3,
    "conv2_window": 128,
    "conv3_window": 256,
    "conv1_dim": 128,
    "conv2_dim": 128,
    "conv3_dim": 128,
    "linear2_dim": 64,
    "output_dim": 2,
}

from torch.cuda import is_available

DEVICE = "cuda" if is_available() else "cpu"