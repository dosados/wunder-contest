import torch
from model import FullModel

DEVICE = "cuda"
model = FullModel(32, 128, 100, 100, 8, 4).to(DEVICE)
x = torch.randn(2, 100, 32, device=DEVICE)  # (batch, time, features)
y = model(x)
print("OK", y.shape)