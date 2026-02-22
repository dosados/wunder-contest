import torch


class _RingBuffer:
    """Внутренний класс кольцевого буфера."""
    def __init__(self, window, feat_dim, device='cpu'):
        self.window = window
        self.feat_dim = feat_dim
        self.device = device

        self.buffer = torch.zeros(window, feat_dim, device=device)
        self.pos = 0
        self.full = False

    def add(self, x):
        self.buffer[self.pos] = x
        self.pos = (self.pos + 1) % self.window
        if self.pos == 0:
            self.full = True

    def get_sequence(self):
        if not self.full:
            return self.buffer[:self.pos]
        return torch.cat(
            [self.buffer[self.pos:], self.buffer[:self.pos]],
            dim=0
        )

    def reset(self):
        self.buffer.zero_()
        self.pos = 0
        self.full = False

    def detach(self):
        """Отвязка буфера от графа вычислений."""
        self.buffer = self.buffer.detach().clone()


class ModelState:
    """Состояние для одного causal Conv1D слоя (кольцевой буфер)."""
    def __init__(self, conv_window, conv_feat_dim, device='cpu'):
        self.device = device
        self.conv = _RingBuffer(conv_window, conv_feat_dim, device)

    def add_conv(self, x):
        self.conv.add(x)

    def get_conv_sequence(self):
        return self.conv.get_sequence()

    def reset(self):
        self.conv.reset()

    def detach(self):
        self.conv.detach()
