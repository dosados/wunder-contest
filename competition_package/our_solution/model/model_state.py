import torch


class _RingBuffer:
    """Внутренний класс кольцевого буфера"""
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
        """Отвязка буфера от графа вычислений (для переноса состояния между батчами)."""
        self.buffer = self.buffer.detach().clone()


class ModelState:
    def __init__(
        self,
        conv1_window,
        conv2_window,
        conv3_window,
        conv1_feat_dim,
        conv2_feat_dim,
        conv3_feat_dim,
        device='cpu'
    ):
        """
        conv{i}_window   : длина окна контекста для i-го Conv слоя
        conv{i}_feat_dim : размерность признаков i-го Conv слоя
        device           : 'cpu' или 'cuda'
        """

        self.device = device

        # Три независимых буфера
        self.conv1 = _RingBuffer(conv1_window, conv1_feat_dim, device)
        self.conv2 = _RingBuffer(conv2_window, conv2_feat_dim, device)
        self.conv3 = _RingBuffer(conv3_window, conv3_feat_dim, device)

    # ===== Conv1 =====
    def add_conv1(self, x):
        self.conv1.add(x)

    def get_conv1_sequence(self):
        return self.conv1.get_sequence()

    # ===== Conv2 =====
    def add_conv2(self, x):
        self.conv2.add(x)

    def get_conv2_sequence(self):
        return self.conv2.get_sequence()

    # ===== Conv3 =====
    def add_conv3(self, x):
        self.conv3.add(x)

    def get_conv3_sequence(self):
        return self.conv3.get_sequence()

    # ===== Reset =====
    def reset(self):
        self.conv1.reset()
        self.conv2.reset()
        self.conv3.reset()

    def detach(self):
        """Отвязка состояния от графа (TBPTT: контекст переносится, градиент через границу батча не идёт)."""
        self.conv1.detach()
        self.conv2.detach()
        self.conv3.detach()