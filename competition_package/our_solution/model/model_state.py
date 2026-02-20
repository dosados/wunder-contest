import torch

class ModelState:
    def __init__(self, input_dim, conv_window, trans_window, conv_feat_dim=None, device='cpu'):
        """
        input_dim     : размерность входного состояния рынка (D)
        conv_window   : длина буфера для Conv1D
        trans_window  : длина буфера после Conv, для Transformer
        conv_feat_dim : размерность выхода Conv (d_model); если None, берётся input_dim
        device        : 'cpu' или 'cuda'
        """
        self.device = device
        self.conv_window = conv_window
        self.trans_window = trans_window
        self.conv_feat_dim = conv_feat_dim if conv_feat_dim is not None else input_dim

        # Буфер для сырых состояний (Conv)
        self.raw_buffer = torch.zeros(conv_window, input_dim, device=device)
        self.raw_pos = 0
        self.raw_full = False

        # Буфер для Conv-фичей (Transformer), размерность d_model
        self.conv_buffer = torch.zeros(trans_window, self.conv_feat_dim, device=device)
        self.conv_pos = 0
        self.conv_full = False

    def add_raw(self, x):
        """Добавляем новое состояние в raw_buffer"""
        self.raw_buffer[self.raw_pos] = x
        self.raw_pos = (self.raw_pos + 1) % self.conv_window
        if self.raw_pos == 0:
            self.raw_full = True

    def get_raw_sequence(self):
        """Возвращает последовательность из raw_buffer в правильном порядке"""
        if not self.raw_full:
            return self.raw_buffer[:self.raw_pos]
        return torch.cat([self.raw_buffer[self.raw_pos:], self.raw_buffer[:self.raw_pos]], dim=0)

    def add_conv(self, conv_feat):
        """Добавляем Conv-фичу в conv_buffer"""
        self.conv_buffer[self.conv_pos] = conv_feat
        self.conv_pos = (self.conv_pos + 1) % self.trans_window
        if self.conv_pos == 0:
            self.conv_full = True

    def get_conv_sequence(self):
        """Возвращает последовательность Conv-фичей для Transformer"""
        if not self.conv_full:
            return self.conv_buffer[:self.conv_pos]
        return torch.cat([self.conv_buffer[self.conv_pos:], self.conv_buffer[:self.conv_pos]], dim=0)

    def reset(self):
        """Сбрасываем все буферы (при смене sequence)"""
        self.raw_buffer.zero_()
        self.raw_pos = 0
        self.raw_full = False

        self.conv_buffer.zero_()
        self.conv_pos = 0
        self.conv_full = False