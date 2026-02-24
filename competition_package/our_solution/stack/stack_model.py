"""
Метамодель: объединяет выходы FullModel и GRUModel через небольшой головной слой.
Интерфейс совместим с solution.PredictionModel: forward(x, seq_ix), reset_state().
Состояние обеих базовых моделей сбрасывается при смене seq_ix (передаётся в forward).
"""
import torch
import torch.nn as nn
from typing import List


def build_meta_head(input_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Module:
    """Строит голову: input_dim -> hidden_dims -> output_dim. Если hidden_dims пустой — один Linear."""
    if not hidden_dims:
        return nn.Linear(input_dim, output_dim)
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU(inplace=True))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class StackModel(nn.Module):
    """
    Объединяет выходы full_model и gru_model (каждый output_dim=2) в вектор длины 4,
    пропускает через meta_head (4 -> hidden_dims -> 2) и возвращает итоговый выход.
    Сброс состояния: при вызове forward(..., seq_ix=X) при смене X каждая базовая модель
    сбрасывает своё состояние (full_model и gru_model сами обрабатывают seq_ix).
    """

    def __init__(
        self,
        full_model: nn.Module,
        gru_model: nn.Module,
        meta_hidden_dims: List[int],
        output_dim: int = 2,
    ):
        super().__init__()
        self.full_model = full_model
        self.gru_model = gru_model
        self.output_dim = output_dim
        meta_input_dim = output_dim * 2  # 4
        self.meta_head = build_meta_head(
            input_dim=meta_input_dim,
            output_dim=output_dim,
            hidden_dims=list(meta_hidden_dims),
        )

    def reset_state(self) -> None:
        """Сброс состояния обеих базовых моделей (при смене последовательности)."""
        if hasattr(self.full_model, "reset_state"):
            self.full_model.reset_state()
        if hasattr(self.gru_model, "reset_state"):
            self.gru_model.reset_state()

    def forward(self, x: torch.Tensor, seq_ix: int | None = None) -> torch.Tensor:
        """
        Один или несколько шагов (streaming). Состояние базовых моделей обновляется внутри них.
        x: (D,) один шаг или (B, D) — B шагов подряд.
        seq_ix: при смене значения каждая базовая модель сбрасывает состояние.
        Возвращает (output_dim,) или (B, output_dim).
        """
        out_full = self.full_model(x, seq_ix=seq_ix)   # (2,) or (B, 2)
        out_gru = self.gru_model(x, seq_ix=seq_ix)     # (2,) or (B, 2)
        combined = torch.cat([out_full, out_gru], dim=-1)  # (4,) or (B, 4)
        return self.meta_head(combined)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Целая последовательность для обучения (без сохранения состояния).
        x: (B, T, D) или (T, D). Возвращает (B, T, output_dim) или (T, output_dim).
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        out_full = self.full_model.forward_sequence(x)   # (B, T, 2)
        out_gru = self.gru_model.forward_sequence(x)     # (B, T, 2)
        combined = torch.cat([out_full, out_gru], dim=-1)  # (B, T, 4)
        B, T, _ = combined.shape
        out = self.meta_head(combined.reshape(-1, combined.size(-1)))
        out = out.reshape(B, T, self.output_dim)
        if squeeze:
            out = out.squeeze(0)
        return out
