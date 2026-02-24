"""
Модуль stack: метамодель, объединяющая выходы FullModel и GRUModel через голову (meta_head).
Обучение мета-головы на валидации при замороженных базовых моделях.
"""

from .stack_model import StackModel, build_meta_head

__all__ = ["StackModel", "build_meta_head"]
