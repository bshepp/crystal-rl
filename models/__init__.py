"""Surrogate models for fast property prediction."""

from models.surrogate import (
    MultiTaskMLP,
    MultiTaskSurrogatePredictor,
    SurrogateMLP,
    SurrogatePredictor,
)

__all__ = [
    "SurrogateMLP",
    "SurrogatePredictor",
    "MultiTaskMLP",
    "MultiTaskSurrogatePredictor",
]
