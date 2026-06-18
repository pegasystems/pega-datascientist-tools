from __future__ import annotations

from enum import Enum
from typing import Literal, TypeAlias


class AdmModelType(Enum):
    """Supported ADM model techniques for champion/challenger operations."""

    GRADIENT_BOOSTING = "Gradient_boost"
    NAIVE_BAYES = "Naive_bayes"


NotificationCategory: TypeAlias = Literal[
    "All",
    "Responses",
    "Performance",
    "Model approval",
    "Output",
    "Predictors",
    "Prediction deployment",
    "Generic",
]
