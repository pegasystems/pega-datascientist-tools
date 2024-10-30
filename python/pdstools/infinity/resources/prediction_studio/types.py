from enum import Enum
from typing import TypeVar, Literal


class AdmModelType(Enum):
    GRADIENT_BOOSTING = "Gradient_boost"
    NAIVE_BAYES = "Naive_bayes"


NotificationCategory = TypeVar(
    "NotificationCategory",
    Literal[
        "All",
        "Responses",
        "Performance",
        "Model approval",
        "Output",
        "Predictors",
        "Prediction deployment",
        "Generic",
    ],
    str,
)
