__all__ = ["_PREDICTOR_TYPE", "_TABLE_NAME", "_CONTRIBUTION_TYPE", "_COL", "ContextInfo"]

from enum import Enum
from typing import TypedDict

class _PREDICTOR_TYPE(Enum):
    NUMERIC = "NUMERIC"
    SYMBOLIC = "SYMBOLIC"

class _TABLE_NAME(Enum):
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    NUMERIC_OVERALL = "numeric_overall"
    SYMBOLIC_OVERALL = "symbolic_overall"
    CREATE = "create"
    
# can also be sort order
class _CONTRIBUTION_TYPE(Enum):
    CONTRIBUTION = "contribution"
    CONTRIBUTION_ABS = "|contribution|"
    CONTRIBUTION_WEIGHTED = "contribution_weighted"
    CONTRIBUTION_WEIGHTED_ABS = "|contribution_weighted|"
    FREQUENCY = "frequency"
    CONTRIBUTION_MIN = "contribution_min"
    CONTRIBUTION_MAX = "contribution_max"

class _COL(Enum):
    PARTITON = "partition"
    PREDICTOR_NAME = "predictor_name"
    PREDICTOR_TYPE = "predictor_type"
    BIN_CONTENTS = "bin_contents"
    BIN_ORDER = "bin_order"
    CONTRIBUTION = "contribution"
    CONTRIBUTION_ABS = "contribution_abs"
    CONTRIBUTION_MIN = "contribution_min"
    CONTRIBUTION_MAX = "contribution_max"
    CONTRIBUTION_WEIGHTED = "contribution_weighted"
    CONTRIBUTION_WEIGHTED_ABS = "contribution_weighted_abs"
    FREQUENCY = "frequency"

class _SPECIAL(Enum):
    REMAINING = "remaining"
    TOTAL_FREQUENCY = "total_frequency"
    MISSING = "missing"

ContextInfo = TypedDict("ContextInfo", {"context_key": str, "context_value": str})
