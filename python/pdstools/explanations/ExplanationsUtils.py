__all__ = [
    "_PREDICTOR_TYPE",
    "_TABLE_NAME",
    "_CONTRIBUTION_TYPE",
    "_COL",
    "_DEFAULT",
    "_SPECIAL",
    "ContextInfo",
    "ContextOperations",
]

import json
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, TypedDict, cast

import polars as pl

from ..utils.namespaces import LazyNamespace


def validate(top_n: Optional[int] = None, top_k: Optional[int] = None):
    """Validate the parameters for top_n and top_k."""
    if top_n:
        if not isinstance(top_n, int) or top_n <= 1:
            raise ValueError(
                f"Invalid top_n value: {top_n}. Must be a positive integer greater than zero."
            )
    if top_k:
        if not isinstance(top_k, int) or top_k <= 1:
            raise ValueError(
                f"Invalid top_k value: {top_k}. Must be a positive integer greater than zero."
            )


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
    def __new__(cls, default, alt):
        obj = object.__new__(cls)
        obj._value_ = default
        obj.alt = alt
        return obj

    def __init__(self, default, alt):
        self.alt = alt

    @classmethod
    def validate_and_get_type(cls, val):
        """get the accepted contribution type which is validated against user input"""

        for member in cls:
            if val == member.value:
                return member
        err = f"Invalid contribution type: {val} \nAccepted types are: {[x.value for x in cls]}"
        raise ValueError(err)

    CONTRIBUTION = ("contribution", "contribution")
    CONTRIBUTION_ABS = ("contribution_abs", "|contribution|")
    CONTRIBUTION_WEIGHTED = ("contribution_weighted", "contribution weighted")
    CONTRIBUTION_WEIGHTED_ABS = ("contribution_weighted_abs", "|contribution weighted|")
    FREQUENCY = ("frequency", "frequency")
    CONTRIBUTION_MIN = ("contribution_min", "contribution min")
    CONTRIBUTION_MAX = ("contribution_max", "contribution max")


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


class _DEFAULT(Enum):
    TOP_N = 10
    TOP_K = 10
    DESCENDING = True
    MISSING = True
    REMAINING = True


ContextInfo = TypedDict("ContextInfo", {"context_key": str, "context_value": str})

if TYPE_CHECKING:
    from .Aggregate import Aggregate


class ContextOperations(LazyNamespace):
    """Context related operations such as to filter unique contexts.
    Parameters:
        aggregate (Aggregate): The aggregate object to operate on.

    Attributes:
        aggregate (Aggregate): The aggregate object.
        _df (Optional[pl.DataFrame]): DataFrame containing context information.
        _context_keys (Optional[List[str]]): List of context keys.
        initialized (bool): Flag indicating if the context operations have been initialized.

    Methods:
        get_context_keys():
            Returns the list of context keys from loaded data.
            Eg. ['pyChannel', 'pyDirection', ...]

        get_df(context_infos=None, with_partition_col=False):
            Returns a DataFrame containing unique contexts
            If `with_partition_col` is True, includes the partition column.
            If `context_infos` is None, returns the full unique contexts,
            else filtered by the context
            Eg. with partition column:
            | pyChannel | pyDirection | ... | partition |
            |-----------|-------------|-----|-----------|
            | channel1  | direction1  | ... | {"partition": {"pyChannel": "channel1", "pyDirection": "direction1"}} |
            | channel1  | direction2  | ... | {"partition": {"pyChannel": "channel1", "pyDirection": "direction2"}} |

        get_list(context_infos=None, with_partition_col=False):
            Returns a List[ContextInfo] containing unique contexts
            If `with_partition_col` is True, includes the partition column.
            If `context_infos` is None, returns the full unique contexts,
            else filtered by the context
            Eg. without partition column:
            [
                {"pyChannel": "channel1", "pyDirection": "direction1", ...},
                {"pyChannel": "channel1", "pyDirection": "direction2", ...},
            ]

        get_context_info_str(context_info, sep="-"):
            Returns a string representation of a single context information.
            Eg. channel1-direction1-...

    """

    dependencies = ["polars"]
    dependency_group = "explanations"

    def __init__(self, aggregate: "Aggregate"):
        self.aggregate = aggregate

        self._df: Optional[pl.DataFrame] = None
        self._context_keys: Optional[List[str]] = None
        self.initialized = False

        super().__init__()

    def _load(self):
        if self.initialized:
            return

        if self._df is None:
            self._df = pl.from_dicts(
                [
                    {**json.loads(ck)[_COL.PARTITON.value], _COL.PARTITON.value: ck}
                    for ck in self.aggregate.get_df_contextual()
                    .select(_COL.PARTITON.value)
                    .unique()
                    .collect()
                    .to_series()
                    .to_list()
                ]
            )
        if self._context_keys is None:
            self._context_keys = list(self._df.select(pl.col("^py.*$")).columns)

        self.initialized = True

    def get_context_keys(self):
        self._load()
        return self._context_keys

    def get_df(
        self,
        context_infos: Optional[List[ContextInfo]] = None,
        with_partition_col: bool = False,
    ) -> pl.DataFrame:
        """Get the DataFrame filtered by the provided context information."""

        self._load()
        df = self._df if with_partition_col else self._get_clean_df(self._df)

        if context_infos is None or len(context_infos) == 0:
            return df

        return self._filter_df_by_context_infos(df, context_infos)

    def get_list(
        self,
        context_infos: Optional[List[ContextInfo]] = None,
        with_partition_col: bool = False,
    ) -> List[ContextInfo]:
        """Get the list of context information filtered by the provided context information."""

        self._load()
        df = self.get_df(context_infos, with_partition_col)
        return cast(
            list[ContextInfo],
            df.unique().to_dicts(),
        )

    def _filter_df_by_context_infos(self, df, context_infos):
        ret_df = pl.DataFrame()

        filter_expressions = self._get_filter_expression(context_infos)
        for expression in filter_expressions:
            ret_df = pl.concat([ret_df, df.filter(expression)])
        return ret_df

    @staticmethod
    def _get_filter_expression(context_infos):
        expressions = []
        for context_info in context_infos:
            expr = [
                pl.col(column_name) == column_value
                for column_name, column_value in context_info.items()
            ]
            expressions.append(expr)
        return expressions

    @staticmethod
    def _get_clean_df(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.exclude(_COL.PARTITON.value))

    @staticmethod
    def get_context_info_str(context_info: ContextInfo, sep: str = "-") -> str:
        return sep.join(f"{value}".strip() for value in context_info.values())
