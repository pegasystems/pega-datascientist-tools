__all__ = [
    "_PREDICTOR_TYPE",
    "_TABLE_NAME",
    "_CONTRIBUTION_TYPE",
    "_COL",
    "ContextInfo",
    "ContextOperations",
]

from enum import Enum
from typing import TypedDict, List, Optional, cast

import polars as pl
import json


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


ContextInfo = TypedDict("ContextInfo", {"context_key": str, "context_value": str})


class ContextOperations:
    def __init__(self, df: pl.LazyFrame):
        self._df = pl.from_dicts(
            [
                {**json.loads(ck)[_COL.PARTITON.value], _COL.PARTITON.value: ck}
                for ck in df.select(_COL.PARTITON.value)
                .unique()
                .collect()
                .to_series()
                .to_list()
            ]
        )

        self._context_keys = list(self._df.select(pl.col("^py.*$")).columns)

    def get_context_keys(self):
        return self._context_keys

    def get_unique_values_in_each_key(self):
        df = self.get_df()
        ret = {}
        for key in self._context_keys:
            ret[key] = df.select(pl.col(key).unique()).to_dict()
        return ret

    def get_df(
        self,
        context_infos: Optional[List[ContextInfo]] = None,
        with_partition_col: bool = False,
    ) -> pl.DataFrame:
        """Get the DataFrame filtered by the provided context information."""

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
    def _get_clean_df(df):
        return df.select(pl.exclude(_COL.PARTITON.value))

    @staticmethod
    def get_context_info_str(context_info: ContextInfo, sep: str = "-") -> str:
        return sep.join(f"{value}".strip() for value in context_info.values())
