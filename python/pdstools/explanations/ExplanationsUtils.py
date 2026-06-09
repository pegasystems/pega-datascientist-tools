from __future__ import annotations
import os

__all__ = [
    "_COL",
    "_CONTRIBUTION_TYPE",
    "_PREDICTOR_TYPE",
    "_SPECIAL",
    "_TABLE_NAME",
    "ContextInfo",
    "ContextOperations",
    "ContributionType",
    "DisplayBy",
    "SortBy",
]

import json
import logging
from pathlib import Path
from enum import Enum
from typing import ClassVar, Literal, TYPE_CHECKING, TypedDict, cast

import polars as pl

from ..utils.namespaces import LazyNamespace

logger = logging.getLogger(__name__)


def validate(top_n: int | None = None, top_k: int | None = None) -> None:
    """Validate the parameters for top_n and top_k."""
    if top_n:
        if not isinstance(top_n, int) or top_n <= 1:
            raise ValueError(
                f"Invalid top_n value: {top_n}. Must be a positive integer greater than zero.",
            )
    if top_k:
        if not isinstance(top_k, int) or top_k <= 1:
            raise ValueError(
                f"Invalid top_k value: {top_k}. Must be a positive integer greater than zero.",
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
    MODEL_CONTEXTS = "model_contexts"


# can also be sort order
class _CONTRIBUTION_TYPE(Enum):
    def __new__(cls, default, alt, text):
        obj = object.__new__(cls)
        obj._value_ = default
        obj.alt = alt
        obj.text = text
        return obj

    def __init__(self, default, alt, text):
        self.alt = alt
        self.text = text

    @classmethod
    def validate_and_get_type(cls, val):
        """Get the accepted contribution type which is validated against user input"""
        for member in cls:
            if val == member.value:
                return member
        err = f"Invalid contribution type: {val} \nAccepted types are: {[x.value for x in cls]}"
        raise ValueError(err)

    CONTRIBUTION = ("contribution", "contribution", "average contribution")
    CONTRIBUTION_ABS = (
        "contribution_abs",
        "|contribution|",
        "absolute average contribution",
    )
    CONTRIBUTION_WEIGHTED = (
        "contribution_weighted",
        "contribution weighted",
        "weighted average contribution",
    )
    CONTRIBUTION_WEIGHTED_ABS = (
        "contribution_weighted_abs",
        "|contribution weighted|",
        "absolute weighted average contribution",
    )
    FREQUENCY = ("frequency", "frequency", "frequency")
    CONTRIBUTION_MIN = ("contribution_min", "contribution min", "minimum contribution")
    CONTRIBUTION_MAX = ("contribution_max", "contribution max", "maximum contribution")


class _COL(Enum):
    PARTITION = "partition"
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


def _resolve_contribution_type(value: "ContributionType") -> _CONTRIBUTION_TYPE:
    """Validate ``sort_by`` / ``display_by`` and return the matching enum member."""
    return _CONTRIBUTION_TYPE.validate_and_get_type(value)


# Public type aliases used in keyword-only signatures across Aggregate, Plots,
# and Reports. Kept as `Literal[...]` so IDE tooltips show valid choices and
# static type-checkers can flag typos at call sites.
SortBy = Literal[
    "contribution",
    "contribution_abs",
    "contribution_weighted",
    "contribution_weighted_abs",
    "frequency",
    "contribution_min",
    "contribution_max",
]
DisplayBy = SortBy
ContributionType = SortBy


class ContextInfo(TypedDict):
    """Context info."""

    context_key: str
    context_value: str


if TYPE_CHECKING:
    from .Aggregate import Aggregate


class ContextOperations(LazyNamespace):
    """Context related operations such as to filter unique contexts.

    Parameters
    ----------
    aggregate : Aggregate
        The aggregate object to operate on.

    """

    dependencies: ClassVar[list[str]] = ["polars"]
    dependency_group = "explanations"
    aggregate: "Aggregate"
    """Parent aggregate object used to load contextual data."""

    initialized: bool
    """Whether the contextual data cache has been loaded."""

    def __init__(self, aggregate: "Aggregate"):
        self.aggregate = aggregate

        self._df: pl.DataFrame | None = None
        self._context_keys: list[str] | None = None
        self.initialized = False

        self.file_batch_limit = int(os.getenv("FILE_BATCH_LIMIT", "100"))
        self.unique_contexts_file = self.aggregate.data_folderpath / "unique_contexts.json"

        super().__init__()

    def _load(self) -> None:
        if self.initialized:
            return

        if self._df is None:
            self._df = pl.from_dicts(
                [
                    {**json.loads(ck)[_COL.PARTITION.value], _COL.PARTITION.value: ck}
                    for ck in self.aggregate.get_df_contextual()
                    .select(_COL.PARTITION.value)
                    .unique()
                    .collect()
                    .to_series()
                    .to_list()
                ],
            )
        if self._context_keys is None:
            self._context_keys = list(self._df.select(pl.col("^py.*$")).columns)

        self.initialized = True

    def get_context_keys(self) -> list[str]:
        """Return the contextual key columns available in the loaded data.

        Returns
        -------
        list[str]
            Context key names such as ``["pyChannel", "pyDirection", ...]``.
        """
        self._load()
        assert self._context_keys is not None
        return self._context_keys

    def get_df(
        self,
        context_infos: list[ContextInfo] | None = None,
        with_partition_col: bool = False,
    ) -> pl.DataFrame:
        """Return unique contexts as a DataFrame.

        Parameters
        ----------
        context_infos : list[ContextInfo] | None, default None
            Optional context filters. When omitted, returns all unique contexts.
        with_partition_col : bool, default False
            Whether to include the raw partition column alongside the expanded
            context keys.

        Returns
        -------
        pl.DataFrame
            A DataFrame of unique contexts. When ``with_partition_col`` is
            ``True``, the partition column is retained in the result.
        """
        self._load()
        assert self._df is not None
        df = self._df if with_partition_col else self._get_clean_df(self._df)

        if context_infos is None or len(context_infos) == 0:
            return df

        return self._filter_df_by_context_infos(df, context_infos)

    def get_list(
        self,
        context_infos: list[ContextInfo] | None = None,
        with_partition_col: bool = False,
    ) -> list[ContextInfo]:
        """Return unique contexts as dictionaries.

        Parameters
        ----------
        context_infos : list[ContextInfo] | None, default None
            Optional context filters. When omitted, returns all unique contexts.
        with_partition_col : bool, default False
            Whether to include the raw partition column in each returned item.

        Returns
        -------
        list[ContextInfo]
            Unique context dictionaries, optionally filtered by the provided
            context information.
        """
        self._load()
        df = self.get_df(context_infos, with_partition_col)
        return cast(
            "list[ContextInfo]",
            df.unique().to_dicts(),
        )

    def create_unique_contexts_file(self) -> dict[str, list[str]]:
        """Create and persist the flat unique-context batch mapping if absent."""
        if self.unique_contexts_file.exists():
            return cast("dict[str, list[str]]", json.loads(self.unique_contexts_file.read_text()))

        list_of_contexts = (
            self.aggregate.get_df_contextual().select(_COL.PARTITION.value).unique().collect().to_series().to_list()
        )
        dict_of_contexts = self._create_context_batches(list_of_contexts)

        with self.unique_contexts_file.open("w", encoding="utf-8") as file:
            json.dump(dict_of_contexts, file)

        return cast("dict[str, list[str]]", json.loads(self.unique_contexts_file.read_text()))

    def create_batch_parquet_files(self, contexts_by_batch: dict[str | int, list[str]]) -> None:
        """Create one batch parquet file per context batch in a separate batches/ subdirectory."""
        batch_dir = Path(self.aggregate.data_folderpath) / "batches"
        batch_dir.mkdir(exist_ok=True)

        for batch_key, contexts in contexts_by_batch.items():
            batch_df = self.aggregate.get_df_contextual().filter(pl.col(_COL.PARTITION.value).is_in(contexts)).collect()
            batch_file_path = batch_dir / f"BATCH_{batch_key}.parquet"
            batch_df.write_parquet(batch_file_path)
            logger.info("Created batch file: %s with %d rows", batch_file_path, len(batch_df))

    def _filter_df_by_context_infos(
        self,
        df: pl.DataFrame,
        context_infos: list[ContextInfo],
    ) -> pl.DataFrame:
        ret_df = pl.DataFrame()

        filter_expressions = self._get_filter_expression(context_infos)
        for expression in filter_expressions:
            ret_df = pl.concat([ret_df, df.filter(expression)])
        return ret_df

    @staticmethod
    def _get_filter_expression(
        context_infos: list[ContextInfo],
    ) -> list[list[pl.Expr]]:
        expressions: list[list[pl.Expr]] = []
        for context_info in context_infos:
            expr: list[pl.Expr] = [
                pl.col(column_name).eq(column_value) for column_name, column_value in context_info.items()
            ]
            expressions.append(expr)
        return expressions

    @staticmethod
    def _get_clean_df(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.exclude(_COL.PARTITION.value))

    @staticmethod
    def get_context_info_str(context_info: ContextInfo, sep: str = "-") -> str:
        """Format a context dictionary as a single string.

        Parameters
        ----------
        context_info : ContextInfo
            Context values to format.
        sep : str, default "-"
            Separator inserted between context values.

        Returns
        -------
        str
            A compact context string such as ``channel1-direction1-...``.
        """
        return sep.join(f"{value}".strip() for value in context_info.values())

    def _create_context_batches(self, all_contexts: list[str]) -> dict[int, list[str]]:
        batch_size = self.file_batch_limit
        return {
            batch_idx: all_contexts[idx : idx + batch_size]
            for batch_idx, idx in enumerate(range(0, len(all_contexts), batch_size))
        }
