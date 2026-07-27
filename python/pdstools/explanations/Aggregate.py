from __future__ import annotations

__all__ = ["Aggregate"]

import logging
from typing import ClassVar, TYPE_CHECKING, cast, overload
from pathlib import Path

import polars as pl

from ..pega_io import scan_parquet_path
from ..utils.namespaces import LazyNamespace
from ._constants import (
    MISSING,
    NUMERIC,
    REMAINING,
    SYMBOLIC,
    TOTAL_FREQUENCY,
    ContributionType,
    validate_contribution_type,
)
from .ContextOperations import ContextOperations

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .Explanations import Explanations


class Aggregate(LazyNamespace):
    """Aggregate."""

    dependencies: ClassVar[list[str]] = ["polars"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations
        self.data_folderpath = Path(explanations.root_dir) / explanations.data_folder
        self.data_pattern = None
        self.df_contextual: pl.LazyFrame | None = None
        self.df_overall: pl.LazyFrame | None = None
        self.context_operations = ContextOperations(aggregate=self)
        self.initialized = False
        super().__init__()

    def get_df_contextual(self) -> pl.LazyFrame:
        """Get the contextual dataframe, loading it if not already loaded."""
        self._load_data()
        assert self.df_contextual is not None
        return self.df_contextual

    def get_df_overall(self) -> pl.LazyFrame:
        """Get the overall dataframe, loading it if not already loaded."""
        self._load_data()
        assert self.df_overall is not None
        return self.df_overall

    def get_predictor_contributions(
        self,
        context: dict[str, str] | None = None,
        top_n: int = 20,
        *,
        sort_by: ContributionType = "contribution_abs",
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
    ) -> pl.DataFrame:
        """Get the top-n predictor contributions for a given context or overall.

        Parameters
        ----------
        context : dict[str, str] | None
            The context to filter contributions by.
            If None, contributions for all contexts will be returned.
        top_n : int
            Number of top predictors.
        sort_by : str
            Column to rank/select top predictors. One of
            ``contribution``, ``contribution_abs``,
            ``contribution_weighted``, ``contribution_weighted_abs``.
            Default: ``"contribution_abs"``.
        descending : bool
            Sort most- or least-impactful first. Default: ``True``.
        missing : bool
            Include missing-value bins. Default: ``True``.
        remaining : bool
            Include an aggregated "remaining" row for predictors outside
            the top-n. Default: ``True``.
        include_numeric_single_bin : bool
            Include numeric predictors that have only a single bin.
            Default: ``False``.
        """
        validate_contribution_type(sort_by)

        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError(f"Invalid top_n value: {top_n}. Must be a positive integer.")

        self._load_data()

        return self._get_predictor_contributions(
            contexts=cast("list[dict[str, str]]", [context]) if context else None,
            limit=top_n,
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
        )

    def get_predictor_value_contributions(
        self,
        predictors: list[str],
        context: dict[str, str] | None = None,
        top_k: int = 20,
        *,
        sort_by: ContributionType = "contribution_abs",
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
    ) -> pl.DataFrame:
        """Get the top-k predictor value contributions for a given context or overall.

        Parameters
        ----------
        predictors : list[str]
            Required. list of predictors to get the contributions for.
        context : dict[str, str] | None
            The context to filter contributions by.
            If None, contributions for all contexts will be returned.
        top_k : int
            Number of unique categorical predictor values to return.
        sort_by : str
            Column to rank/select top predictors. One of
            ``contribution``, ``contribution_abs``,
            ``contribution_weighted``, ``contribution_weighted_abs``.
            Default: ``"contribution_abs"``.
        descending : bool
            Sort most- or least-impactful first. Default: ``True``.
        missing : bool
            Include missing-value bins. Default: ``True``.
        remaining : bool
            Include an aggregated "remaining" row for values outside
            the top-k. Default: ``True``.
        include_numeric_single_bin : bool
            Include numeric predictors that have only a single bin.
            Default: ``False``.
        """
        validate_contribution_type(sort_by)

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"Invalid top_k value: {top_k}. Must be a positive integer.")

        self._load_data()

        return self._get_predictor_value_contributions(
            contexts=cast("list[dict[str, str]]", [context]) if context else None,
            predictors=predictors,
            limit=top_k,
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
        )

    def get_unique_contexts_list(
        self,
        context_infos: list[dict[str, str]] | None = None,
        with_partition_col: bool = False,
    ) -> list[dict[str, str]]:
        """Get unique contexts list."""
        return self.context_operations.get_list(context_infos, with_partition_col)

    def _load_data(self):
        if self.initialized:
            return

        try:
            self.explanations.validate_data_folder()
        except FileNotFoundError as e:
            logger.error("Error validating aggregates folder: %s", e)
            raise

        selected_columns = [
            "context_partition",
            "contribution",
            "contribution_abs",
            "frequency",
            "predictor_type",
            "predictor_name",
            "bin_contents",
            "bin_order",
            "contribution_min",
            "contribution_max",
        ]

        context_ = self.data_folderpath / (self.data_pattern if self.data_pattern else "BY_CONTEXT.parquet")

        self.df_contextual = (
            scan_parquet_path(context_)
            .select(selected_columns)
            .filter(pl.col("contribution") != 0.0)
            .sort(by="predictor_name")
        )
        self.df_overall = (
            scan_parquet_path(self.data_folderpath / "OVERVIEW.parquet")
            .select(selected_columns)
            .filter(pl.col("contribution") != 0.0)
            .sort(by="predictor_name")
        )

        self.initialized = True

    def _get_predictor_contributions(
        self,
        contexts: list[dict[str, str]] | None = None,
        predictors: list[str] | None = None,
        limit: int = 20,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
        sort_by: str = "contribution_abs",
    ) -> pl.DataFrame:
        contexts = contexts or []
        predictors = predictors or []

        # if no contexts are provided, then we return the overall data
        # if contexts are provided, then we generate the context filters
        # and load the data for those contexts
        df = self._get_df(contexts)

        if not include_numeric_single_bin:
            df = self._filter_single_bin_numeric_predictors(df)

        # If predictors are specified we filter the dataframe for those predictors
        if len(predictors) > 0:
            df = self._filter_for_predictors(df, predictors)

        # If we do not want to include the missing predictor values, we filter them out
        if not missing:
            df = df.filter(pl.col("bin_contents") != MISSING)

        # Aggregate all the different types of contributions
        # note: total_frequency is computed per predictor so the weighted average
        # divides by that predictor's own bin frequencies, not the entire partition.
        df = self._calculate_aggregates(
            df,
            frequency_over=[
                "context_partition",
                "predictor_name",
                "predictor_type",
            ],
            aggregate_over=[
                "context_partition",
                "predictor_name",
                "predictor_type",
            ],
        )

        # Take the top predictors per partition, sorted by sort_by
        df_top_predictors = self._get_df_with_top_limit(
            df,
            sort_by=sort_by,
            over=["context_partition"],
            limit=limit,
            descending=descending,
        )

        # If we want to include the cumulative contribution of all predictors
        # outside of the top `limit`, we calculate the remaining contributions
        if remaining:
            # Calculate the remaining contributions by aggregating the
            # contributions of all predictors not in the top `limit`
            # We provide the top predictors as anti-join to calculate the remaining contributions
            df_remaining = self._calculate_remaining_aggregates(
                df,
                df_anti=df_top_predictors,
                anti_on=[
                    "context_partition",
                    "predictor_name",
                    "predictor_type",
                ],
                frequency_over=["context_partition"],
                aggregate_over=["context_partition"],
            )
            df_top_predictors = pl.concat(
                df.select(sorted(df.collect_schema().names())) for df in [df_remaining, df_top_predictors]
            )

        # Ensure all predictors are unique and sorted by sort_by. `unique` does not
        # preserve order, so the sort needs a total ordering: predictor names are
        # unique here and break ties in `sort_by` deterministically.
        df_out = df_top_predictors.unique()
        df_out = df_out.sort(
            by=[sort_by, "predictor_name"],
            descending=[descending, False],
        )

        return df_out.collect()

    def _get_predictor_value_contributions(
        self,
        contexts: list[dict[str, str]] | None = None,
        predictors: list[str] | None = None,
        limit: int = 20,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
        sort_by: str = "contribution_abs",
    ) -> pl.DataFrame:
        # if no contexts are provided, then we return the overall data
        # if contexts are provided, then we generate the context filters
        # and load the data for those contexts
        df = self._get_df(contexts)

        if not include_numeric_single_bin:
            df = self._filter_single_bin_numeric_predictors(df)

        # If predictors are specified we filter the dataframe for those predictors
        predictors = predictors or []
        if len(predictors) > 0:
            df = self._filter_for_predictors(df, predictors)

        # If we do not want to include the missing predictor values, we filter them out
        if not missing:
            df = df.filter(pl.col("bin_contents") != MISSING)

        # Aggregate all the different types of contributions
        # note: we need to aggregate frequency over partition to calculate weighted contributions
        df = self._calculate_aggregates(
            df,
            frequency_over=[
                "context_partition",
                "predictor_name",
                "predictor_type",
            ],
            aggregate_over=[
                "context_partition",
                "predictor_name",
                "predictor_type",
                "bin_order",
                "bin_contents",
            ],
        )

        # Append a sort column and value, note these are not used when
        # finding the top predictors, but are used for logically sorting the final output
        # e.g.:
        # - numeric predictors are sorted by bin order
        # - symbolic predictors are sorted by sort_by
        df = self._get_df_with_sort_info(df, sort_by=sort_by)

        # Take the top predictors per partition, sorted by sort_by
        df_top_predictor_values = self._get_df_with_top_limit(
            df,
            sort_by=sort_by,
            over=[
                "context_partition",
                "predictor_name",
                "predictor_type",
            ],
            limit=limit,
            descending=descending,
        )

        # If we want to force inclusion of the missing predictor values, concat with the top n
        if missing:
            df_missing = self._get_missing_predictor_values_df(df)
            df_top_predictor_values = pl.concat([df_top_predictor_values, df_missing])

        # If we want to include the cumulative contribution of
        # all predictor values outside of the top `limit`, we calculate the remaining contributions
        if remaining:
            df_remaining = self._calculate_remaining_aggregates(
                df_all=df,
                df_anti=df_top_predictor_values,
                anti_on=[
                    "context_partition",
                    "predictor_name",
                    "predictor_type",
                    "bin_order",
                    "bin_contents",
                ],
                frequency_over=[
                    "context_partition",
                    "predictor_name",
                    "predictor_type",
                ],
                aggregate_over=[
                    "context_partition",
                    "predictor_name",
                    "predictor_type",
                ],
            )

            # Add sort information and concat with the top predictor values
            df_remaining = self._get_df_with_sort_info(
                df_remaining,
                sort_by=sort_by,
            )
            df_top_predictor_values = pl.concat(
                df.select(sorted(df.collect_schema().names())) for df in [df_remaining, df_top_predictor_values]
            )

        # Ensure all predictor values are unique and sorted according to predictor type.
        # `unique` does not preserve order, so `bin_contents` is appended as a final
        # tiebreaker to keep the output deterministic when `sort_value` ties (the
        # forced MISSING bin and the 'remaining' rollup share a sort value).
        df_out = df_top_predictor_values.unique()
        df_out = df_out.sort(
            by=[
                *self._get_sort_over_columns(predictors=None),
                "sort_value",
                "bin_contents",
            ],
        )
        return df_out.collect()

    def _get_df_with_sort_info(
        self,
        df: pl.LazyFrame,
        sort_by: str = "contribution_abs",
    ) -> pl.LazyFrame:
        """Add a sort column and value to the dataframe based on the predictor type.
        # Sort logic:
        #  - numeric predictors are sorted by bin order
        #  - symbolic predictors are sorted by contribution type
        """
        return df.with_columns(
            pl.when(pl.col("predictor_type") == NUMERIC)
            .then(pl.lit("bin_order"))
            .otherwise(pl.lit(sort_by))
            .alias("sort_column"),
            pl.when(pl.col("predictor_type") == NUMERIC)
            .then(pl.col("bin_order"))
            .otherwise(pl.col(sort_by))
            .alias("sort_value"),
        )

    def _filter_for_predictors(
        self,
        df: pl.LazyFrame,
        predictors: list[str],
    ) -> pl.LazyFrame:
        return df.filter(pl.col("predictor_name").is_in(predictors))

    def _get_df_with_top_limit(
        self,
        df: pl.LazyFrame,
        over: list[str],
        sort_by: str = "contribution_abs",
        limit: int = 20,
        descending: bool = True,
    ) -> pl.LazyFrame:
        """Return the top `limit` rows per group, ranked by `sort_by`.

        For each unique combination of values in `over`, keeps only the `limit`
        rows with the highest (or lowest) value in `sort_by`.

        When `descending=True` (the default), the rows with the **largest**
        values are kept — i.e. the most impactful contributions rise to the top.
        When `descending=False`, the rows with the **smallest** values are kept
        instead, which is useful when selecting the least influential predictors.

        Note: Polars' `top_k_by` uses a `reverse` parameter whose semantics are
        the **opposite** of `descending`. `reverse=False` returns the k largest
        values, while `reverse=True` returns the k smallest. To keep the
        caller-facing API intuitive (`descending=True` → largest values), we
        pass `reverse=not descending` to Polars internally.
        """
        return df.select(
            pl.all()
            .top_k_by(sort_by, k=limit, reverse=not descending)
            .over(
                over,
                mapping_strategy="explode",
            ),
        )

    def _get_missing_predictor_values_df(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Return the rows holding the "missing" bin, which is keyed on bin contents."""
        return df.filter(
            pl.col("bin_contents") == MISSING,
        )

    def _get_df(
        self,
        contexts: list[dict[str, str]] | None = None,
    ):
        contexts = contexts or []

        if len(contexts) == 0:
            df = self._get_base_df()
        else:
            df_filtered_contexts = self.context_operations.get_df(contexts, True)
            df = self._get_base_df(df_filtered_contexts)
        return df

    def _get_base_df(
        self,
        df_filtered_contexts: pl.DataFrame | None = None,
    ) -> pl.LazyFrame:
        if self.df_overall is None or self.df_contextual is None:
            self._load_data()
        assert self.df_overall is not None
        assert self.df_contextual is not None

        if df_filtered_contexts is None:
            return self.df_overall
        return self.df_contextual.join(
            df_filtered_contexts.lazy(),
            on="context_partition",
            how="inner",
        )

    def _get_sort_over_columns(
        self,
        predictors: list[str] | None = None,
    ) -> list[str]:
        if predictors is None or len(predictors) == 0:
            return ["predictor_name", "context_partition"]
        return ["context_partition"]

    def _calculate_remaining_aggregates(
        self,
        df_all: pl.LazyFrame,
        df_anti: pl.LazyFrame,
        anti_on: list[str],
        frequency_over: list[str],
        aggregate_over: list[str],
    ) -> pl.LazyFrame:
        """Anti-join to isolate non-top rows, aggregate, and label as 'remaining'."""
        df_remaining = df_all.join(df_anti, on=anti_on, how="anti")
        df_remaining = self._calculate_aggregates(df_remaining, frequency_over, aggregate_over)
        return self._label_remaining(df_remaining, aggregate_over)

    @staticmethod
    def _label_remaining(df: pl.LazyFrame, aggregate_over: list[str]) -> pl.LazyFrame:
        """Add 'remaining' labels based on aggregation granularity."""
        if len(aggregate_over) == 1 and aggregate_over[0] == "context_partition":
            return df.with_columns(
                pl.lit(REMAINING).alias("predictor_name"),
                pl.lit(SYMBOLIC).alias("predictor_type"),
            )
        return df.with_columns(
            pl.lit(REMAINING).alias("bin_contents"),
            pl.lit(0).cast(pl.Int64).alias("bin_order"),
        )

    def _calculate_aggregates(
        self,
        df: pl.LazyFrame,
        frequency_over: list[str],
        aggregate_over: list[str],
    ) -> pl.LazyFrame:
        """Enrich with total_frequency at frequency_over level, then aggregate at aggregate_over level."""
        data = self._add_total_frequency_to_df(df, frequency_over)
        return self._agg_over_columns_in_df(data, aggregate_over)

    @staticmethod
    @overload
    def _add_total_frequency_to_df(df: pl.DataFrame, group_by: list[str]) -> pl.DataFrame: ...

    @staticmethod
    @overload
    def _add_total_frequency_to_df(df: pl.LazyFrame, group_by: list[str]) -> pl.LazyFrame: ...

    @staticmethod
    def _add_total_frequency_to_df(
        df: pl.DataFrame | pl.LazyFrame,
        group_by: list[str],
    ) -> pl.DataFrame | pl.LazyFrame:
        if isinstance(df, pl.DataFrame):
            grouped_df = df.group_by(group_by).agg(pl.sum("frequency").alias(TOTAL_FREQUENCY))
            return grouped_df.join(df, on=group_by, how="left")

        grouped_lf = df.group_by(group_by).agg(pl.sum("frequency").alias(TOTAL_FREQUENCY))
        return grouped_lf.join(df, on=group_by, how="left")

    def add_frequency_pct_to_df(self, df: pl.DataFrame, group_by: list[str]) -> pl.DataFrame:
        """Add a frequency percentage column to the dataframe based on the total frequency per group."""

        df_with_total_frequency = self._add_total_frequency_to_df(df, group_by)
        return df_with_total_frequency.with_columns(
            pl.when(pl.col(TOTAL_FREQUENCY) == 0)
            .then(0.0)
            # round(4) to preserve very small frequency shares (e.g. 0.02%)
            .otherwise((pl.col("frequency") / pl.col(TOTAL_FREQUENCY) * 100).round(4))
            .alias("frequency_pct")
        )

    def add_context_frequency_pct_to_df(
        self,
        df: pl.DataFrame,
        join_on: list[str],
    ) -> pl.DataFrame:
        """Add frequency_pct showing this context's share of the overall model.

        For each row, computes
        ``frequency_pct = df.frequency / overall_model_frequency * 100``
        where the overall model frequency is summed over ``join_on`` columns
        from the overall (non-contextual) dataset.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with a ``frequency`` column (context data).
        join_on : list[str]
            Columns to join on, typically
            ``[predictor_name, predictor_type]``.

        Returns
        -------
        pl.DataFrame
            *df* with an added ``frequency_pct`` column (0–100).
        """
        overall_freq = (
            self.get_df_overall().group_by(join_on).agg(pl.sum("frequency").alias("overall_total_frequency")).collect()
        )
        df_joined = df.join(overall_freq, on=join_on, how="left")
        return df_joined.with_columns(
            pl.when(pl.col("overall_total_frequency").is_null() | (pl.col("overall_total_frequency") == 0))
            .then(0.0)
            .otherwise((pl.col("frequency") / pl.col("overall_total_frequency") * 100).round(4))
            .alias("frequency_pct")
        )

    @staticmethod
    def _get_mean_aggregates():
        """Get mean contribution aggregates."""

        def _apply(col):
            return pl.col(col).mean().alias(col)

        return [
            _apply("contribution"),
            _apply("contribution_abs"),
        ]

    @staticmethod
    def _get_weighted_aggregates():
        """Get frequency-weighted contribution aggregates normalized by total frequency."""

        def _apply(col, alias):
            return ((pl.col(col) * pl.col("frequency")).sum() / pl.col(TOTAL_FREQUENCY).first()).alias(alias)

        return [
            _apply("contribution", "contribution_weighted"),
            _apply("contribution_abs", "contribution_weighted_abs"),
        ]

    @staticmethod
    def _get_frequency_aggregate():
        """Get frequency sum aggregate."""
        return [pl.col("frequency").sum().alias("frequency")]

    @staticmethod
    def _get_bounds_aggregates():
        """Get min and max contribution bounds."""
        return [
            pl.col("contribution_min").min().alias("contribution_min"),
            pl.col("contribution_max").max().alias("contribution_max"),
        ]

    def _agg_over_columns_in_df(self, df, group_by):
        """Aggregate contribution metrics over specified columns."""
        aggregate_by_list = [
            *self._get_mean_aggregates(),
            *self._get_weighted_aggregates(),
            *self._get_frequency_aggregate(),
            *self._get_bounds_aggregates(),
        ]
        return df.group_by(group_by).agg(aggregate_by_list)

    @staticmethod
    def _filter_single_bin_numeric_predictors(df: pl.LazyFrame) -> pl.LazyFrame:
        """Remove numeric predictors that have only a single non-missing bin."""
        single_bin_predictors = (
            df.filter((pl.col("predictor_type") == NUMERIC) & (pl.col("bin_contents") != MISSING))
            .group_by(["context_partition", "predictor_name"])
            .agg(pl.col("bin_order").n_unique().alias("bin_count"))
            .filter(pl.col("bin_count") <= 1)
            .select(["context_partition", "predictor_name"])
        )
        return df.join(
            single_bin_predictors,
            on=["context_partition", "predictor_name"],
            how="anti",
        )
