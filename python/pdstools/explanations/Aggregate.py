__all__ = ["Aggregate"]

import logging
import pathlib
from typing import TYPE_CHECKING, List, Optional

import polars as pl

from ..utils.namespaces import LazyNamespace
from .ExplanationsUtils import (
    _COL,
    _CONTRIBUTION_TYPE,
    _PREDICTOR_TYPE,
    _SPECIAL,
    defaults,
    ContextInfo,
    ContextOperations,
    validate,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .Explanations import Explanations


class Aggregate(LazyNamespace):
    dependencies = ["polars"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations
        self.data_folderpath = explanations.preprocess.data_folderpath
        self.data_pattern = None
        self.df_contextual = None
        self.df_overall = None
        self.context_operations = ContextOperations(aggregate=self)
        self.initialized = False
        super().__init__()

    def get_df_contextual(self) -> pl.LazyFrame:
        """Get the contextual dataframe, loading it if not already loaded."""
        self._load_data()
        return self.df_contextual
    
    def get_df_overall(self) -> pl.LazyFrame:
        """Get the overall dataframe, loading it if not already loaded."""
        self._load_data()
        return self.df_overall

    def get_predictor_contributions(
        self,
        context: Optional[dict[str, str]] = None,
        top_n: int = defaults.TOP_N,
        descending: bool = defaults.DESCENDING,
        missing: bool = defaults.MISSING,
        remaining: bool = defaults.REMAINING,
        sort_by: str = defaults.SORT_BY.value,
    ):
        """Get the top-n predictor contributions for a given context or overall.

        Args:
            context (Optional[dict[str, str]]):
                The context to filter contributions by.
                If None, contributions for all contexts will be returned.
            top_n (int):
                Number of top predictors
            descending (bool):
                Whether to sort contributions in descending order.
            missing (bool):
                Whether to include contributions for missing predictor values.
            remaining (bool):
                Whether to include contributions for remaining predictors outside the top-n.
            sort_by (str):
                Method to sort/select top contributions. Options include
                `contribution`, `contribution_abs`, `contribution_weighted`.
                Default is `contribution_abs` which sorts by absolute average contributions.
        """

        validated_sort_by = _CONTRIBUTION_TYPE.validate_and_get_type(
            sort_by
        )

        try:
            validate(top_n=top_n)
        except ValueError as e:
            logger.error("Invalid parameters: %s", e)
            raise

        self._load_data()

        return self._get_predictor_contributions(
            contexts=[context] if context else None,
            limit=top_n,
            descending=descending,
            missing=missing,
            remaining=remaining,
            sort_by=validated_sort_by.value,
        )

    def get_predictor_value_contributions(
        self,
        predictors: List[str],
        context: Optional[dict[str, str]] = None,
        top_k: int = defaults.TOP_K,
        descending: bool = defaults.DESCENDING,
        missing: bool = defaults.MISSING,
        remaining: bool = defaults.REMAINING,
        sort_by: str = defaults.SORT_BY.value,
    ):
        """Get the top-k predictor value contributions for a given context or overall.

        Args:
            predictors (List[str]): Required.
                List of predictors to get the contributions for.
            context (Optional[dict[str, str]]):
                The context to filter contributions by.
                If None, contributions for all contexts will be returned.
            top_k (int):
                Number of unique categorical predictor values to return.
            descending (bool):
                Whether to sort contributions in descending order.
            missing (bool):
                Whether to include contributions for missing predictor values.
            remaining (bool):
                Whether to include contributions for remaining predictors outside the top-n.
            sort_by (str):
                Method to sort/select top contributions. Options include
                `contribution`, `contribution_abs`, `contribution_weighted`.
                Default is `contribution_abs` which sorts by absolute average contributions.
        """

        validated_sort_by = _CONTRIBUTION_TYPE.validate_and_get_type(
            sort_by
        )

        try:
            validate(top_k=top_k)
        except ValueError as e:
            logger.error("Invalid parameters: %s", e)
            raise

        self._load_data()

        return self._get_predictor_value_contributions(
            contexts=[context] if context else None,
            predictors=predictors,
            limit=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            sort_by=validated_sort_by.value,
        )

    def validate_folder(self):
        """Check if the aggregates folder exists.
        Raises:
            FileNotFoundError: If the aggregates folder does not exist or is empty.
        """
        folder = pathlib.Path(self.data_folderpath)

        if not folder.exists():
            raise FileNotFoundError(
                f"Aggregates folder {folder.name} does not exist. Please ensure the aggregates are generated before loading data."
            )

        # Check if the aggregates folder contains any files
        if not any(folder.iterdir()):
            raise FileNotFoundError(
                f"Aggregates folder {folder.name} is empty. Please ensure the aggregates are generated before loading data."
            )

    def get_unique_contexts_list(
        self,
        context_infos: Optional[List[ContextInfo]] = None,
        with_partition_col: bool = False,
    ) -> List[ContextInfo]:
        return self.context_operations.get_list(context_infos, with_partition_col)

    def _load_data(self):
        if self.initialized:
            return

        try:
            self.validate_folder()
        except FileNotFoundError as e:
            logger.error("Error validating aggregates folder: %s", e)
            raise

        selected_columns = [
            _COL.PARTITON.value,
            _COL.CONTRIBUTION.value,
            _COL.CONTRIBUTION_ABS.value,
            _COL.FREQUENCY.value,
            _COL.PREDICTOR_TYPE.value,
            _COL.PREDICTOR_NAME.value,
            _COL.BIN_CONTENTS.value,
            _COL.BIN_ORDER.value,
            _COL.CONTRIBUTION_MIN.value,
            _COL.CONTRIBUTION_MAX.value,
        ]

        context_ = f"{self.data_folderpath}/{self.data_pattern if self.data_pattern else '*_BATCH_*.parquet'}"

        self.df_contextual = (
            pl.scan_parquet(context_)
            .select(selected_columns)
            .filter(pl.col(_COL.CONTRIBUTION.value) != 0.0)
            .sort(by=_COL.PREDICTOR_NAME.value)
        )
        self.df_contextual = self._filter_single_bin_numeric_predictors(self.df_contextual)

        self.df_overall = (
            pl.scan_parquet(f"{self.data_folderpath}/*_OVERALL.parquet")
            .select(selected_columns)
            .filter(pl.col(_COL.CONTRIBUTION.value) != 0.0)
            .sort(by=_COL.PREDICTOR_NAME.value)
        )
        self.df_overall = self._filter_single_bin_numeric_predictors(self.df_overall)

        self.initialized = True

    def _get_predictor_contributions(
        self,
        contexts: Optional[List[ContextInfo]] = None,
        predictors: Optional[List[str]] = None,
        limit: int = defaults.TOP_N,
        descending: bool = defaults.DESCENDING,
        missing: bool = defaults.MISSING,
        remaining: bool = defaults.REMAINING,
        sort_by: str = defaults.SORT_BY.value,
    ) -> pl.DataFrame:
        contexts = contexts or []
        predictors = predictors or []

        # if no contexts are provided, then we return the overall data
        # if contexts are provided, then we generate the context filters
        # and load the data for those contexts
        df = self._get_df(contexts)

        # If predictors are specified we filter the dataframe for those predictors
        if len(predictors) > 0:
            df = self._filter_for_predictors(df, predictors)

        # If we do not want to include the missing predictor values, we filter them out
        if not missing:
            df = df.filter(_COL.BIN_CONTENTS.value != _SPECIAL.MISSING.name)

        # Aggregate all the different types of contributions
        # note: we need to aggregate frequency over partition to calculate weighted contributions
        df = self._calculate_aggregates(
            df,
            aggregate_frequency_over=[_COL.PARTITON.value],
            aggregate_over=[
                _COL.PARTITON.value,
                _COL.PREDICTOR_NAME.value,
                _COL.PREDICTOR_TYPE.value,
            ],
        )

        # Take the top predictors per partition, sorted by sort_by
        df_top_predictors = self._get_df_with_top_limit(
            df,
            contribution_type=sort_by,
            over=[_COL.PARTITON.value],
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
                    _COL.PARTITON.value,
                    _COL.PREDICTOR_NAME.value,
                    _COL.PREDICTOR_TYPE.value,
                ],
                aggregate_over=[_COL.PARTITON.value],
            )
            df_top_predictors = pl.concat(
                df.select(sorted(df.collect_schema().names()))
                for df in [df_remaining, df_top_predictors]
            )

        # Ensure all predictors are unique and sorted by sort_by
        df_out = df_top_predictors.unique()
        df_out = df_out.sort(by=sort_by, descending=descending)

        return df_out.collect()

    def _get_predictor_value_contributions(
        self,
        contexts: Optional[List[ContextInfo]] = None,
        predictors: Optional[List[str]] = None,
        limit: int = defaults.TOP_K,
        descending: bool = defaults.DESCENDING,
        missing: bool = defaults.MISSING,
        remaining: bool = defaults.REMAINING,
        sort_by: str = defaults.SORT_BY.value,
    ) -> pl.DataFrame:
        # if no contexts are provided, then we return the overall data
        # if contexts are provided, then we generate the context filters
        # and load the data for those contexts
        df = self._get_df(contexts)

        # If predictors are specified we filter the dataframe for those predictors
        predictors = predictors or []
        if predictors is not None or len(predictors) > 0:
            df = self._filter_for_predictors(df, predictors)

        # Aggregate all the different types of contributions
        # note: we need to aggregate frequency over partition to calculate weighted contributions
        df = self._calculate_aggregates(
            df,
            aggregate_frequency_over=[
                _COL.PARTITON.value,
                _COL.PREDICTOR_NAME.value,
                _COL.PREDICTOR_TYPE.value,
            ],
            aggregate_over=[
                _COL.PARTITON.value,
                _COL.PREDICTOR_NAME.value,
                _COL.PREDICTOR_TYPE.value,
                _COL.BIN_ORDER.value,
                _COL.BIN_CONTENTS.value,
            ],
        )

        # Append a sort column and value, note these are not used when
        # finding the top predictors, but are used for logically sorting the final output
        # e.g.:
        # - numeric predictors are sorted by bin order
        # - symbolic predictors are sorted by sort_by
        df = self._get_df_with_sort_info(df, sort_by_column=sort_by)

        # Take the top predictors per partition, sorted by sort_by
        df_top_predictor_values = self._get_df_with_top_limit(
            df,
            contribution_type=sort_by,
            over=[
                _COL.PARTITON.value,
                _COL.PREDICTOR_NAME.value,
                _COL.PREDICTOR_TYPE.value,
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
                    _COL.PARTITON.value,
                    _COL.PREDICTOR_NAME.value,
                    _COL.PREDICTOR_TYPE.value,
                    _COL.BIN_ORDER.value,
                    _COL.BIN_CONTENTS.value,
                ],
                aggregate_over=[
                    _COL.PARTITON.value,
                    _COL.PREDICTOR_NAME.value,
                    _COL.PREDICTOR_TYPE.value,
                ],
            )

            # Add sort information and concat with the top predictor values
            df_remaining = self._get_df_with_sort_info(
                df_remaining, sort_by_column=sort_by
            )
            df_top_predictor_values = pl.concat(
                df.select(sorted(df.collect_schema().names()))
                for df in [df_remaining, df_top_predictor_values]
            )

        # Ensure all predictor values are unique and sorted according to predictor type
        df_out = df_top_predictor_values.unique()
        df_out = df_out.sort(
            by=[*self._get_sort_over_columns(predictors=None), "sort_value"]
        )
        return df_out.collect()

    def _get_df_with_sort_info(
        self,
        df: pl.LazyFrame,
        sort_by_column: str = defaults.SORT_BY.value,
    ) -> pl.LazyFrame:
        """Add a sort column and value to the dataframe based on the predictor type.
        # Sort logic:
        #  - numeric predictors are sorted by bin order
        #  - symbolic predictors are sorted by contribution type
        """

        return df.with_columns(
            pl.when(pl.col(_COL.PREDICTOR_TYPE.value) == _PREDICTOR_TYPE.NUMERIC.value)
            .then(pl.lit(_COL.BIN_ORDER.value))
            .otherwise(pl.lit(sort_by_column))
            .alias("sort_column"),
            pl.when(pl.col(_COL.PREDICTOR_TYPE.value) == _PREDICTOR_TYPE.NUMERIC.value)
            .then(pl.col(_COL.BIN_ORDER.value))
            .otherwise(pl.col(sort_by_column).abs())
            .alias("sort_value"),
        )

    def _filter_for_predictors(
        self, df: pl.LazyFrame, predictors: List[str]
    ) -> pl.LazyFrame:
        if predictors is None or len(predictors) == 0:
            return df

        return df.filter(pl.col(_COL.PREDICTOR_NAME.value).is_in(predictors))

    def _get_df_with_top_limit(
        self,
        df: pl.LazyFrame,
        over: List[str],
        contribution_type: str = defaults.SORT_BY.value,
        limit: int = defaults.TOP_K,
        descending: bool = defaults.DESCENDING,
    ) -> pl.LazyFrame:
        return df.select(
            pl.all()
            .top_k_by(contribution_type, k=limit)
            .over(
                over,
                mapping_strategy="explode",
            )
        )

    def _get_missing_predictor_values_df(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(
            pl.col(_COL.PREDICTOR_NAME.value) == _SPECIAL.MISSING.name
        ).select(pl.all())

    def _get_df(
        self,
        contexts: Optional[List[ContextInfo]] = None,
    ):
        contexts = contexts or []

        if len(contexts) == 0:
            df = self._get_base_df()
        else:
            df_filtered_contexts = self.context_operations.get_df(contexts, True)
            df = self._get_base_df(df_filtered_contexts)
        return df

    def _get_base_df(
        self, df_filtered_contexts: Optional[pl.DataFrame] = None
    ) -> pl.LazyFrame:
        if self.df_overall is None or self.df_contextual is None:
            self._load_data()

        if df_filtered_contexts is None:
            return self.df_overall
        else:
            return self.df_contextual.join(
                df_filtered_contexts.lazy(), on=_COL.PARTITON.value, how="inner"
            )

    def _get_group_by_columns(
        self, predictors: Optional[List[str]] = None
    ) -> List[str]:
        if predictors is None or len(predictors) == 0:
            return [
                _COL.PREDICTOR_NAME.value,
                _COL.PREDICTOR_TYPE.value,
                _COL.PARTITON.value,
            ]
        else:
            return [
                _COL.PREDICTOR_NAME.value,
                _COL.PREDICTOR_TYPE.value,
                _COL.PARTITON.value,
                _COL.BIN_CONTENTS.value,
                _COL.BIN_ORDER.value,
            ]

    def _get_sort_over_columns(
        self, predictors: Optional[List[str]] = None
    ) -> List[str]:
        if predictors is None or len(predictors) == 0:
            return [_COL.PREDICTOR_NAME.value, _COL.PARTITON.value]
        else:
            return [_COL.PARTITON.value]

    def _calculate_remaining_aggregates(
        self,
        df_all: pl.LazyFrame,
        df_anti: pl.LazyFrame,
        aggregate_over: List[str],
        anti_on: List[str],
    ) -> pl.LazyFrame:
        # Needed for calculating the weighted contributions
        df_frequencies = self._add_total_frequency_to_df(df_all, group_by=[_COL.PARTITON.value])

        # Get the remaining contributions by anti-joining the top predictors with the overall data
        # Join on the total frequencies previously calculated
        df_remaining = df_all.join(
            df_anti,
            on=anti_on,
            how="anti",
        ).join(
            df_frequencies,
            on=aggregate_over,
        )

        # Aggregate the remaining contributions
        df_remaining = self._agg_over_columns_in_df(df_remaining, group_by=aggregate_over)

        # If we only aggregate over partition, there will be no bin contents or bin order
        if len(aggregate_over) == 1 and aggregate_over[0] == _COL.PARTITON.value:
            df_remaining = df_remaining.with_columns(
                pl.lit(_SPECIAL.REMAINING.value).alias(_COL.PREDICTOR_NAME.value),
                pl.lit(_PREDICTOR_TYPE.SYMBOLIC).alias(_COL.PREDICTOR_TYPE.value),
            )
        # if we aggregate over partition and predictor name,
        # we need to add the bin contents and bin order
        else:
            df_remaining = df_remaining.with_columns(
                pl.lit(_SPECIAL.REMAINING.value).alias(_COL.BIN_CONTENTS.value),
                pl.lit(0).cast(pl.Int64).alias(_COL.BIN_ORDER.value),
            )

        return df_remaining

    def _calculate_aggregates(
        self,
        df: pl.LazyFrame,
        aggregate_frequency_over: List[str],
        aggregate_over: List[str],
    ) -> pl.LazyFrame:

        data = self._add_total_frequency_to_df(df, aggregate_frequency_over)
        return self._agg_over_columns_in_df(data, aggregate_over)

    @staticmethod
    def _add_total_frequency_to_df(df, group_by):
        df_grouped = (
            df
            .group_by(group_by)
            .agg(
                pl.sum(_COL.FREQUENCY.value)
                .alias(_SPECIAL.TOTAL_FREQUENCY.value)))

        return df_grouped.join(
            df,
            on=group_by,
            how="left"
        )

    def add_frequency_pct_to_df(self, df, group_by):
        """Add a frequency percentage column to the dataframe based on the total frequency per group."""

        df_with_total_frequency = self._add_total_frequency_to_df(df, group_by)
        return df_with_total_frequency.with_columns(
            pl.when(pl.col(_SPECIAL.TOTAL_FREQUENCY.value) == 0)
            .then(0.0)
            .otherwise(
            (pl.col(_COL.FREQUENCY.value) / pl.col(_SPECIAL.TOTAL_FREQUENCY.value) * 100).round(1)
            )
            .alias("frequency_pct")
        )

    @staticmethod
    def _get_mean_aggregates():
        """Get mean contribution aggregates."""
        def _apply(col):
            return pl.col(col).mean().alias(col)
        
        return [
            _apply(_COL.CONTRIBUTION.value),
            _apply(_COL.CONTRIBUTION_ABS.value),
        ]

    @staticmethod
    def _get_weighted_aggregates():
        """Get frequency-weighted contribution aggregates normalized by total frequency."""
        def _apply(col, alias):
            return (
                (pl.col(col) * pl.col(_COL.FREQUENCY.value)).mean()
                / pl.col(_SPECIAL.TOTAL_FREQUENCY.value).first()
            ).alias(alias)
            
        return [
            _apply(_COL.CONTRIBUTION.value, _COL.CONTRIBUTION_WEIGHTED.value),
            _apply(_COL.CONTRIBUTION_ABS.value, _COL.CONTRIBUTION_WEIGHTED_ABS.value),
        ]

    @staticmethod
    def _get_frequency_aggregate():
        """Get frequency sum aggregate."""
        return [pl.col(_COL.FREQUENCY.value).sum().alias(_COL.FREQUENCY.value)]

    @staticmethod
    def _get_bounds_aggregates():
        """Get min and max contribution bounds."""
        return [
            pl.col(_COL.CONTRIBUTION_MIN.value).min().alias(_COL.CONTRIBUTION_MIN.value),
            pl.col(_COL.CONTRIBUTION_MAX.value).max().alias(_COL.CONTRIBUTION_MAX.value),
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
            df.filter(
                (pl.col(_COL.PREDICTOR_TYPE.value) == _PREDICTOR_TYPE.NUMERIC.value)
                & (pl.col(_COL.BIN_CONTENTS.value) != _SPECIAL.MISSING.name)
            )
            .group_by([_COL.PARTITON.value, _COL.PREDICTOR_NAME.value])
            .agg(pl.col(_COL.BIN_ORDER.value).n_unique().alias("bin_count"))
            .filter(pl.col("bin_count") <= 1)
            .select([_COL.PARTITON.value, _COL.PREDICTOR_NAME.value])
        )
        return df.join(
            single_bin_predictors,
            on=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value],
            how="anti",
        )
