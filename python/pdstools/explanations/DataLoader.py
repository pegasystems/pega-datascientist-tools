__all__ = ["DataLoader"]

from typing import TYPE_CHECKING, List, Optional

import polars as pl
import logging
import pathlib

from ..utils.namespaces import LazyNamespace
from .ExplanationsUtils import (
    _COL,
    _CONTRIBUTION_TYPE,
    _PREDICTOR_TYPE,
    _SPECIAL,
    ContextInfo,
    ContextOperations,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .Explanations import Explanations


class DataLoader(LazyNamespace):
    dependencies = ["polars"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations
        self.df_contextual = None
        self.df_overall = None
        self.context_operations = ContextOperations(data_loader=self)
        super().__init__()
        self.initialized = False

    def load_data(self):
        try:
            self._validate_aggregates_folder()
        except FileNotFoundError as e:
            logger.error(f"Error validating aggregates folder: {e}")
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

        self.df_contextual = (
            pl.scan_parquet(
                f"{self.explanations.aggregates.aggregates_folder}/*_BATCH_*.parquet"
            )
            .select(selected_columns)
            .sort(by=_COL.PREDICTOR_NAME.value)
        )

        self.df_overall = (
            pl.scan_parquet(
                f"{self.explanations.aggregates.aggregates_folder}/*_OVERALL.parquet"
            )
            .select(selected_columns)
            .sort(by=_COL.PREDICTOR_NAME.value)
        )

        self.initialized = True

    def get_df_contextual(self) -> pl.LazyFrame:
        if not self.initialized:
            self.load_data()
        return self.df_contextual

    def get_context_operations(self) -> ContextOperations | None:
        return self.context_operations

    def get_top_n_predictor_contribution_overall(
        self,
        top_n: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> pl.DataFrame:
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        if not self.initialized:
            self.load_data()

        # if no contexts are provided, then we return the overall data
        return self._get_predictor_contributions(
            limit=top_n,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_type.value,
        )

    def get_top_k_predictor_value_contribution_overall(
        self,
        predictors: List[str],
        top_k: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> pl.DataFrame:
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        if not self.initialized:
            self.load_data()

        return self._get_predictor_value_contributions(
            predictors=predictors,
            limit=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_type.value,
        )

    def get_top_n_predictor_contribution_by_context(
        self,
        context: ContextInfo,
        top_n: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> pl.DataFrame:
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        if not self.initialized:
            self.load_data()

        return self._get_predictor_contributions(
            contexts=[context],
            limit=top_n,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_type.value,
        )

    def get_top_k_predictor_value_contribution_by_context(
        self,
        context: ContextInfo,
        predictors: List[str],
        top_k: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ):
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        if not self.initialized:
            self.load_data()

        return self._get_predictor_value_contributions(
            contexts=[context],
            predictors=predictors,
            limit=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_type.value,
        )

    def get_unique_contexts_df(
        self,
        context_infos: Optional[List[ContextInfo]] = None,
        with_partition_col: bool = False,
    ) -> pl.DataFrame:
        return self.context_operations.get_df(context_infos, with_partition_col)

    def get_unique_contexts_list(
        self,
        context_infos: Optional[List[ContextInfo]] = None,
        with_partition_col: bool = False,
    ) -> List[ContextInfo]:
        return self.context_operations.get_list(context_infos, with_partition_col)

    def _validate_aggregates_folder(self):
        folder = pathlib.Path(self.explanations.aggregates.aggregates_folder)

        """Check if the aggregates folder exists."""
        if not folder.exists():
            raise FileNotFoundError(
                f"Aggregates folder {folder.name} does not exist. Please ensure the aggregates are generated before loading data."
            )

        # Check if the aggregates folder contains any files
        if not any(folder.iterdir()):
            raise FileNotFoundError(
                f"Aggregates folder {folder.name} is empty. Please ensure the aggregates are generated before loading data."
            )

    def _get_predictor_value_contributions(
        self,
        contexts: Optional[List[ContextInfo]] = None,
        predictors: Optional[List[str]] = None,
        limit: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> pl.DataFrame:
        # if no contexts are provided, then we return the overall data
        # if contexts are provided, then we generate the context filters and load the data for those contexts
        df = self._get_df(contexts)

        # If predictors are specified we filter the dataframe for those predictors
        predictors = predictors or []
        if predictors is not None or len(predictors) > 0:
            df = self._filter_for_predictors(df, predictors)

        # Aggregate all the different types of contributions, note we need to aggregate frequency over partition to calculate weighted contributions
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

        # Append a sort column and value, note these are note used when finding the top predictors, but are used for logically sorting the final output
        # e.g.:
        # - numeric predictors are sorted by bin order
        # - symbolic predictors are sorted by contribution type
        df = self._get_df_with_sort_info(df, sort_by_column=contribution_type)

        # Take the top predictors per partition, sorted by contribution type
        df_top_predictor_values = self._get_df_with_top_limit(
            df,
            contribution_type=contribution_type,
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

        # If we want to include the cumulative contribution of all predictor values outside of the top `limit`, we calculate the remaining contributions
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
                df_remaining, sort_by_column=contribution_type
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

    def _get_predictor_contributions(
        self,
        contexts: Optional[List[ContextInfo]] = None,
        predictors: Optional[List[str]] = None,
        limit: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> pl.DataFrame:
        contexts = contexts or []
        predictors = predictors or []

        # if no contexts are provided, then we return the overall data
        # if contexts are provided, then we generate the context filters and load the data for those contexts
        df = self._get_df(contexts)

        # If predictors are specified we filter the dataframe for those predictors
        if len(predictors) > 0:
            df = self._filter_for_predictors(df, predictors)

        # If we do not want to include the missing predictor values, we filter them out
        if not missing:
            df = df.filter(_COL.BIN_CONTENTS.value != _SPECIAL.MISSING.name)

        # Aggregate all the different types of contributions, note we need to aggregate frequency over partition to calculate weighted contributions
        df = self._calculate_aggregates(
            df,
            aggregate_frequency_over=[_COL.PARTITON.value],
            aggregate_over=[
                _COL.PARTITON.value,
                _COL.PREDICTOR_NAME.value,
                _COL.PREDICTOR_TYPE.value,
            ],
        )

        # Take the top predictors per partition, sorted by contribution type
        df_top_predictors = self._get_df_with_top_limit(
            df,
            contribution_type=contribution_type,
            over=[_COL.PARTITON.value],
            limit=limit,
            descending=descending,
        )

        # If we want to include the cumulative contribution of all predictors outside of the top `limit`, we calculate the remaining contributions
        if remaining:
            # Calculate the remaining contributions by aggregating the contributions of all predictors not in the top `limit`
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

        # Ensure all predictors are unique and sorted by contribution type
        df_out = df_top_predictors.unique()
        df_out = df_out.sort(by=contribution_type)

        return df_out.collect()

    # if passing a list of predictors, then sorted by the predictor type
    #  - numeric predictors are sorted by bin order
    #  - symbolic predictors are sorted by contribution type
    def _get_df_with_sort_info(
        self,
        df: pl.LazyFrame,
        sort_by_column: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> pl.LazyFrame:
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

        # filter for predictors
        return df.filter(pl.col(_COL.PREDICTOR_NAME.value).is_in(predictors))

    def _get_df_with_top_limit(
        self,
        df: pl.LazyFrame,
        over: List[str],
        contribution_type: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
        limit: int = 10,
        descending: bool = True,
    ) -> pl.LazyFrame:
        return df.select(
            pl.all()
            .top_k_by(contribution_type, k=limit, reverse=descending)
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
            df_filtered_contexts = self.get_unique_contexts_df(contexts, True)
            df = self._get_base_df(df_filtered_contexts)
        return df

    def _get_base_df(
        self, df_filtered_contexts: Optional[pl.DataFrame] = None
    ) -> pl.LazyFrame:
        if self.df_overall is None or self.df_contextual is None:
            self.load_data()

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
        df_frequencies = df_all.group_by(aggregate_over).agg(
            pl.col(_COL.FREQUENCY.value).sum().alias(_SPECIAL.TOTAL_FREQUENCY.value)
        )

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

        aggregate_by_list = [
            pl.col(_COL.CONTRIBUTION.value).mean().alias(_COL.CONTRIBUTION.value),
            pl.col(_COL.CONTRIBUTION_ABS.value)
            .mean()
            .alias(_COL.CONTRIBUTION_ABS.value),
            (
                (pl.col(_COL.CONTRIBUTION.value) * pl.col(_COL.FREQUENCY.value)).mean()
                / (pl.col(_SPECIAL.TOTAL_FREQUENCY.value).first())
            ).alias(_COL.CONTRIBUTION_WEIGHTED.value),
            (
                (
                    pl.col(_COL.CONTRIBUTION_ABS.value) * pl.col(_COL.FREQUENCY.value)
                ).mean()
                / (pl.col(_SPECIAL.TOTAL_FREQUENCY.value).first())
            ).alias(_COL.CONTRIBUTION_WEIGHTED_ABS.value),
            pl.col(_COL.FREQUENCY.value).sum().alias(_COL.FREQUENCY.value),
            pl.col(_COL.CONTRIBUTION_MIN.value)
            .min()
            .alias(_COL.CONTRIBUTION_MIN.value),
            pl.col(_COL.CONTRIBUTION_MAX.value)
            .max()
            .alias(_COL.CONTRIBUTION_MAX.value),
        ]

        # Aggregate the remaining contributions
        df_remaining = df_remaining.group_by(aggregate_over).agg(aggregate_by_list)

        # If we only aggregate over partition, there will be no bin contents or bin order
        if len(aggregate_over) == 1 and aggregate_over[0] == _COL.PARTITON.value:
            df_remaining = df_remaining.with_columns(
                pl.lit(_SPECIAL.REMAINING.value).alias(_COL.PREDICTOR_NAME.value),
                pl.lit(_PREDICTOR_TYPE.SYMBOLIC).alias(_COL.PREDICTOR_TYPE.value),
            )
        # if we aggregate over partition and predictor name, we need to add the bin contents and bin order
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
        # Needed for calculating the weighted contributions
        df_frequencies = df.group_by(aggregate_frequency_over).agg(
            pl.col(_COL.FREQUENCY.value).sum().alias(_SPECIAL.TOTAL_FREQUENCY.value)
        )

        df_remaining = df.join(
            df_frequencies,
            on=aggregate_frequency_over,
        )

        aggregate_by_list = [
            pl.col(_COL.CONTRIBUTION.value).mean().alias(_COL.CONTRIBUTION.value),
            pl.col(_COL.CONTRIBUTION_ABS.value)
            .mean()
            .alias(_COL.CONTRIBUTION_ABS.value),
            (
                (pl.col(_COL.CONTRIBUTION.value) * pl.col(_COL.FREQUENCY.value)).mean()
                / (pl.col(_SPECIAL.TOTAL_FREQUENCY.value).first())
            ).alias(_COL.CONTRIBUTION_WEIGHTED.value),
            (
                (
                    pl.col(_COL.CONTRIBUTION_ABS.value) * pl.col(_COL.FREQUENCY.value)
                ).mean()
                / (pl.col(_SPECIAL.TOTAL_FREQUENCY.value).first())
            ).alias(_COL.CONTRIBUTION_WEIGHTED_ABS.value),
            pl.col(_COL.FREQUENCY.value).sum().alias(_COL.FREQUENCY.value),
            pl.col(_COL.CONTRIBUTION_MIN.value)
            .min()
            .alias(_COL.CONTRIBUTION_MIN.value),
            pl.col(_COL.CONTRIBUTION_MAX.value)
            .max()
            .alias(_COL.CONTRIBUTION_MAX.value),
        ]

        return df_remaining.group_by(aggregate_over).agg(aggregate_by_list)
