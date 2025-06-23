__all__ = ["ExplanationsDataLoader"]

import json
import polars as pl
from typing import Optional, List, cast

from .ExplanationsUtils import _COL, _CONTRIBUTION_TYPE, _PREDICTOR_TYPE, ContextInfo

class ExplanationsDataLoader:

    _TO_SELECT = [
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

    def __init__(self, data_location):
        self.data_location = data_location
        self._scan_data()

        self.context_df = pl.from_dicts(
            [
                {**json.loads(ck)[_COL.PARTITON.value], "filter_string": ck}
                for ck in self.df_contextual.collect()[_COL.PARTITON.value].unique().to_list()
            ]
        )

    def _scan_data(self):
        self.df_contextual = (pl.scan_parquet(f'{self.data_location}/*_BATCH_*.parquet')
                              .select(self._TO_SELECT)
                              .sort(by=_COL.PREDICTOR_NAME.value))

        self.df_overall = (pl.scan_parquet(f"{self.data_location}/*_OVERALL.parquet")
                           .select(self._TO_SELECT)
                           .sort(by=_COL.PREDICTOR_NAME.value))

    def get_predictor_contributions(
        self,
        contexts: Optional[List[ContextInfo]] = None,
        predictors: Optional[List[str]] = None,
        limit: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION,
    ) -> pl.DataFrame:
        contexts = contexts or []
        predictors = predictors or []
        
        contexts_str = self._get_context_filters(contexts)
        base_df = self._get_base_df(contexts_str)

        sort_columns = {
            predictor: self._get_sort_column(predictor, contribution_calculation)
            for predictor in predictors
        }
        df = self._get_aggregates(base_df, predictors).with_columns(
            (
                pl.col(_COL.PREDICTOR_NAME.value).replace(
                    sort_columns, default=contribution_calculation.value
                )
            ).alias("sort_column"),
        )

        df = df.with_columns(
            pl.coalesce(
                pl.when(pl.col("sort_column") == col).then(pl.col(col))
                for col in df.collect_schema().names()
            )
            .cast(pl.Float32)
            .abs()
            .alias("sort_value")
        )

        df_top_k = df.select(
            pl.all()
            .top_k_by(contribution_calculation.value, k=limit, reverse=descending)
            .over(self._get_sort_over_columns(predictors), mapping_strategy="explode")
        )

        if missing:
            df_top_k = pl.concat(
                [
                    df_top_k,
                    df.filter(pl.col(_COL.PREDICTOR_NAME.value) == "MISSING").select(pl.all()),
                ]
            ).unique()

        if remaining:
            remaining_base_df = base_df.join(
                df_top_k, on=self._get_group_by_columns(predictors), how="anti"
            )

            if len(predictors) == 0:
                remaining_base_df = remaining_base_df.with_columns(
                    pl.lit("REMAINING").alias(_COL.PREDICTOR_NAME.value),
                    pl.lit(_PREDICTOR_TYPE.SYMBOLIC.value).alias("predictor_type"),
                )
            else:
                remaining_base_df = remaining_base_df.with_columns(
                    pl.lit("REMAINING").alias("bin_contents"),
                    pl.lit(0).cast(pl.Int64).alias("bin_order"),
                )

            remaining_df = self._get_aggregates(
                remaining_base_df, predictors
            ).with_columns(
                (
                    pl.col(_COL.PREDICTOR_NAME.value).replace(
                        sort_columns, default=contribution_calculation.value
                    )
                ).alias("sort_column"),
            )

            remaining_df = remaining_df.with_columns(
                pl.coalesce(
                    pl.when(pl.col("sort_column") == col).then(pl.col(col))
                    for col in remaining_df.collect_schema().names()
                )
                .cast(pl.Float32)
                .abs()
                .alias("sort_value")
            )

            df_top_k = pl.concat([df_top_k, remaining_df])

        df_sorted = df_top_k.sort(
            by=[*self._get_sort_over_columns(predictors), "sort_value"]
        )

        columns = [
            contribution_calculation.value,
            _COL.PREDICTOR_NAME.value,
            _COL.BIN_CONTENTS.value,
            "sort_column",
            "sort_value",
            _COL.FREQUENCY.value,
            _COL.CONTRIBUTION_MIN.value,
            _COL.CONTRIBUTION_MAX.value,
            _COL.PARTITON.value,
        ]

        columns = [
            column for column in columns if column in df_sorted.collect_schema().names()
        ]

        return df_sorted.collect().select(columns)

    def get_context_keys(self) -> list:
        """Get the context keys for the current data filter.

        Returns
        -------
        list
            A list of context keys.
        """
        return self.context_df.select(pl.exclude("filter_string")).columns

    def _get_filtered_context_df(
        self, 
        context_infos: Optional[List[ContextInfo]] = None
    ) -> pl.DataFrame:
        context_infos = context_infos or []

        # If no context_infos are provided, return the full context_df
        if len(context_infos) == 0:
            df = self.context_df
        else:
            df = pl.DataFrame()
            for context_info in context_infos:
                expr = [
                    pl.col(column_name) == column_value
                    for column_name, column_value in context_info.items()
                ]

                df = pl.concat([df, self.context_df.filter(expr)])
        return df

    def _get_context_filters(
        self, 
        context_infos: Optional[List[ContextInfo]] = None
    ) -> list[str]:
        context_infos = context_infos or []
        df = self._get_filtered_context_df(context_infos)
        return df.select("filter_string").unique().to_series().to_list()

    def get_context_infos(
        self, 
        context_infos: Optional[List[ContextInfo]] = None
    ) -> List[ContextInfo]:
        """Get the possible context key filters for the provided context info.

        Parameters
        ----------
        context_keys : ContextInfo, optional
            A dictionary of context keys and their values.

        Returns
        -------
        list[str]
            A list of context key partitions.
        """
        
        df = self._get_filtered_context_df(context_infos)
        return cast(
            list[ContextInfo], 
            df.select(pl.exclude("filter_string")).unique().to_dicts()
        )

    def _get_base_df(self, contexts: List[str]) -> pl.LazyFrame:
        if contexts is None:
            # print("No contexts provided, returning overall data")
            return self.df_overall.with_columns(pl.lit("").alias(_COL.PARTITON.value))
        else:
            # print(f"returning for {contexts}")
            return self.df_contextual.filter(
                pl.col(_COL.PARTITON.value).str.contains_any(contexts)
            )

    def _get_group_by_columns(self, predictors: List[str]) -> List[str]:
        columns = (
            [_COL.PREDICTOR_NAME.value, "bin_contents", "bin_order"]
            if len(predictors) > 0
            else [_COL.PREDICTOR_NAME.value, "predictor_type"]
        )
        columns.append(_COL.PARTITON.value)
        return columns

    def _get_sort_over_columns(self, predictors: List[str]) -> List[str]:
        columns = [_COL.PREDICTOR_NAME.value] if len(predictors) > 0 else []
        columns.append(_COL.PARTITON.value)
        return columns

    def _get_aggregates(self, df: pl.LazyFrame, predictors: List[str]) -> pl.LazyFrame:
        aggregate_by_list = [
            pl.col(_COL.CONTRIBUTION.value).mean().alias(_COL.CONTRIBUTION.value),
            pl.col(_COL.CONTRIBUTION_ABS.value).mean().alias(_COL.CONTRIBUTION_ABS.value),
            (
                (pl.col(_COL.CONTRIBUTION.value) * pl.col(_COL.FREQUENCY.value)).mean() 
                / pl.col(_COL.FREQUENCY.value).sum()
            ).alias(_COL.CONTRIBUTION_WEIGHTED.value),
            (
                (pl.col(_COL.CONTRIBUTION_ABS.value) * pl.col(_COL.FREQUENCY.value)).mean()
                / pl.col(_COL.FREQUENCY.value).sum()
            ).alias(_COL.CONTRIBUTION_WEIGHTED_ABS.value),
            pl.col(_COL.FREQUENCY.value).sum().alias(_COL.FREQUENCY.value),
            pl.col(_COL.CONTRIBUTION_MIN.value).min().alias(_COL.CONTRIBUTION_MIN.value),
            pl.col(_COL.CONTRIBUTION_MAX.value).max().alias(_COL.CONTRIBUTION_MAX.value),
        ]
        if len(predictors) > 0:
            return (
                df.filter(pl.col(_COL.PREDICTOR_NAME.value).is_in(predictors))
                .group_by(self._get_group_by_columns(predictors))
                .agg(aggregate_by_list)
            )
        else:
            return (df
                    .group_by(self._get_group_by_columns(predictors))
                    .agg(aggregate_by_list))

    def _get_sort_column(
        self, 
        predictor, 
        contribution_calculation: _CONTRIBUTION_TYPE
    ) -> str:
        # Determine the sort column based on the predictor and contribution calculation type.
        if predictor is None:
            return contribution_calculation.value
        else:
            predictor_type = (
                self.df_overall.filter(pl.col(_COL.PREDICTOR_NAME.value) == predictor)
                .select("predictor_type")
                .first()
                .collect()
                .item()
            )
            return (
                "bin_order"
                if predictor_type == _PREDICTOR_TYPE.NUMERIC.value
                else contribution_calculation.value
            )

    # TODO: what is this?
    def _get_sort_expression(
        self, 
        predictor: str, 
        contribution_calculation: _CONTRIBUTION_TYPE
    ) -> List[pl.Expr]:
        sort_column = self._get_sort_column(predictor, contribution_calculation)
        return (
            pl.col(sort_column).abs()
            if sort_column in [e.value for e in _CONTRIBUTION_TYPE]
            else pl.col(sort_column)
        )
        
        

