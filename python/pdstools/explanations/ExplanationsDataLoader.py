__all__ = ["ExplanationsDataLoader"]

import json
import polars as pl
from typing import Optional, List, cast

from .ExplanationsUtils import _COL, _SPECIAL, _CONTRIBUTION_TYPE, _PREDICTOR_TYPE, ContextInfo

class ExplanationsDataLoader:

    def __init__(self, data_location):
        self.data_location = data_location
        self._scan_data()

        self.unique_context_df = pl.from_dicts(
            [
                {**json.loads(ck)[_COL.PARTITON.value], "filter_string": ck}
                for ck in self.df_contextual.collect()[_COL.PARTITON.value].unique().to_list()
            ]
        )

    def _scan_data(self):
        
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
        
        self.df_contextual = (pl.scan_parquet(f'{self.data_location}/*_BATCH_*.parquet')
                              .select(selected_columns)
                              .sort(by=_COL.PREDICTOR_NAME.value))

        self.df_overall = (pl.scan_parquet(f"{self.data_location}/*_OVERALL.parquet")
                           .select(selected_columns)
                           .sort(by=_COL.PREDICTOR_NAME.value))
    
    # get top-n predictor contributions for overall model level contributions or for a list of contexts
    def get_top_n_predictor_contribution_overall(
        self,
        top_n: int = 10,                                         
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION
    ) -> pl.DataFrame:
        
        return self._get_predictor_contributions(
            limit = top_n,
            descending = descending,
            missing = missing,
            remaining = remaining,
            contribution_type = contribution_type.value
        )
    
    def get_top_k_predictor_value_contribution_overall(
        self,
        predictors: list[str],
        top_k: int = 10,                                         
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION
    ) -> pl.DataFrame:
        return self._get_predictor_contributions(
            predictors=predictors,
            limit=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_type.value
        )  
    
    def get_top_n_predictor_contribution_by_context(
        self,
        context: ContextInfo,
        top_n: int = 10,                                         
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION
    ) -> pl.DataFrame:
        return self._get_predictor_contributions(
            contexts=[context],
            limit=top_n,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_type.value
        )
    
    def get_top_k_predictor_value_contribution_by_context(
        self,
        context: ContextInfo,
        predictors: List[str],
        top_k: int = 10,                                         
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION
    ):
        return self._get_predictor_contributions(
            contexts=[context],
            predictors=predictors,
            limit=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_type.value
        )
        
    def _get_predictor_contributions(self, 
            contexts: Optional[List[ContextInfo]] = None,
            predictors: Optional[List[str]] = None,
            limit: int = 10,
            descending: bool = True,
            missing: bool = True,
            remaining: bool = True,
            contribution_type: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> pl.DataFrame:
        
        context_list = self._get_context_filters(contexts)
        df = self._get_base_df(context_list)
        
        df = self._get_df_with_sort_info(df,
                                         predictors=predictors,
                                         sort_by_column=contribution_type)
        
        df_top_n = self._get_df_with_top_limit(df, 
                                         contribution_type = contribution_type, 
                                         limit=limit, 
                                         descending=descending)
        
        if missing:
            df_missing = self._get_missing_predictor_values_df(df)
            
        if remaining:
            df_remaining = self._get_remaining_predictor_values_df(df,
                                                          df_top_n,
                                                          predictors=predictors,
                                                          contribution_type=contribution_type)
            
        df_top_n = pl.concat([df_top_n, df_missing, df_remaining])
        
        df_top_n = df_top_n.sort(
            by=[*self._get_sort_over_columns(predictors=None), "sort_value"]
        )
        
        return df_top_n.collect()
    
    # if passing a list of predictors, then sorted by the predictor type
    #  - numeric predictors are sorted by bin order
    #  - symbolic predictors are sorted by contribution type
    # if no predictors are provided, then sorted by passed contribution type
    def _get_df_with_sort_info(self, 
                               df: pl.LazyFrame, 
                               predictors: Optional[list[str]] = None,
                               sort_by_column: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value, 
    ) -> pl.LazyFrame:
        
        df = self._grouped_predictors_by_contribution_type(df, predictors)
        
        # If no predictors are provided, sort by passed contribution type
        if predictors is None or len(predictors) == 0:
            return df.with_columns(
                pl.lit(sort_by_column).alias("sort_column"),
                pl.col(sort_by_column).abs().alias("sort_value")
            )
        else:
            return df.with_columns(
                pl.when(pl.col(_COL.PREDICTOR_TYPE.value) == _PREDICTOR_TYPE.NUMERIC.value)
                .then(pl.lit(_COL.BIN_ORDER.value))
                .otherwise(pl.lit(sort_by_column))
                .alias("sort_column"),
                
                pl.when(pl.col(_COL.PREDICTOR_TYPE.value) == _PREDICTOR_TYPE.NUMERIC.value)
                .then(pl.col(_COL.BIN_ORDER.value).abs())
                .otherwise(pl.col(sort_by_column).abs())
                .alias("sort_value")
            )
    
    def _get_df_with_top_limit(self, 
                               df: pl.LazyFrame, 
                               predictors: Optional[List[str]] = None,
                               contribution_type: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value, 
                               limit: int = 10, 
                               descending: bool = True) -> pl.LazyFrame:
        return df.select(
            pl.all()
            .top_k_by(contribution_type, k=limit, reverse=descending)
            .over(self._get_sort_over_columns(predictors=predictors), mapping_strategy="explode")
        )
    
    def _get_missing_predictor_values_df(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.col(_COL.PREDICTOR_NAME.value) == _SPECIAL.MISSING.name).select(pl.all())
    
    def _get_remaining_predictor_values_df(self, 
                                            df: pl.LazyFrame,
                                            df_subset: pl.LazyFrame,
                                            predictors: Optional[List[str]] = None,
                                            contribution_type: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value
                                            ):
        df_remaining_base = df.join(
            df_subset, 
            on=self._get_group_by_columns(predictors), 
            how="anti"
        )

        if predictors is None or len(predictors) == 0:
            # add `remaining` predictor name and give it symbolic type
            df_remaining_base = df_remaining_base.with_columns(
                pl.lit(_SPECIAL.REMAINING.value).alias(_COL.PREDICTOR_NAME.value),
                pl.lit(_PREDICTOR_TYPE.SYMBOLIC.value).alias(_COL.PREDICTOR_TYPE.value),
            )
        else:
            df_remaining_base = df_remaining_base.with_columns(
                pl.lit(_SPECIAL.REMAINING.value).alias(_COL.BIN_CONTENTS.value),
                pl.lit(0).cast(pl.Int64).alias(_COL.BIN_ORDER.value),
            )

        return self._get_df_with_sort_info(df_remaining_base, predictors, contribution_type)

    def get_context_keys(self) -> list:
        """Get the context keys for the current data filter.

        Returns
        -------
        list
            A list of context keys.
        """
        return self.unique_context_df.select(pl.exclude("filter_string")).columns

    def _get_filtered_context_df(
        self, 
        context_infos: Optional[List[ContextInfo]] = None
    ) -> pl.DataFrame:
        context_infos = context_infos or []

        # If no context_infos are provided, return the full context_df
        if len(context_infos) == 0:
            df = self.unique_context_df
        else:
            df = pl.DataFrame()
            for context_info in context_infos:
                expr = [
                    pl.col(column_name) == column_value
                    for column_name, column_value in context_info.items()
                ]

                df = pl.concat([df, self.unique_context_df.filter(expr)])
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

    def _get_base_df(self, contexts: Optional[List[str]] = None) -> pl.LazyFrame:
        
        if contexts is None:
            # print("No contexts provided, returning overall data")
            return self.df_overall
        else:
            # print(f"returning for {contexts}")
            return self.df_contextual.filter(
                pl.col(_COL.PARTITON.value).str.contains_any(contexts)
            )

    def _get_group_by_columns(self, predictors: Optional[List[str]] = None) -> List[str]:
        if predictors is None or len(predictors) == 0:
            return [_COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value, _COL.PARTITON.value]
        else:
            return [_COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value, _COL.BIN_CONTENTS.value, _COL.BIN_ORDER.value, _COL.PARTITON.value]
    
    def _get_sort_over_columns(self, predictors: Optional[List[str]] = None) -> List[str]:
        columns = [] if predictors is None else [_COL.PREDICTOR_NAME.value]
        columns.append(_COL.PARTITON.value)
        return columns


    def _grouped_predictors_by_contribution_type(
        self, 
        df: pl.LazyFrame, 
        predictors: Optional[List[str]] = None
    ) -> pl.LazyFrame:
        # group by 
        # Aggregate by
        # [ contribution, contribution_abs, contribution_weighted, contribution_weighted_abs
        # frequency, contribution_min, contribution_max ]
        
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
        
        df_tmp = df.filter(pl.col(_COL.PREDICTOR_NAME.value).is_in(predictors)) if predictors else df
        return df_tmp.group_by(self._get_group_by_columns(predictors)).agg(aggregate_by_list)
        
        

