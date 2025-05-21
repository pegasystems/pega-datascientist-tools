import os
from typing import Dict, List, Optional, Union
from datetime import datetime
import json

import polars as pl
import polars.selectors as cs

from .Plots import Plots
from ..utils.cdh_utils import _polars_capitalize, _apply_query
from ..utils.types import QUERY
from ..pega_io.File import read_ds_export


class ImpactAnalyzer:
    ia_data: pl.LazyFrame

    def __init__(self, raw_data: pl.LazyFrame):
        self.plot = Plots(ia=self)

        # print(raw_data.head(6).collect())
        # Column names may be PDC specific, we should be changing once w got more sources going

        nba_data = (
            raw_data.filter(pl.col("IsActive"))
            .group_by(["SnapshotTime", "ChannelName"])
            .agg(
                pl.lit("NBA").alias("ExperimentName"),
                pl.col("Impressions_NBA", "Accepts_NBA", "ActionValuePerImp_NBA")
                .top_k_by("Impressions_NBA", k=1)
                .first(),
            )
            .with_columns(
                Impressions="Impressions_NBA",
                Accepts="Accepts_NBA",
                ActionValuePerImp="ActionValuePerImp_NBA",
            )
            # .rename(lambda x: x.removesuffix("_NBA"))
            # .with_columns(Reference=pl.lit(True))
        )

        other_data = (
            raw_data.filter(pl.col("IsActive"))
            .select(
                "SnapshotTime",
                "ChannelName",
                "ExperimentName",
                "Impressions_Control",
                "Accepts_Control",
                "ActionValuePerImp_Control",
                "Impressions_NBA",
                "Accepts_NBA",
                "ActionValuePerImp_NBA",
            )
            .rename(lambda x: x.removesuffix("_Control"))
            # .with_columns(Reference=pl.lit(False))
        )

        self.ia_data = (
            pl.concat([nba_data, other_data], how="diagonal_relaxed")
            .sort("SnapshotTime", "ChannelName", "ExperimentName")
            .rename({"ChannelName": "Channel", "ExperimentName": "Experiment"})
        )

    @classmethod
    def from_pdc(
        cls,
        pdc_source: Union[os.PathLike, str, dict],
        *,
        query: Optional[QUERY] = None,
        return_df: Optional[bool] = False,
    ):
        """Create an ImpactAnalyzer instance from a PDC file

        Parameters
        ----------
        pdc_filename : Union[os.PathLike, str]
            The full path to the PDC file
        query : Optional[QUERY], optional
            An optional argument to filter out selected data, by default None

        Returns
        -------
        ImpactAnalyzer
            The properly initialized ImpactAnalyzer object

        """
        if isinstance(pdc_source, dict):
            return cls._from_pdc_json(pdc_source, query=query, return_df=return_df)
        else:
            with open(pdc_source) as pdc_json_data:
                return cls._from_pdc_json(
                    json.load(pdc_json_data), query=query, return_df=return_df
                )

    @classmethod
    def _from_pdc_json(
        cls,
        json_data: dict,
        *,
        query: Optional[QUERY] = None,
        return_df: Optional[bool] = False,
    ):
        """Create an ImpactAnalyzer instance from PDC JSON data"""
        if len(json_data["pxResults"]) != 1:
            raise Exception("Expected just one result under 1st level pxResults.")
        # lets hope the time format is consistent!
        # can we use utils here?
        date = datetime.strptime(
            json_data["pxResults"][0]["SnapshotTime"], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        actual_ia_data = json_data["pxResults"][0]["pxResults"]
        if len(actual_ia_data) < 1:
            # No data
            return None

        wide_data = pl.DataFrame(actual_ia_data).lazy()

        if query is not None:
            wide_data = _apply_query(wide_data, query=query)

        df = (
            wide_data.select(
                [
                    "SnapshotTime",  # will be overwritten but nicely at the front
                    "ExperimentName",
                    "IsActive",
                    "LastDataReceived",
                    "AggregationFrequency",
                    # keys
                    "ChannelName",
                    # raw data
                    "Impressions_NBA",
                    "Impressions_Control",
                    "Accepts_NBA",
                    "Accepts_Control",
                    "ActionValuePerImp_NBA",
                    "ActionValuePerImp_Control",
                    # derived
                    "AcceptRate_Control",
                    "AcceptRate_NBA",
                    "ValueLift",
                    "EngagementLift",
                    "EngagementLiftInterval",
                    "ValueLiftInterval",
                    "IsSignificant",
                ]
            )
            .filter(AggregationFrequency="Daily")
            .with_columns(pl.lit(date).cast(pl.Date).alias("SnapshotTime"))
            .with_columns(
                pl.when(LastDataReceived="Yesterday")
                .then(pl.col("SnapshotTime") - pl.duration(days=1))
                .otherwise(pl.col("SnapshotTime"))
                .alias("SnapshotTime")
            )
            .drop(["LastDataReceived", "AggregationFrequency"])
            .sort(["SnapshotTime", "ChannelName", "ExperimentName"])
        )

        if return_df:
            return df

        return ImpactAnalyzer(df)

    def summary_by_channel(self):
        return self._summarize(by=["Channel"])
    
    def _summarize(self, by: List[str]):
        return (
            self.ia_data.sort(by + ["Experiment"])
            .group_by(by + ["Experiment"], maintain_order=True)
            .agg(
                CTR=pl.sum("Accepts") / pl.sum("Impressions"),
                ActionValuePerImp=pl.sum("ActionValuePerImp") / pl.sum("Impressions"),
                CTR_NBA=pl.sum("Accepts_NBA") / pl.sum("Impressions_NBA"),
                ActionValuePerImp_NBA=pl.sum("ActionValuePerImp_NBA")
                / pl.sum("Impressions_NBA"),
            )
            .with_columns(
                CTR_Lift=(pl.col("CTR_NBA") - pl.col("CTR")) / pl.col("CTR"),
                Value_Lift=(
                    pl.col("ActionValuePerImp_NBA") - pl.col("ActionValuePerImp")
                )
                / pl.col("ActionValuePerImp"),
                # TODO confidence intervals and significance
            )
        )
