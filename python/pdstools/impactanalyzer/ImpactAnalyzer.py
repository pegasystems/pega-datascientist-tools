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
    # Raw dq
    ia_data: pl.LazyFrame

    def __init__(self, raw_data: pl.LazyFrame):
        self.plot = Plots(ia=self)

        # Column names may be PDC specific, we should be changing once w got more sources going

        nba_data = (
            raw_data.filter(pl.col("IsActive"))
            .group_by(["SnapshotTime", "ChannelName"])
            .agg(
                pl.lit("NBA").alias("ExperimentName"),
                pl.col(["Impressions_NBA", "Accepts_NBA", "ActionValuePerImp_NBA"])
                .top_k_by("Impressions_NBA", k=1)
                .first(),
            )
            .rename(lambda x: x.removesuffix("_NBA"))
            .with_columns(Reference=pl.lit(True))
        )

        other_data = (
            raw_data.filter(pl.col("IsActive"))
            .select(
                [
                    "SnapshotTime",
                    "ChannelName",
                    "ExperimentName",
                    "Impressions_Control",
                    "Accepts_Control",
                    "ActionValuePerImp_Control",
                ]
            )
            .rename(lambda x: x.removesuffix("_Control"))
            .with_columns(Reference=pl.lit(False))
        )

        ia_metrics_data = (
            pl.concat([nba_data, other_data], how="diagonal_relaxed")
            # date trunc would happen here, with a simple agg on impression, accepts and actionvalueperimp
            .sort(
                ["SnapshotTime", "ChannelName", "ExperimentName", "Reference"]
            ).rename({"ChannelName": "Channel"})
        )

        # self.data = _polars_capitalize(data)
        # Initialize impact_data with the required fields
        self.ia_data = ia_metrics_data

    @classmethod
    def from_pdc(
        cls,
        pdc_filename: Union[os.PathLike, str],
        query: Optional[QUERY] = None,
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
        with open(pdc_filename) as fd:
            json_data = json.load(fd)
            if len(json_data["pxResults"]) != 1:
                raise Exception("Expected just one result under 1st level pxResults.")
            # lets hope the time format is consistent!
            # can we use utils here?
            date = datetime.strptime(
                json_data["pxResults"][0]["SnapshotTime"], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            wide_data = pl.DataFrame(json_data["pxResults"][0]["pxResults"]).lazy()

            if query is not None:
                wide_data = _apply_query(wide_data, query=query)

            return ImpactAnalyzer(
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

    def _summarize(self, by: List[str]):
        return (
            self.ia_data.sort("ExperimentName", "Reference")
            .group_by(by + ["ExperimentName", "Reference"], maintain_order=True)
            .agg(
                CTR=pl.sum("Accepts") / pl.sum("Impressions"),
                ActionValuePerImp=pl.sum("ActionValuePerImp") / pl.sum("Impressions"),
            )
            .with_columns(
                pl.repeat(pl.col("CTR").first(), pl.len())
                .over(None if by == [] else by)
                .alias("CTR_Reference"),
                pl.repeat(pl.col("ActionValuePerImp").first(), pl.len())
                .over(None if by == [] else by)
                .alias("ActionValuePerImp_Reference"),
            )
            .with_columns(
                CTR_Lift=(pl.col("CTR") - pl.col("CTR_Reference")) / pl.col("CTR"),
                Value_Lift=(
                    pl.col("ActionValuePerImp") - pl.col("ActionValuePerImp_Reference")
                )
                / pl.col("ActionValuePerImp"),
                # TODO confidence intervals and significance
            )
            .drop(cs.ends_with("_Reference"))
        )
