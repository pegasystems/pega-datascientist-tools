import os
from typing import Dict, List, Optional, Union
from datetime import datetime
import json

import polars as pl
import polars.selectors as cs

from .Plots import Plots
from ..utils.cdh_utils import _polars_capitalize, _apply_query, weighted_average_polars
from ..utils.types import QUERY
from ..pega_io.File import read_ds_export


class ImpactAnalyzer:
    ia_data: pl.LazyFrame

    default_ia_experiments = {
        # lists test and control groups for the default experiments
        "NBA vs Random": ("NBAHealth_NBAPrioritization", "NBAHealth_NBA"),
        "NBA vs Propensity Only": ("NBAHealth_PropensityPriority", "NBAHealth_NBA"),
        "NBA vs No Levers": ("NBAHealth_LeverPriority", "NBAHealth_NBA"),
        "NBA vs Only Eligibility Rules": (
            "NBAHealth_EngagementPolicy",
            "NBAHealth_NBA",
        ),
        "Adaptive Models vs Random Propensity": (
            # "NBAHealth_ModelControl",
            # "NBAHealth_PropensityPriority",
            "NBAHealth_ModelControl_1",
            "NBAHealth_ModelControl_2",
        ),
    }

    default_ia_controlgroups = {
        # ID of the control groups
        "MktValue": [
            "NBAHealth_NBAPrioritization",
            "NBAHealth_PropensityPriority",
            "NBAHealth_LeverPriority",
            "NBAHealth_EngagementPolicy",
            "NBAHealth_ModelControl_1",
            "NBAHealth_ModelControl_2",  # NBAHealth_ModelControl_2 is conceptually the same as NBAHealth_PropensityPriority and will be phased out in Pega 24.1/24.2.
            "NBAHealth_NBA",  # None in the VBD data
        ],
        # ID of the associated experiment
        "MktType": [
            "NBAPrioritization",
            "PropensityPriority",
            "LeverPriority",
            "EngagementPolicy",
            "ModelControl",
            None,
            None,
        ],
        "Description": [
            "Random eligible action (all engagement policies but randomly prioritized)",
            "Prioritized with model propensity only (no V, C or L)",
            "Prioritized with no levers (only p, V and C)",
            "Only Eligibility policies applied (no Applicability or Suitability, and prioritized with pVCL)",
            "Prioritized with Random (p) only",
            "Prioritized with model propensity only (no V, C or L)",
            "Arbitrated with your full NBA as configured",
        ],
    }

    def __init__(self, raw_data: pl.LazyFrame):
        self.plot = Plots(ia=self)

        # Column names may still be PDC specific, we should be changing once w got more sources going

        required_cols = [
            "SnapshotTime",
            "ControlGroup",
            "Impressions",
            "Accepts",
            "ValuePerImpression",
            "Channel",
            # and optionally more dimensions
        ]
        missing_cols = set(required_cols).difference(raw_data.collect_schema().names())
        if len(missing_cols) > 0:
            raise ValueError(f"Missing required inputs: {missing_cols}")

        self.ia_data = raw_data

        # .select(
        #     raw_data.collect_schema().names()[
        #         0 : raw_data.collect_schema().names().index("***") # hacky wacky way to exclude redundant columns
        #     ]
        # )

    @classmethod
    def from_pdc(
        cls,
        pdc_source: Union[os.PathLike, str, dict],
        *,
        query: Optional[QUERY] = None,
        return_input_df: Optional[bool] = False,
        return_df: Optional[bool] = False,
    ):
        """Create an ImpactAnalyzer instance from a PDC file

        Parameters
        ----------
        pdc_filename : Union[os.PathLike, str]
            The full path to the PDC file
        query : Optional[QUERY], optional
            An optional argument to filter out selected data, by default None
        return_input_df : Optional[QUERY], optional
            Debugging option to return the wide data from the raw JSON file as a DataFrame, by default False
        return_df : Optional[QUERY], optional
            Returns the processed input data as a DataFrame. Multiple of these can be stacked up and used to initialize the ImpactAnalyzer class, by default False

        Returns
        -------
        ImpactAnalyzer
            The properly initialized ImpactAnalyzer object

        """
        if isinstance(pdc_source, dict):
            return cls._from_pdc_json(
                pdc_source,
                query=query,
                return_input_df=return_input_df,
                return_df=return_df,
            )
        else:
            with open(pdc_source, encoding="utf-8") as pdc_json_data:
                return cls._from_pdc_json(
                    # TODO use read_ds_export/import_file from io lib for the first part 
                    json.load(pdc_json_data),
                    query=query,
                    return_input_df=return_input_df,
                    return_df=return_df,
                )

    @classmethod
    def _from_pdc_json(
        cls,
        json_data: dict,
        *,
        query: Optional[QUERY] = None,
        return_input_df: Optional[bool] = False,
        return_df: Optional[bool] = False,
    ):
        """Internal method to create an ImpactAnalyzer instance from PDC JSON data

        The PDC data is really structured as a list of expriments: control group A vs control group B. There
        is no explicit indicator whether the B's are really the same customers or not. The PDC data also contains
        a lot of UI related information that is not necessary.

        We turn this data into a series of control groups with just counts of impressions and accepts. This
        does need to assume a few implicit assumptions.
        """
        if len(json_data["pxResults"]) != 1:
            raise Exception("Expected just one result under 1st level pxResults.")
        # lets hope the time format is consistent!
        # can we use cdh_utils here?
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

        if return_input_df:
            return wide_data.drop(
                "Heading",
                "RunType",
                "ApplicationStack",
                "KeyIdentifier",
                # "ExperimentLabel",
                "pzInsKey",
                "Guidance",
                "pxObjClass",
                "Type",
                "ExperimentColor",
            )

        df = (
            wide_data.filter(pl.col("IsActive"))
            .filter(pl.col("ChannelName") != "All channels")
            .select(
                [
                    "SnapshotTime",
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
            # We're only using the most granular data, we will aggregate up ourselves
            .filter(AggregationFrequency="Daily")
            # Snapshot time is not set, overwrite with top level value
            .with_columns(pl.lit(date).cast(pl.Date).alias("SnapshotTime"))
            .with_columns(
                pl.when(LastDataReceived="Yesterday")
                .then(pl.col("SnapshotTime") - pl.duration(days=1))
                .otherwise(pl.col("SnapshotTime"))
                .alias("SnapshotTime")
            )
            .drop(["LastDataReceived", "AggregationFrequency"])
            .rename({"ChannelName": "Channel"})
        )

        nba_data = df.group_by(["SnapshotTime", "Channel"]).agg(
            ControlGroup=pl.lit("NBAHealth_NBA"),
            Impressions=pl.col("Impressions_NBA").max(),
            Accepts=pl.col("Accepts_NBA").top_k_by("Impressions_NBA", k=1).first(),
            # these two we can't currently reproduce from the data, for NBA they're 1.0 by definition
            ValuePerImpression=pl.lit(None).cast(pl.Float64),
            Pega_ValueLift=pl.lit(1.0),
            Pega_ValueLiftInterval=pl.lit(0.0),
        )

        # ModelControl conducts an experiment of random p vs model p. Model p was running as a separate control group
        # for this (ModelControl_2) but it really is the same as PropensityPriority. This has been changed in a later
        # product version. Here, we split the data and assign the test part to PropensityPriority. This should really
        # only be done up to certain versions. After that (when ModelControl_2 is dropped) the test group for this
        # experiment is not separate anymore and should not be added up.

        model_control_2_data = (
            df.filter(ExperimentName="NBAHealth_ModelControl")
            .group_by(["SnapshotTime", "Channel"])
            .agg(
                ControlGroup=pl.lit("NBAHealth_ModelControl_2"),
                Impressions=pl.col("Impressions_NBA").first(),
                Accepts=pl.col("Accepts_NBA").first(),
                # these two we can't currently reproduce from the data, for the reference group they're 1.0 by definition
                ValuePerImpression=pl.lit(None).cast(pl.Float64),
                Pega_ValueLift=pl.lit(1.0),
                Pega_ValueLiftInterval=pl.lit(0.0),
            )
        )

        other_data = (
            df.select(  # filter(pl.col("Experiment") != "NBAHealth_ModelControl")
                "SnapshotTime",
                "Channel",
                "ExperimentName",
                "Impressions_Control",
                "Accepts_Control",
                "ActionValuePerImp_Control",
                # these two we can't currently reproduce from the data as ActionValuePerImp is not set
                pl.col("ValueLift", "ValueLiftInterval").name.prefix("Pega_"),
                # # for debugging
                # pl.lit(None).alias("***"),
                # "Impressions_NBA",
                # "Accepts_NBA",
                # "ActionValuePerImp_NBA",
                # "AcceptRate_Control",
                # "AcceptRate_NBA",
                # "EngagementLift",
                # "EngagementLiftInterval",
                # "IsSignificant",
            )
            .rename(lambda x: x.removesuffix("_Control"))
            .rename(
                {
                    "ExperimentName": "ControlGroup",
                    "ActionValuePerImp": "ValuePerImpression",
                }
            )
            .with_columns(
                ValuePerImpression=pl.lit(None).cast(pl.Float64),
                ControlGroup=pl.when(pl.col("ControlGroup") == "NBAHealth_ModelControl")
                .then(pl.lit("NBAHealth_ModelControl_1"))
                .otherwise("ControlGroup"),
            )
        )

        result = (
            pl.concat(
                [nba_data, model_control_2_data, other_data],
                how="diagonal_relaxed",
            ).sort("SnapshotTime", "Channel", "ControlGroup")
            # .with_columns(pl.col("ControlGroup").str.strip_prefix("NBAHealth_")) # TODO lookup
        )

        if return_df:
            return result

        return ImpactAnalyzer(result)

    # TODO consider dates, output descriptions etc. just like ADMDatamart, Predictions etc.
    def summary_by_channel(self) -> pl.LazyFrame:
        """Summarization of the experiments in Impact Analyzer split by Channel.

        Parameters
        ----------

        Returns
        -------
        pl.LazyFrame
            Summary across all running Impact Analyzer experiments as a dataframe with the following fields:

            Channel Identification:
            - Channel: The channel name

            Performance Metrics:
            - CTR_Lift Adaptive Models vs Random Propensity: Lift in Engagement when testing prioritization with just Adaptive Models vs just Random Propensity
            - CTR_Lift NBA vs No Levers: Lift in Engagement for the full NBA Framework as configured vs prioritization without levers (only p, V and C)
            - CTR_Lift NBA vs Only Eligibility Rules: Lift in Engagement for the full NBA Framework as configured vs Only Eligibility policies applied (no Applicability or Suitability, and prioritized with pVCL)
            - CTR_Lift NBA vs Propensity Only: Lift in Engagement for the full NBA Framework as configured vs prioritization with model propensity only (no V, C or L)
            - CTR_Lift NBA vs Random: Lift in Engagement for the full NBA Framework as configured vs a Random eligible action (all engagement policies but randomly prioritized)
            - Value_Lift Adaptive Models vs Random Propensity: Lift in Expected Value when testing prioritization with just Adaptive Models vs just Random Propensity
            - Value_Lift NBA vs No Levers: Lift in Expected Value for the full NBA Framework as configured vs prioritization without levers (only p, V and C)
            - Value_Lift NBA vs Only Eligibility Rules: Lift in Expected Value for the full NBA Framework as configured vs Only Eligibility policies applied (no Applicability or Suitability, and prioritized with pVCL)
            - Value_Lift NBA vs Propensity Only: Lift in Expected Value for the full NBA Framework as configured vs prioritization with model propensity only (no V, C or L)
            - Value_Lift NBA vs Random: Lift in Expected Value for the full NBA Framework as configured vs a Random eligible action (all engagement policies but randomly prioritized)
        """

        return (
            self.summarize_experiments(by="Channel")
            .with_columns(Dummy=pl.lit(None))
            .collect()
            .pivot(
                on="Experiment",
                index="Channel",
                values=["CTR_Lift", "Value_Lift"],
                separator=" ",
            )
            .lazy()
        )

    # TODO consider dates, output descriptions etc. just like ADMDatamart, Predictions etc.

    def overall_summary(self) -> pl.LazyFrame:
        """Summarization of the experiments in Impact Analyzer.

        Parameters
        ----------

        Returns
        -------
        pl.LazyFrame
            Summary across all running Impact Analyzer experiments as a dataframe with the following fields:

            Performance Metrics:
            - CTR_Lift Adaptive Models vs Random Propensity: Lift in Engagement when testing prioritization with just Adaptive Models vs just Random Propensity
            - CTR_Lift NBA vs No Levers: Lift in Engagement for the full NBA Framework as configured vs prioritization without levers (only p, V and C)
            - CTR_Lift NBA vs Only Eligibility Rules: Lift in Engagement for the full NBA Framework as configured vs Only Eligibility policies applied (no Applicability or Suitability, and prioritized with pVCL)
            - CTR_Lift NBA vs Propensity Only: Lift in Engagement for the full NBA Framework as configured vs prioritization with model propensity only (no V, C or L)
            - CTR_Lift NBA vs Random: Lift in Engagement for the full NBA Framework as configured vs a Random eligible action (all engagement policies but randomly prioritized)
            - Value_Lift Adaptive Models vs Random Propensity: Lift in Expected Value when testing prioritization with just Adaptive Models vs just Random Propensity
            - Value_Lift NBA vs No Levers: Lift in Expected Value for the full NBA Framework as configured vs prioritization without levers (only p, V and C)
            - Value_Lift NBA vs Only Eligibility Rules: Lift in Expected Value for the full NBA Framework as configured vs Only Eligibility policies applied (no Applicability or Suitability, and prioritized with pVCL)
            - Value_Lift NBA vs Propensity Only: Lift in Expected Value for the full NBA Framework as configured vs prioritization with model propensity only (no V, C or L)
            - Value_Lift NBA vs Random: Lift in Expected Value for the full NBA Framework as configured vs a Random eligible action (all engagement policies but randomly prioritized)
        """
        return (
            self.summarize_experiments()
            .with_columns(Dummy=pl.lit(None))
            .collect()
            .pivot(
                on="Experiment",
                index="Dummy",
                values=["CTR_Lift", "Value_Lift"],
                separator=" ",
            )
            .drop("Dummy")
            .lazy()
        )

    def summarize_control_groups(
        self, by: Optional[Union[List[str], str]] = None, drop_internal_cols=True
    ) -> pl.LazyFrame:
        if not by:
            by = []
        if isinstance(by, str):
            by = [by]
        return (
            self.ia_data.sort(by + ["ControlGroup"])
            .group_by(by + ["ControlGroup"], maintain_order=True)
            .agg(
                pl.sum("Impressions", "Accepts"),
                CTR=pl.sum("Accepts") / pl.sum("Impressions"),
                ValuePerImpression=weighted_average_polars(
                    "ValuePerImpression", "Impressions"
                ),
                # this is only a backup in case the associated value per impression is missing like in PDC data
                Pega_ValueLift=weighted_average_polars("Pega_ValueLift", "Impressions"),
            )
            .drop(["Pega_ValueLift"] if drop_internal_cols else [])
        )

    def summarize_experiments(
        self, by: Optional[Union[List[str], str]] = None
    ) -> pl.LazyFrame:
        if not by:
            by = []
        if isinstance(by, str):
            by = [by]

        def _lift_pl(test, control):
            return (pl.col(test) - pl.col(control)) / pl.col(control)

        control_groups_summary = self.summarize_control_groups(
            by, drop_internal_cols=False
        )

        return (
            pl.LazyFrame(
                {
                    "Experiment": ImpactAnalyzer.default_ia_experiments.keys(),
                    "Test": [
                        v[1] for v in ImpactAnalyzer.default_ia_experiments.values()
                    ],
                    "Control": [
                        v[0] for v in ImpactAnalyzer.default_ia_experiments.values()
                    ],
                }
            )
            .join(
                control_groups_summary.select(*by, pl.exclude(by).name.suffix("_Test")),
                how="left",
                left_on="Test",
                right_on="ControlGroup_Test",
            )
            .join(
                control_groups_summary.select(
                    *by, pl.exclude(by).name.suffix("_Control")
                ),
                how="left",
                left_on=["Control"] + by,
                right_on=["ControlGroup_Control"] + by,
            )
            .with_columns(
                Control_Fraction=pl.col("Impressions_Control")
                / (pl.col("Impressions_Control") + pl.col("Impressions_Test")),
                CTR_Lift=_lift_pl(
                    "CTR_Test", "CTR_Control"
                ),  # TODO I got myself confused now
                # this is temp and should only be tried when the value per impression is missing like in PDC data
                Value_Lift=pl.col("Pega_ValueLift_Control"),
                # TODO figure out confidence intervals etc.
            )
            .drop(cs.starts_with("Pega_"))
            .sort(["Experiment"] + by)
        )
