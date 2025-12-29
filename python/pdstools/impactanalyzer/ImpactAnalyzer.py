from pathlib import Path
import os
from typing import Callable, Dict, List, Optional, Union
from datetime import datetime
import json

import polars as pl
import polars.selectors as cs

from .Plots import Plots
from ..utils.cdh_utils import (
    _apply_query,
    weighted_average_polars,
    _polars_capitalize,
    parse_pega_date_time_formats,
)
from ..pega_io.File import read_ds_export
from ..utils.types import QUERY


class ImpactAnalyzer:
    """
    Analyze and visualize Impact Analyzer experiment results from Pega Customer Decision Hub.

    The ImpactAnalyzer class provides comprehensive analysis and visualization capabilities
    for NBA (Next-Best-Action) Impact Analyzer experiments. It processes experiment data
    from Pega's Customer Decision Hub to compare the effectiveness of different NBA strategies
    including adaptive models, propensity prioritization, lever usage, and engagement policies.

    When reading from PDC, the ImpactAnalyzer class only keeps the counts of impressions, accepts
    and the action value per impression and re-calculates all the derived values on demand. It
    drops inactive experiments and adds rows for the "NBA" group. The "All channels" is dropped.
    ValueLift and ValueLiftInterval are copied from the PDC data as this can currently not be
    re-calculated from the available raw numbers (ValuePerImpression is empty).

    Engagement Lift is calculated as (SuccessRate(test) - SuccessRate(control))/SuccessRate(control)

    Value Lift is calculated as (ValueCapture(test) - ValueCapture(control))/ValueCapture(control)

    For value, aggregting the Value property

    """

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
        pdc_source: Union[os.PathLike, str, List[os.PathLike], List[str]],
        *,
        reader: Optional[Callable] = None,
        query: Optional[QUERY] = None,
        return_wide_df: Optional[bool] = False,
        return_df: Optional[bool] = False,
    ):
        """Create an ImpactAnalyzer instance from a PDC file

        Parameters
        ----------
        pdc_source : Union[os.PathLike, str, List[os.PathLike], List[str]]
            The full path to the PDC file, or a list of such paths
        reader: Optional[Callable]
            Function to read the source data into a dict. If None uses standard file reader.
        query : Optional[QUERY], optional
            An optional argument to filter out selected data, by default None
        return_wide_df : Optional[QUERY], optional
            Debugging option to return the wide data from the raw JSON file as a LazyFrame, for debugging, by default False
        return_df : Optional[QUERY], optional
            Returns the processed input data as a LazyFrame. Multiple of these can be stacked up and used to initialize the ImpactAnalyzer class, by default False

        Returns
        -------
        ImpactAnalyzer
            The properly initialized ImpactAnalyzer object

        """

        def default_reader(f):
            with open(f, encoding="utf-8") as pdc_json_data:
                # TODO use read_ds_export/import_file from io lib for the first part
                return json.load(pdc_json_data)

        if reader is None:
            reader = default_reader

        if isinstance(pdc_source, list):
            all_json_data = [reader(src) for src in pdc_source]
            if not all_json_data:
                raise ValueError("Empty list of source data")
            normalized_ia_data = pl.concat(
                [
                    cls._normalize_pdc_ia_data(
                        json_data,
                        query=query,
                        return_wide_df=return_wide_df,
                    )
                    for json_data in all_json_data  # if json_data.height
                ],
                how="diagonal_relaxed",
            ).lazy()
        else:
            json_data = reader(pdc_source)
            normalized_ia_data = cls._normalize_pdc_ia_data(
                json_data,
                query=query,
                return_wide_df=return_wide_df,
            )
        if return_wide_df or return_df:
            return normalized_ia_data

        return ImpactAnalyzer(normalized_ia_data)

    @classmethod
    def from_vbd(
        cls,
        vbd_source: Union[os.PathLike, str],
        *,
        return_df: Optional[bool] = False,
    ):
        """Create an ImpactAnalyzer instance from VBD (Value-Based Decisioning) data

        This method will process VBD Actuals or VBD Scenario Planner Actuals data
        to reconstruct Impact Analyzer experiment metrics. This allows for more
        flexible time ranges and data selection compared to PDC exports.

        IA uses **pyReason**, **MktType**, **MktValue** and **ModelControlGroup** to define
        the various experiments. For the standard NBA decisions (no experiment), values are left empty (null).

        Prior to Impact Analyzer, or when turned off, Predictions from Prediction Studio manage two
        groups through the **ModelControlGroup** property. A value of **Test** is used for model driven arbitration, **Control** for the random control group (defaults to 2%).

        When IA is on, the distinct values from just **MktValue** are sufficient to identify the
        different experiments. In the future, more and custom experiments may be supported.

        For the full NBA interactions the value of the marker fields is left empty.

        TODO: NBAHealth_ModelControl_2 is conceptually the same as NBAHealth_PropensityPriority and will be phased out in Pega 24.1/24.2.

        The usage of "Default" issues and groups indicates that there is no action. These need to be filtered out for proper reporting.

        TODO: should we exclude these from analysis?

        TODO: what about things with inactive status? And how can we know?

        NOTE: Impact Analyzer goes back from today's date, also when the data is from an earlier date.

        Parameters
        ----------
        vbd_source : Union[os.PathLike, str, List[os.PathLike], List[str]]
            Path to VBD export file(s) or URL(s)
        return_df : Optional[bool], optional
            Return processed data instead of ImpactAnalyzer instance, by default False

        Returns
        -------
        ImpactAnalyzer
            The properly initialized ImpactAnalyzer object with reconstructed experiment data


        Examples
        --------
        >>> # Load from VBD export
        >>> ia = ImpactAnalyzer.from_vbd('Data-pxStrategyResult_ActualsExport.zip')

        Raises
        ------
        NotImplementedError
            This method is not yet implemented. Use from_pdc() for current functionality.
        """

        vbd_data = read_ds_export(str(Path(vbd_source).absolute()))
        if vbd_data is None:
            return None

        # TODO share with IH class
        pos_outcomes = ["Accept", "Accepted", "Click", "Clicked"]
        imp_outcomes = ["Impression"]

        ia_data = (
            _polars_capitalize(vbd_data)
            .with_columns(
                SnapshotTime=parse_pega_date_time_formats("OutcomeTime").dt.truncate(
                    "1h"
                ),
                Channel=pl.concat_str(
                    "Channel", "Direction", separator="/", ignore_nulls=True
                ),
            )
            .group_by(
                "SnapshotTime",
                "MktValue",
                "Reason",
                "MktType",
                "Application",
                "ApplicationVersion",
                "Channel",
                "Issue",
                "Group",
                "Name",
                "Treatment",
            )
            .agg(
                pl.col("AggregateCount")
                .filter(pl.col("Outcome").is_in(imp_outcomes))
                .sum()
                .alias("Impressions"),
                pl.col("AggregateCount")
                .filter(pl.col("Outcome").is_in(pos_outcomes))
                .sum()
                .alias("Accepts"),
                (
                    pl.col("Value").filter(pl.col("Outcome").is_in(pos_outcomes)).sum()
                    / (
                        pl.col("AggregateCount")
                        .filter(pl.col("Outcome").is_in(imp_outcomes))
                        .sum()
                    )
                ).alias("ValuePerImpression"),
                # rest just for analysis
                pl.col("AggregateCount", "Value", "Outcome"),
            )
            .filter(pl.col("Accepts") <= pl.col("Impressions"))
            .sort(
                "SnapshotTime",
                "Channel",
                "Issue",
                "Group",
                "Name",
                "Treatment",
                "MktValue",
                "Reason",
                "MktType",
            )
            .rename({"MktValue": "ControlGroup"})
        )

        return ImpactAnalyzer(ia_data)

    @classmethod
    def _normalize_pdc_ia_data(
        cls,
        json_data: dict,
        *,
        query: Optional[QUERY] = None,
        return_wide_df: Optional[bool] = False,
    ):
        """Internal method to turn PDC Impact Analyzer JSON data into a proper long format

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
        actual_ia_data: Dict = json_data["pxResults"][0]["pxResults"]
        if len(actual_ia_data) < 1:
            # No data
            return pl.LazyFrame()

        wide_data = pl.DataFrame(actual_ia_data).lazy()

        if query is not None:
            wide_data = _apply_query(wide_data, query=query)

        wide_data = wide_data.drop(
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
            "SnapshotTime",
        )

        if return_wide_df:
            return wide_data

        df = (
            wide_data.filter(pl.col("IsActive"))
            .filter(pl.col("ChannelName") != "All channels")
            .select(
                [
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

        return result

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
        self,
        by: Optional[Union[List[str], List[pl.Expr], str, pl.Expr]] = None,
        drop_internal_cols=True,
    ) -> pl.LazyFrame:
        if not by:
            group_by = []
        elif not isinstance(by, list):
            group_by = [by]
        else:
            group_by = by
        return (
            self.ia_data.sort(group_by + ["ControlGroup"])
            .group_by(group_by + ["ControlGroup"], maintain_order=True)
            .agg(
                pl.sum("Impressions", "Accepts"),
                CTR=pl.sum("Accepts") / pl.sum("Impressions"),
                ValuePerImpression=weighted_average_polars(
                    "ValuePerImpression", "Impressions"
                ),
                # this is only a backup in case the associated value per impression is missing like in PDC data
                Pega_ValueLift=weighted_average_polars("Pega_ValueLift", "Impressions"),
            )
            .drop(cs.starts_with("Pega_") if drop_internal_cols else [])
        )

    def summarize_experiments(
        self,
        by: Optional[Union[List[str], List[pl.Expr], str, pl.Expr]] = None,
    ) -> pl.LazyFrame:
        if not by:
            by = []
        if isinstance(by, str):
            by = [by]

        def _lift_pl(test, control):
            return (pl.col(test) - pl.col(control)) / pl.col(control)

        # Extract column names from expressions for use with pl.exclude()
        def _get_column_names(by_list):
            column_names = []
            for item in by_list:
                if isinstance(item, pl.Expr):
                    # Extract the root column name from the expression
                    column_names.append(item.meta.output_name())
                else:
                    column_names.append(item)
            return column_names

        by_column_names = _get_column_names(by)

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
                control_groups_summary.select(
                    *by, pl.exclude(by_column_names).name.suffix("_Test")
                ),
                how="left",
                left_on="Test",
                right_on="ControlGroup_Test",
            )
            .join(
                control_groups_summary.select(
                    *by, pl.exclude(by_column_names).name.suffix("_Control")
                ),
                how="left",
                left_on=["Control"] + by_column_names,
                right_on=["ControlGroup_Control"] + by_column_names,
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
            .sort(["Experiment"] + by_column_names)
        )
