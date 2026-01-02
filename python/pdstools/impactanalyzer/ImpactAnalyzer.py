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
    Analyze and visualize Impact Analyzer experiment results from Pega CDH.

    The ImpactAnalyzer class provides analysis and visualization capabilities
    for NBA (Next-Best-Action) Impact Analyzer experiments. It processes experiment
    data from Pega's Customer Decision Hub to compare the effectiveness of different
    NBA strategies including adaptive models, propensity prioritization, lever usage,
    and engagement policies.

    Data can be loaded from three sources:

    - **PDC exports** via :meth:`from_pdc`: Uses pre-aggregated experiment data from
      PDC JSON exports. Value Lift is copied from PDC data as it cannot be
      re-calculated from the available numbers.
    - **VBD exports** via :meth:`from_vbd`: Reconstructs experiment metrics from raw
      VBD Actuals or Scenario Planner Actuals data. Allows flexible time ranges and
      data selection. Value Lift is calculated from ValuePerImpression.
    - **Interaction History** via :meth:`from_ih`: Loads experiment metrics from
      Interaction History data. Not yet implemented.

    .. math::

        \\text{Engagement Lift} = \\frac{\\text{SuccessRate}_{test} - \\text{SuccessRate}_{control}}{\\text{SuccessRate}_{control}}

    .. math::

        \\text{Value Lift} = \\frac{\\text{ValueCapture}_{test} - \\text{ValueCapture}_{control}}{\\text{ValueCapture}_{control}}

    Attributes
    ----------
    ia_data : pl.LazyFrame
        The underlying experiment data containing control group metrics.
    plot : Plots
        Plot accessor for visualization methods.

    See Also
    --------
    pdstools.adm.ADMDatamart : For ADM model analysis.
    pdstools.ih.IH : For Interaction History analysis.

    Examples
    --------
    >>> from pdstools import ImpactAnalyzer
    >>> ia = ImpactAnalyzer.from_pdc("impact_analyzer_export.json")
    >>> ia.overall_summary().collect()
    >>> ia.plot.overview()
    """

    ia_data: pl.LazyFrame

    default_ia_experiments = {
        "NBA vs Random": ("NBAPrioritization", "NBA"),
        "NBA vs Propensity Only": ("PropensityPriority", "NBA"),
        "NBA vs No Levers": ("LeverPriority", "NBA"),
        "NBA vs Only Eligibility Rules": (
            "EngagementPolicy",
            "NBA",
        ),
        "Adaptive Models vs Random Propensity": ("ModelControl_1", "ModelControl_2"),
    }
    """Default experiments mapping experiment names to (control, test) group tuples."""

    outcome_labels = {
        "Impressions": ["Impression"],
        "Accepts": ["Accept", "Accepted", "Click", "Clicked"],
    }
    """Mapping of metric names to outcome labels used for aggregation."""

    default_ia_controlgroups = {
        "MktValue": [
            "NBAHealth_NBAPrioritization",
            "NBAHealth_PropensityPriority",
            "NBAHealth_LeverPriority",
            "NBAHealth_EngagementPolicy",
            "NBAHealth_ModelControl_1",
            "NBAHealth_ModelControl_2",
            "NBAHealth_NBA",
        ],
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
        """Initialize an ImpactAnalyzer instance.

        Parameters
        ----------
        raw_data : pl.LazyFrame
            Pre-processed experiment data containing control group metrics.
            Must include columns: SnapshotTime, ControlGroup, Impressions,
            Accepts, ValuePerImpression, Channel.

        Raises
        ------
        ValueError
            If required columns are missing from the input data.

        Notes
        -----
        Use the class methods :meth:`from_pdc`, :meth:`from_vbd`, or :meth:`from_ih`
        to create instances from raw data exports.
        """
        self.plot = Plots(ia=self)

        required_cols = [
            "SnapshotTime",
            "ControlGroup",
            "Impressions",
            "Accepts",
            "ValuePerImpression",
            "Channel",
        ]
        missing_cols = set(required_cols).difference(raw_data.collect_schema().names())
        if len(missing_cols) > 0:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.ia_data = raw_data

    @classmethod
    def from_pdc(
        cls,
        pdc_source: Union[os.PathLike, str, List[os.PathLike], List[str]],
        *,
        reader: Optional[Callable] = None,
        query: Optional[QUERY] = None,
        return_wide_df: bool = False,
        return_df: bool = False,
    ) -> Union["ImpactAnalyzer", pl.LazyFrame]:
        """Create an ImpactAnalyzer instance from PDC JSON export(s).

        Loads pre-aggregated experiment data from Pega Decision Central JSON exports.
        Value Lift metrics are copied directly from the PDC data.

        Parameters
        ----------
        pdc_source : Union[os.PathLike, str, List[os.PathLike], List[str]]
            Path to PDC JSON file, or a list of paths to concatenate.
        reader : Optional[Callable], optional
            Custom function to read source data into a dict. If None, uses
            standard JSON file reader. Default is None.
        query : Optional[QUERY], optional
            Polars expression to filter the data. Default is None.
        return_wide_df : bool, optional
            If True, return the raw wide-format data as a LazyFrame for
            debugging. Default is False.
        return_df : bool, optional
            If True, return the processed data as a LazyFrame instead of
            an ImpactAnalyzer instance. Default is False.

        Returns
        -------
        ImpactAnalyzer or pl.LazyFrame
            ImpactAnalyzer instance, or LazyFrame if return_df or return_wide_df
            is True.

        Raises
        ------
        ValueError
            If an empty list of source files is provided.

        Examples
        --------
        >>> ia = ImpactAnalyzer.from_pdc("CDH_Metrics_ImpactAnalyzer.json")
        >>> ia.overall_summary().collect()
        """

        def default_reader(f):
            with open(f, encoding="utf-8") as pdc_json_data:
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
        return_df: bool = False,
    ) -> Union["ImpactAnalyzer", pl.LazyFrame, None]:
        """Create an ImpactAnalyzer instance from VBD data.

        Processes VBD Actuals or Scenario Planner Actuals data to reconstruct
        Impact Analyzer experiment metrics. Provides more flexible time ranges
        and data selection compared to PDC exports.

        Value Lift is calculated from ValuePerImpression since raw value data
        is available in VBD exports.

        Parameters
        ----------
        vbd_source : Union[os.PathLike, str]
            Path to VBD export file (parquet, csv, ndjson, or zip).
        return_df : bool, optional
            If True, return processed data as LazyFrame instead of
            ImpactAnalyzer instance. Default is False.

        Returns
        -------
        ImpactAnalyzer or pl.LazyFrame or None
            ImpactAnalyzer instance, LazyFrame if return_df is True,
            or None if the source contains no data.

        Examples
        --------
        >>> ia = ImpactAnalyzer.from_vbd("ScenarioPlannerActuals.zip")
        >>> ia.summary_by_channel().collect()
        """

        vbd_data = read_ds_export(str(Path(vbd_source).absolute()))
        if vbd_data is None:
            return None

        ia_data = (
            _polars_capitalize(vbd_data)
            .with_columns(
                SnapshotTime=parse_pega_date_time_formats("OutcomeTime").dt.truncate(
                    "1d"
                ),
                Channel=pl.concat_str(
                    "Channel", "Direction", separator="/", ignore_nulls=True
                ),
            )
            .group_by(
                "SnapshotTime",
                "MktValue",
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
                .filter(pl.col("Outcome").is_in(cls.outcome_labels["Impressions"]))
                .sum()
                .alias("Impressions"),
                pl.col("AggregateCount")
                .filter(pl.col("Outcome").is_in(cls.outcome_labels["Accepts"]))
                .sum()
                .alias("Accepts"),
                (
                    pl.col("Value")
                    .filter(pl.col("Outcome").is_in(cls.outcome_labels["Accepts"]))
                    .sum()
                    / (
                        pl.col("AggregateCount")
                        .filter(
                            pl.col("Outcome").is_in(cls.outcome_labels["Impressions"])
                        )
                        .sum()
                    )
                ).alias("ValuePerImpression"),
                # rest just for debugging
                pl.col("AggregateCount", "Value", "Outcome"),
            )
            .filter(pl.col("Accepts") <= pl.col("Impressions"))
            .rename({"MktValue": "ControlGroup"})
            .with_columns(
                pl.col("ControlGroup").str.strip_prefix("NBAHealth_").fill_null("NBA")
            )
            .sort(
                "SnapshotTime",
                "Channel",
                "Issue",
                "Group",
                "Name",
                "Treatment",
                "ControlGroup",
            )
        )

        if return_df:
            return ia_data

        return ImpactAnalyzer(ia_data)

    @classmethod
    def from_ih(
        cls,
        ih_source: Union[os.PathLike, str],
        *,
        return_df: bool = False,
    ) -> Union["ImpactAnalyzer", pl.LazyFrame, None]:
        """Create an ImpactAnalyzer instance from Interaction History data.

        .. note::
            This method is not yet implemented.

        Reconstructs experiment metrics from Interaction History data, allowing
        analysis of experiments using detailed interaction-level records.

        Parameters
        ----------
        ih_source : Union[os.PathLike, str]
            Path to Interaction History export file.
        return_df : bool, optional
            If True, return processed data as LazyFrame instead of
            ImpactAnalyzer instance. Default is False.

        Returns
        -------
        ImpactAnalyzer or pl.LazyFrame or None
            ImpactAnalyzer instance, LazyFrame if return_df is True,
            or None if the source contains no data.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("from_ih is not yet implemented")

    @classmethod
    def _normalize_pdc_ia_data(
        cls,
        json_data: dict,
        *,
        query: Optional[QUERY] = None,
        return_wide_df: bool = False,
    ) -> pl.LazyFrame:
        """Transform PDC Impact Analyzer JSON into normalized long format.

        Converts the hierarchical PDC JSON structure (organized by experiments)
        into a flat structure organized by control groups with impression and
        accept counts.

        Parameters
        ----------
        json_data : dict
            Parsed JSON data from PDC export.
        query : Optional[QUERY], optional
            Polars expression to filter the data. Default is None.
        return_wide_df : bool, optional
            If True, return intermediate wide-format data. Default is False.

        Returns
        -------
        pl.LazyFrame
            Normalized data with columns: SnapshotTime, Channel, ControlGroup,
            Impressions, Accepts, ValuePerImpression, Pega_ValueLift.
        """
        if len(json_data["pxResults"]) != 1:
            raise ValueError("Expected exactly one result under pxResults.")

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
            )
            .with_columns(pl.col("ControlGroup").str.strip_prefix("NBAHealth_"))
            .sort("SnapshotTime", "Channel", "ControlGroup")
        )

        return result

    def summary_by_channel(self) -> pl.LazyFrame:
        """Get experiment summary pivoted by channel.

        Returns experiment lift metrics (CTR_Lift and Value_Lift) for each
        experiment, with one row per channel.

        Returns
        -------
        pl.LazyFrame
            Wide-format summary with columns:

            - **Channel**: Channel name
            - **CTR_Lift <Experiment>**: Engagement lift for each experiment
            - **Value_Lift <Experiment>**: Value lift for each experiment

        See Also
        --------
        overall_summary : Summary without channel breakdown.
        summarize_experiments : Long-format experiment summary.

        Examples
        --------
        >>> ia.summary_by_channel().collect()
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

    def overall_summary(self) -> pl.LazyFrame:
        """Get overall experiment summary aggregated across all channels.

        Returns experiment lift metrics (CTR_Lift and Value_Lift) for each
        experiment, aggregated across all data.

        Returns
        -------
        pl.LazyFrame
            Single-row wide-format summary with columns:

            - **CTR_Lift <Experiment>**: Engagement lift for each experiment
            - **Value_Lift <Experiment>**: Value lift for each experiment

        See Also
        --------
        summary_by_channel : Summary with channel breakdown.
        summarize_experiments : Long-format experiment summary.

        Examples
        --------
        >>> ia.overall_summary().collect()
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
        drop_internal_cols: bool = True,
    ) -> pl.LazyFrame:
        """Aggregate metrics by control group.

        Summarizes impressions, accepts, CTR, and value metrics for each
        control group, optionally grouped by additional dimensions.

        Parameters
        ----------
        by : Optional[Union[List[str], List[pl.Expr], str, pl.Expr]], optional
            Column name(s) or expression(s) to group by in addition to
            ControlGroup. Default is None (aggregate all data).
        drop_internal_cols : bool, optional
            If True, drop internal columns prefixed with 'Pega_'.
            Default is True.

        Returns
        -------
        pl.LazyFrame
            Aggregated metrics with columns: ControlGroup, Impressions,
            Accepts, CTR, ValuePerImpression, plus any grouping columns.

        Examples
        --------
        >>> ia.summarize_control_groups().collect()
        >>> ia.summarize_control_groups(by="Channel").collect()
        """
        if not by:
            group_by = []
        elif not isinstance(by, list):
            group_by = [by]
        else:
            group_by = by

        agg_exprs = [
            pl.sum("Impressions", "Accepts"),
            (pl.sum("Accepts") / pl.sum("Impressions")).alias("CTR"),
            weighted_average_polars("ValuePerImpression", "Impressions").alias(
                "ValuePerImpression"
            ),
        ]

        # Pega_ValueLift is only present in PDC data, not in VBD data
        if "Pega_ValueLift" in self.ia_data.collect_schema().names():
            agg_exprs.append(
                weighted_average_polars("Pega_ValueLift", "Impressions").alias(
                    "Pega_ValueLift"
                )
            )

        return (
            self.ia_data.sort(group_by + ["ControlGroup"])
            .group_by(group_by + ["ControlGroup"], maintain_order=True)
            .agg(agg_exprs)
            .drop(cs.starts_with("Pega_") if drop_internal_cols else [])
        )

    def summarize_experiments(
        self,
        by: Optional[Union[List[str], List[pl.Expr], str, pl.Expr]] = None,
    ) -> pl.LazyFrame:
        """Summarize experiment metrics comparing test vs control groups.

        Computes lift metrics for each defined experiment by comparing
        test and control group performance.

        .. note::
            Returns all default experiments regardless of whether they are
            active in the data. Experiments without data will have null values
            for all metrics (Impressions, Accepts, CTR_Lift, Value_Lift, etc.).

        Parameters
        ----------
        by : Optional[Union[List[str], List[pl.Expr], str, pl.Expr]], optional
            Column name(s) or expression(s) to group by. Default is None
            (aggregate all data).

        Returns
        -------
        pl.LazyFrame
            Experiment summary with columns:

            - **Experiment**: Experiment name
            - **Test**, **Control**: Control group names for the experiment
            - **Impressions_Test**, **Impressions_Control**: Impression counts (null if not active)
            - **Accepts_Test**, **Accepts_Control**: Accept counts (null if not active)
            - **CTR_Test**, **CTR_Control**: Click-through rates (null if not active)
            - **Control_Fraction**: Fraction of impressions in control group
            - **CTR_Lift**: Engagement lift (null if experiment not active)
            - **Value_Lift**: Value lift (null if experiment not active)

        See Also
        --------
        summarize_control_groups : Lower-level control group aggregation.
        overall_summary : Pivoted overall summary.
        summary_by_channel : Pivoted summary by channel.

        Examples
        --------
        >>> ia.summarize_experiments().collect()
        >>> ia.summarize_experiments(by="Channel").collect()
        """
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

        has_pega_value_lift = "Pega_ValueLift" in self.ia_data.collect_schema().names()

        result = (
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
                CTR_Lift=_lift_pl("CTR_Test", "CTR_Control"),
            )
        )

        # Value_Lift from Pega_ValueLift is only available in PDC data
        if has_pega_value_lift:
            result = result.with_columns(Value_Lift=pl.col("Pega_ValueLift_Control"))
        else:
            # For VBD data, calculate Value_Lift from ValuePerImpression
            result = result.with_columns(
                Value_Lift=_lift_pl(
                    "ValuePerImpression_Test", "ValuePerImpression_Control"
                )
            )

        return result.drop(cs.starts_with("Pega_")).sort(
            ["Experiment"] + by_column_names
        )
