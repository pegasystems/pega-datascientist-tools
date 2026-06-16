from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal, overload, TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from ..pega_io.File import read_ds_export
from ..utils.cdh_utils import (
    _apply_query,
    _apply_schema_types,
    _polars_capitalize,
    parse_pega_date_time_formats,
    weighted_average_polars,
)
from ..utils.pega_outcomes import resolve_outcome_labels as _resolve_outcome_labels
from .Plots import Plots
from .Schema import REQUIRED_IA_COLUMNS, ImpactAnalyzerData
from .statistics import lift_pl

if TYPE_CHECKING:
    from ..utils.types import QUERY
    from collections.abc import Callable, Sequence
    import os

logger = logging.getLogger(__name__)


class ImpactAnalyzer:
    """Analyze and visualize Impact Analyzer experiment results from Pega CDH.

    The ImpactAnalyzer class provides analysis and visualization capabilities
    for NBA (Next-Best-Action) Impact Analyzer experiments. It processes experiment
    data from Pega's Customer Decision Hub to compare the effectiveness of different
    NBA strategies including adaptive models, propensity prioritization, lever usage,
    and engagement policies.

    Data can be loaded from four sources:

    - **PDC exports** via :meth:`from_pdc`: Uses pre-aggregated experiment data from
      PDC JSON exports. Value Lift is copied from PDC data as it cannot be
      re-calculated from the available numbers.
    - **Pega Infinity Impact Analyzer Excel export** via :meth:`from_excel`:
      Reads the ``Data`` sheet of the ``.xlsx`` file produced by the Impact
      Analyzer landing page in Pega Infinity. Pre-paired Test vs Control
      counts are exploded to long form and NBA traffic is deduplicated
      across experiments.
    - **VBD exports** via :meth:`from_vbd`: Reconstructs experiment metrics from raw
      VBD Actuals or Scenario Planner Actuals data. Allows flexible time ranges and
      data selection. Value Lift is calculated from ValuePerImpression.
    - **Interaction History** via :meth:`from_ih`: Loads experiment metrics from
      Interaction History data. Not yet implemented.

    .. math::

        \\text{Engagement Lift} = \\frac{\\text{SuccessRate}_{test} - \\text{SuccessRate}_{control}}{\\text{SuccessRate}_{control}}

    .. math::

        \\text{Value Lift} = \\frac{\\text{ValueCapture}_{test} - \\text{ValueCapture}_{control}}{\\text{ValueCapture}_{control}}

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
    """The underlying experiment data containing control group metrics."""

    outcome_labels_used: dict | None
    plot: Plots
    """Plot accessor for visualization methods."""

    default_ia_experiments: ClassVar[dict[str, tuple[str, str]]] = {
        "NBA vs Random Relevant Action": ("NBAPrioritization", "NBA"),
        "NBA vs Arbitrating by Propensity-only": ("PropensityPriority", "NBA"),
        "NBA vs Arbitrating with No Levers": ("LeverPriority", "NBA"),
        "NBA vs Only Eligibility Criteria": (
            "EngagementPolicy",
            "NBA",
        ),
        "Adaptive Model Propensity vs Random Propensity": (
            "ModelControl_1",
            "ModelControl_2",
        ),
    }
    """Default experiments mapping experiment names to (control, test) group tuples.

    Names and ordering match the Pega Infinity Impact Analyzer product UI.
    Insertion order is the canonical display order — see
    :meth:`summarize_experiments` for how it is preserved through aggregation."""

    outcome_labels: ClassVar[dict[str, list[str]]] = {
        "Impressions": ["Impression"],
        "Accepts": ["Accept", "Accepted", "Click", "Clicked"],
    }
    """Mapping of metric names to outcome labels used for aggregation."""

    default_ia_controlgroups: ClassVar[dict[str, list[str | None]]] = {
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
        self.ia_data = self._validate_ia_data(raw_data)
        self.outcome_labels_used = None

    @staticmethod
    def _validate_ia_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """Validate that the input frame has the required Impact Analyzer columns.

        Mirrors the ``_validate_*_data`` pattern used by
        :class:`pdstools.adm.ADMDatamart`.

        Parameters
        ----------
        df : pl.LazyFrame
            Pre-processed experiment data.

        Returns
        -------
        pl.LazyFrame
            The same frame, unchanged, once validation passes.

        Raises
        ------
        ValueError
            If any column listed in
            :data:`pdstools.impactanalyzer.Schema.REQUIRED_IA_COLUMNS` is absent.
        """
        missing_cols = set(REQUIRED_IA_COLUMNS).difference(df.collect_schema().names())
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return _apply_schema_types(df, ImpactAnalyzerData)

    # When return_wide_df=True, always returns LazyFrame
    @classmethod
    @overload
    def from_pdc(
        cls,
        pdc_source: str | Path | os.PathLike | list[str] | list[Path] | list[os.PathLike],
        *,
        reader: Callable | None = None,
        query: QUERY | None = None,
        return_wide_df: Literal[True],
        return_df: bool = ...,
    ) -> pl.LazyFrame: ...

    # When return_df=True, always returns LazyFrame
    @classmethod
    @overload
    def from_pdc(
        cls,
        pdc_source: str | Path | os.PathLike | list[str] | list[Path] | list[os.PathLike],
        *,
        reader: Callable | None = None,
        query: QUERY | None = None,
        return_wide_df: Literal[False] = ...,
        return_df: Literal[True],
    ) -> pl.LazyFrame: ...

    # Default case: when both are False or not provided, returns ImpactAnalyzer
    @classmethod
    @overload
    def from_pdc(
        cls,
        pdc_source: str | Path | os.PathLike | list[str] | list[Path] | list[os.PathLike],
        *,
        reader: Callable | None = None,
        query: QUERY | None = None,
    ) -> "ImpactAnalyzer": ...

    @classmethod
    def from_pdc(
        cls,
        pdc_source: str | Path | os.PathLike | list[str] | list[Path] | list[os.PathLike],
        *,
        reader: Callable | None = None,
        query: QUERY | None = None,
        return_wide_df: bool = False,
        return_df: bool = False,
    ) -> "ImpactAnalyzer | pl.LazyFrame":
        """Create an ImpactAnalyzer instance from PDC JSON export(s).

        Loads pre-aggregated experiment data from Pega Decision Central JSON exports.
        Value Lift metrics are copied directly from the PDC data.

        Parameters
        ----------
        pdc_source : Union[Path, str, os.PathLike, list[Union[Path, str, os.PathLike]]]
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
    def _build_outcome_filter(
        cls,
        metric: str,
        outcome_labels: dict | None,
    ) -> pl.Expr:
        """Return a boolean Polars expression for the given metric.

        Parameters
        ----------
        metric : str
            "Impressions" or "Accepts"
        outcome_labels : dict or None
            ``{"Impressions": [...], "Accepts": [...]}`` → global override applied to all channels.
            ``{"Channel/Dir": {"Impressions": [...], "Accepts": [...]}, ...}`` → per-channel.
            Channels absent from a per-channel config return False for this metric.
        """
        is_per_channel = outcome_labels is not None and any(isinstance(v, dict) for v in outcome_labels.values())

        if not is_per_channel:
            labels = (outcome_labels or cls.outcome_labels).get(metric, [])
            return pl.col("Outcome").is_in(labels)

        # Per-channel mode: start from class defaults as the fallback expression
        default_labels = cls.outcome_labels.get(metric, [])
        expr: pl.Expr = pl.col("Outcome").is_in(default_labels)

        for channel, channel_labels in outcome_labels.items():
            if metric in channel_labels:
                expr = (
                    pl.when(pl.col("Channel") == channel)
                    .then(pl.col("Outcome").is_in(channel_labels[metric]))
                    .otherwise(expr)
                )

        return expr

    @classmethod
    @overload
    def from_vbd(
        cls,
        vbd_source: os.PathLike | str,
        *,
        outcome_labels: dict | None = None,
        return_df: Literal[True],
    ) -> pl.LazyFrame | None: ...

    # Default case: when return_df is not provided or False, returns ImpactAnalyzer
    @classmethod
    @overload
    def from_vbd(
        cls,
        vbd_source: os.PathLike | str,
        *,
        outcome_labels: dict | None = None,
    ) -> "ImpactAnalyzer | None": ...

    @classmethod
    def from_vbd(
        cls,
        vbd_source: os.PathLike | str,
        *,
        outcome_labels: dict | None = None,
        return_df: bool = False,
    ) -> "ImpactAnalyzer | pl.LazyFrame | None":
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
        outcome_labels : dict or None, optional
            Outcome value mappings for Impressions and Accepts. Accepts two formats:

            **Global override** — replaces class defaults for all channels::

                {"Impressions": ["Sent"], "Accepts": ["Click", "Clicked"]}

            **Per-channel** — overrides per channel; unconfigured channels fall back
            to the class-level defaults::

                {
                    "Email/Outbound": {
                        "Impressions": ["Sent"],
                        "Accepts": ["Click", "Clicked"],
                    }
                }

            Default is None (use :attr:`outcome_labels` class attribute).
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

        # Compute Channel/Direction early so we can scan available outcomes.
        raw = _polars_capitalize(vbd_data).with_columns(
            Channel=pl.concat_str(
                "Channel",
                "Direction",
                separator="/",
                ignore_nulls=True,
            ),
        )

        # Resolve outcome labels — always explicit, never implicit.
        # When caller provides None: derive channel-aware defaults from the data.
        if outcome_labels is None:
            channel_outcome_scan = (
                raw.select("Channel", "Outcome").unique().group_by("Channel").agg(pl.col("Outcome").sort()).collect()
            )
            outcome_labels = _resolve_outcome_labels({row[0]: row[1] for row in channel_outcome_scan.iter_rows()})

        ia_data = raw.with_columns(
            SnapshotTime=parse_pega_date_time_formats("OutcomeTime").dt.truncate(
                "1d",
            ),
        )

        # Treatment column is optional — some VBD exports (e.g. Rabo
        # ScenarioPlannerActuals) don't include it.  Use it when present
        # so that per-treatment granularity is preserved, otherwise omit.
        available_cols = ia_data.collect_schema().names()
        _treatment_col: str | None = None
        if "Treatment" in available_cols:
            _treatment_col = "Treatment"
        elif "TreatmentName" in available_cols:
            ia_data = ia_data.rename({"TreatmentName": "Treatment"})
            _treatment_col = "Treatment"

        _group_cols: list[str] = [
            "SnapshotTime",
            "MktValue",
            "Application",
            "ApplicationVersion",
            "Channel",
            "Issue",
            "Group",
            "Name",
        ]
        # Only add optional columns if they exist in the data
        if _treatment_col:
            _group_cols.append(_treatment_col)
        # Filter group_cols to those actually present
        _group_cols = [c for c in _group_cols if c in ia_data.collect_schema().names()]

        _sort_cols = [*_group_cols, "ControlGroup"]

        # Value column is optional — some VBD exports don't include it,
        # and in some it's stored as a string.  Ensure it's numeric.
        _has_value = "Value" in ia_data.collect_schema().names()
        if not _has_value:
            ia_data = ia_data.with_columns(pl.lit(0.0).alias("Value"))
        else:
            ia_data = ia_data.with_columns(
                pl.col("Value").cast(pl.Float64, strict=False).fill_null(0.0),
            )

        # AggregateCount can also be a string in some VBD exports.
        ia_data = ia_data.with_columns(
            pl.col("AggregateCount").cast(pl.Int64, strict=False).fill_null(1),
        )

        _agg_exprs = [
            pl.col("AggregateCount")
            .filter(cls._build_outcome_filter("Impressions", outcome_labels))
            .sum()
            .alias("Impressions"),
            pl.col("AggregateCount")
            .filter(cls._build_outcome_filter("Accepts", outcome_labels))
            .sum()
            .alias("Accepts"),
            (
                pl.col("Value").filter(cls._build_outcome_filter("Accepts", outcome_labels)).sum()
                / pl.col("AggregateCount").filter(cls._build_outcome_filter("Impressions", outcome_labels)).sum()
            ).alias("ValuePerImpression"),
            # kept for debugging and UI discovery
            pl.col("AggregateCount", "Value", "Outcome"),
        ]

        ia_data = (
            ia_data.group_by(_group_cols)
            .agg(_agg_exprs)
            .filter(pl.col("Accepts") <= pl.col("Impressions"))
            .rename({"MktValue": "ControlGroup"})
            .with_columns(
                pl.col("ControlGroup").str.strip_prefix("NBAHealth_").fill_null("NBA"),
            )
            .sort(
                *[c for c in _sort_cols if c != "MktValue"],
            )
        )

        if return_df:
            return ia_data

        instance = ImpactAnalyzer(ia_data)
        instance.outcome_labels_used = outcome_labels
        return instance

    @classmethod
    @overload
    def from_ih(
        cls,
        ih_source: os.PathLike | str,
        *,
        return_df: Literal[True],
    ) -> pl.LazyFrame | None: ...

    # Default case: when return_df is not provided or False, returns ImpactAnalyzer
    @classmethod
    @overload
    def from_ih(
        cls,
        ih_source: os.PathLike | str,
    ) -> "ImpactAnalyzer | None": ...

    @classmethod
    def from_ih(
        cls,
        ih_source: os.PathLike | str,
        *,
        return_df: bool = False,
    ) -> "ImpactAnalyzer | pl.LazyFrame | None":
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
        query: QUERY | None = None,
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
            json_data["pxResults"][0]["SnapshotTime"],
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        actual_ia_data: dict = json_data["pxResults"][0]["pxResults"]
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

        return cls._process_wide_df(wide_data, date=date)

    @classmethod
    def _process_wide_df(
        cls,
        wide_data: pl.LazyFrame,
        *,
        date: datetime,
    ) -> pl.LazyFrame:
        """Transform wide-format PDC IA data into the normalized long format.

        This shared helper is called by both :meth:`_normalize_pdc_ia_data`
        (JSON path) and :meth:`from_excel` (Excel path).

        Parameters
        ----------
        wide_data : pl.LazyFrame
            Wide-format data with one row per experiment / channel combination.
            Must contain the columns selected by :meth:`_normalize_pdc_ia_data`.
        date : datetime
            Snapshot date to stamp on every row.

        Returns
        -------
        pl.LazyFrame
            Normalized data with columns: SnapshotTime, Channel, ControlGroup,
            Impressions, Accepts, ValuePerImpression, Pega_ValueLift.

        """
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
                ],
            )
            # We're only using the most granular data, we will aggregate up ourselves
            .filter(AggregationFrequency="Daily")
            # Snapshot time is not set, overwrite with top level value
            .with_columns(pl.lit(date).cast(pl.Date).alias("SnapshotTime"))
            .with_columns(
                pl.when(LastDataReceived="Yesterday")
                .then(pl.col("SnapshotTime") - pl.duration(days=1))
                .otherwise(pl.col("SnapshotTime"))
                .alias("SnapshotTime"),
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
                },
            )
            .with_columns(
                ValuePerImpression=pl.lit(None).cast(pl.Float64),
                ControlGroup=pl.when(pl.col("ControlGroup") == "NBAHealth_ModelControl")
                .then(pl.lit("NBAHealth_ModelControl_1"))
                .otherwise("ControlGroup"),
            )
        )

        return (
            pl.concat(
                [nba_data, model_control_2_data, other_data],
                how="diagonal_relaxed",
            )
            .with_columns(pl.col("ControlGroup").str.strip_prefix("NBAHealth_"))
            .sort("SnapshotTime", "Channel", "ControlGroup")
        )

    # Mapping from Pega Infinity IA UI experiment names → (test_arm, control_arm).
    # The keys must match the strings in the "Experiment Name" column of the
    # Data sheet exactly. Unknown experiment names trigger a warning and the
    # rows are dropped.
    _excel_experiment_to_arms: ClassVar[dict[str, tuple[str, str]]] = {
        "NBA vs Random relevant action": ("NBA", "NBAPrioritization"),
        "NBA vs Arbitrating with propensity only": ("NBA", "PropensityPriority"),
        "NBA vs NBA without levers": ("NBA", "LeverPriority"),
        "NBA vs NBA with eligibility polices only": ("NBA", "EngagementPolicy"),
        "AdaptiveModel (p) vs Random (p)": ("ModelControl_2", "ModelControl_1"),
    }

    @classmethod
    def from_excel(
        cls,
        excel_source: str | Path | os.PathLike,
        *,
        sheet_name: str = "Data",
        query: QUERY | None = None,
        return_df: bool = False,
    ) -> "ImpactAnalyzer | pl.LazyFrame":
        """Create an ImpactAnalyzer instance from a Pega Infinity IA Excel export.

        Reads the ``Data`` sheet of the Impact Analyzer Excel export produced
        by the Impact Analyzer landing page in Pega Infinity. Each row of the
        Data sheet describes one (Date, Channel, Direction, Issue, Group,
        Action, Treatment, Experiment) bucket with pre-paired Test and Control
        impression / accept / value counts.  This method explodes those rows
        to the long format used by :class:`ImpactAnalyzer` and deduplicates
        the NBA test arm across experiments (the same NBA traffic is reported
        against multiple control experiments and would otherwise be double
        counted).

        The Channel field is built as ``"<Channel>/<Direction>"`` to match the
        convention used by :meth:`from_vbd`.

        Excel reading is handled by the internal Excel reader built on
        polars' ``calamine`` engine.

        Parameters
        ----------
        excel_source : Union[str, Path, os.PathLike]
            Path to the ``.xlsx`` file.
        sheet_name : str, default "Data"
            Sheet to read.  The exporter ships several sheets; only ``Data``
            carries the row-level counts needed for analysis.
        query : Optional[QUERY], optional
            Polars expression to filter the long-form data before
            aggregation.  Default is ``None``.
        return_df : bool, optional
            If ``True``, return the normalised data as a :class:`~polars.LazyFrame`
            instead of an :class:`ImpactAnalyzer` instance.  Default is ``False``.

        Returns
        -------
        ImpactAnalyzer or pl.LazyFrame
            An :class:`ImpactAnalyzer` instance, or a :class:`~polars.LazyFrame`
            when *return_df* is ``True``.

        Raises
        ------
        ValueError
            If the requested sheet is missing required columns.

        Examples
        --------
        >>> ia = ImpactAnalyzer.from_excel("ImpactAnalyzerExport.xlsx")
        >>> ia.overall_summary().collect()

        """
        from ..pega_io.File import _read_excel

        df = _read_excel(excel_source, sheet_name=sheet_name)
        normalized = cls._normalize_excel_data_sheet(df.lazy(), query=query)

        if return_df:
            return normalized

        return ImpactAnalyzer(normalized)

    # ------------------------------------------------------------------
    # Excel Data-sheet normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _excel_to_datetime(col: str) -> pl.Expr:
        """Coerce a column to Datetime, tolerating locale-dependent formats.

        Excel exports may surface dates as native datetimes, ISO strings, or
        locale-specific strings (``31/03/2026`` vs ``03/31/2026`` vs
        ``31-Mar-2026``).  We try the most common shapes and coalesce the
        first non-null result per row.
        """
        c = pl.col(col)
        # First attempt: trust the dtype as polars/calamine read it.
        # cast(strict=False) yields null for incompatible source dtypes
        # (e.g. String → Datetime), letting subsequent attempts take over.
        attempts: list[pl.Expr] = [c.cast(pl.Datetime, strict=False)]
        # String-based fallbacks. Cast everything to String first so an
        # already-numeric column (Excel serial) still gets tried below.
        s = c.cast(pl.String, strict=False).str.strip_chars()
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%.fZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%d-%b-%Y",
            "%d %b %Y",
            "%b %d, %Y",
        ):
            attempts.append(s.str.strptime(pl.Datetime, fmt, strict=False))
        # Excel serial date fallback: number of days since 1899-12-30.
        serial = c.cast(pl.Float64, strict=False)
        attempts.append(
            (
                pl.lit(datetime(1899, 12, 30)).cast(pl.Datetime)
                + pl.duration(seconds=(serial * 86400).cast(pl.Int64, strict=False))
            )
        )
        return pl.coalesce(attempts).alias(col)

    @staticmethod
    def _excel_to_numeric(col: str) -> pl.Expr:
        """Coerce a column to Float64, tolerating locale-formatted numbers.

        Handles strings with thousands separators (``"1,234"``,
        ``"1 234"``) and European decimal commas (``"1.234,56"``) by trying
        several normalisation strategies in order.
        """
        c = pl.col(col)
        s = c.cast(pl.String, strict=False).str.strip_chars()
        return pl.coalesce(
            # Already numeric
            c.cast(pl.Float64, strict=False),
            # US/UK style: "1,234.56" → strip commas + spaces
            s.str.replace_all(r"[,\s]", "").cast(pl.Float64, strict=False),
            # European style: "1.234,56" → strip dots and spaces, swap comma → dot
            s.str.replace_all(r"[.\s]", "").str.replace(",", ".").cast(pl.Float64, strict=False),
        ).alias(col)

    @classmethod
    def _normalize_excel_data_sheet(
        cls,
        wide: pl.LazyFrame,
        *,
        query: QUERY | None = None,
    ) -> pl.LazyFrame:
        """Normalise the wide Data-sheet rows into the long IA format.

        Each input row carries Test and Control counts for one experiment.
        We split it into one row per arm, then aggregate (max) across
        experiments so the shared NBA test traffic is counted once.
        """
        required = {
            "Date",
            "Experiment Name",
            "Issue",
            "Group",
            "Action",
            "Treatment",
            "Direction",
            "Channel",
            "Impressions_Test",
            "Accepts_Test",
            "Impressions_Control",
            "Accepts_Control",
        }
        present = set(wide.collect_schema().names())
        missing = required - present
        if missing:
            raise ValueError(f"Excel Data sheet is missing required columns: {sorted(missing)}")

        # Optional value columns — tolerate their absence in older exports.
        has_value_test = "ActionValueImpression_Test" in present
        has_value_control = "ActionValueImpression_Control" in present

        numeric_cols = [
            "Impressions_Test",
            "Accepts_Test",
            "Impressions_Control",
            "Accepts_Control",
        ]
        if has_value_test:
            numeric_cols.append("ActionValueImpression_Test")
        if has_value_control:
            numeric_cols.append("ActionValueImpression_Control")

        # Coerce dtypes defensively (locale-tolerant).
        coerced = wide.with_columns(
            cls._excel_to_datetime("Date"),
            *[cls._excel_to_numeric(c) for c in numeric_cols],
        )

        # Validate experiment names; warn on unknowns and drop them.
        known_experiments = list(cls._excel_experiment_to_arms.keys())
        observed = coerced.select(pl.col("Experiment Name").unique()).collect()["Experiment Name"].to_list()
        unknown = sorted(set(observed) - set(known_experiments))
        if unknown:
            logger.warning(
                "Dropping rows for unknown Impact Analyzer experiments: %s. Known experiments: %s.",
                unknown,
                known_experiments,
            )
            coerced = coerced.filter(pl.col("Experiment Name").is_in(known_experiments))

        # Map experiment → (test_arm, control_arm) via two parallel literal expressions.
        test_arm_expr = pl.col("Experiment Name")
        control_arm_expr = pl.col("Experiment Name")
        for name, (test_arm, control_arm) in cls._excel_experiment_to_arms.items():
            test_arm_expr = test_arm_expr.replace(name, test_arm)
            control_arm_expr = control_arm_expr.replace(name, control_arm)

        common_cols = [
            pl.col("Date").alias("SnapshotTime"),
            (pl.col("Channel").cast(pl.String) + "/" + pl.col("Direction").cast(pl.String)).alias("Channel"),
            pl.col("Issue"),
            pl.col("Group"),
            pl.col("Action").alias("Name"),
            pl.col("Treatment"),
        ]

        test_long = coerced.select(
            *common_cols,
            test_arm_expr.alias("ControlGroup"),
            pl.col("Impressions_Test").alias("Impressions"),
            pl.col("Accepts_Test").alias("Accepts"),
            (pl.col("ActionValueImpression_Test") if has_value_test else pl.lit(None, dtype=pl.Float64)).alias(
                "_ValueImpression"
            ),
        )

        control_long = coerced.select(
            *common_cols,
            control_arm_expr.alias("ControlGroup"),
            pl.col("Impressions_Control").alias("Impressions"),
            pl.col("Accepts_Control").alias("Accepts"),
            (pl.col("ActionValueImpression_Control") if has_value_control else pl.lit(None, dtype=pl.Float64)).alias(
                "_ValueImpression"
            ),
        )

        long = pl.concat([test_long, control_long], how="vertical_relaxed")

        if query is not None:
            long = _apply_query(long, query=query)

        # Aggregate: for non-NBA arms there is one row per (date, channel,
        # action, treatment, ControlGroup); for the NBA arm there are up to
        # 4 identical rows (one per NBA-vs-X experiment). max() collapses
        # them to a single canonical value without double counting.
        group_cols = [
            "SnapshotTime",
            "Channel",
            "Issue",
            "Group",
            "Name",
            "Treatment",
            "ControlGroup",
        ]
        return (
            long.group_by(group_cols)
            .agg(
                pl.col("Impressions").max(),
                pl.col("Accepts").max(),
                pl.col("_ValueImpression").max(),
            )
            .filter(pl.col("Accepts") <= pl.col("Impressions"))
            .with_columns(
                ValuePerImpression=pl.when(pl.col("Impressions") > 0)
                .then(pl.col("_ValueImpression") / pl.col("Impressions"))
                .otherwise(None),
            )
            .drop("_ValueImpression")
            .sort("SnapshotTime", "Channel", "Issue", "Group", "Name", "Treatment", "ControlGroup")
        )

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
        by: Sequence[str | pl.Expr] | str | pl.Expr | None = None,
        drop_internal_cols: bool = True,
    ) -> pl.LazyFrame:
        """Aggregate metrics by control group.

        Summarizes impressions, accepts, CTR, and value metrics for each
        control group, optionally grouped by additional dimensions.

        Parameters
        ----------
        by : Optional[Union[list[str], list[pl.Expr], str, pl.Expr]], optional
            Column name(s) or expression(s) to group by in addition to
            ControlGroup. Default is None (aggregate all data).
        drop_internal_cols : bool, optional
            If True, drop internal columns prefixed with ``Pega_``.
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
        if by is None:
            group_by: list[str | pl.Expr] = []
        elif isinstance(by, (str, pl.Expr)):
            group_by = [by]
        else:
            group_by = list(by)

        agg_exprs = [
            pl.sum("Impressions", "Accepts"),
            (pl.sum("Accepts") / pl.sum("Impressions")).alias("CTR"),
            weighted_average_polars("ValuePerImpression", "Impressions").alias(
                "ValuePerImpression",
            ),
        ]

        # Pega_ValueLift is only present in PDC data, not in VBD data
        if "Pega_ValueLift" in self.ia_data.collect_schema().names():
            agg_exprs.append(
                weighted_average_polars("Pega_ValueLift", "Impressions").alias(
                    "Pega_ValueLift",
                ),
            )

        return (
            self.ia_data.sort([*group_by, "ControlGroup"])
            .group_by([*group_by, "ControlGroup"], maintain_order=True)
            .agg(agg_exprs)
            .drop(cs.starts_with("Pega_") if drop_internal_cols else [])
        )

    def summarize_experiments(
        self,
        by: Sequence[str | pl.Expr] | str | pl.Expr | None = None,
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
        by : Optional[Union[list[str], list[pl.Expr], str, pl.Expr]], optional
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
        # Normalize 'by' parameter to a sequence
        if by is None:
            by_list: Sequence[str | pl.Expr] = []
        elif isinstance(by, (str, pl.Expr)):
            by_list = [by]
        else:
            # Already a sequence (list, tuple, etc.)
            by_list = by

        # Extract column names from expressions for use with pl.exclude()
        def _get_column_names(items: Sequence[str | pl.Expr]) -> list[str]:
            column_names: list[str] = []
            for item in items:
                if isinstance(item, pl.Expr):
                    # Extract the root column name from the expression
                    column_names.append(item.meta.output_name())
                else:
                    column_names.append(item)
            return column_names

        by_column_names: list[str] = _get_column_names(by_list)

        control_groups_summary = self.summarize_control_groups(
            by_list if by_list else None,
            drop_internal_cols=False,
        )

        has_pega_value_lift = "Pega_ValueLift" in self.ia_data.collect_schema().names()

        # Cast Experiment to Enum so the canonical product order from
        # default_ia_experiments survives joins/aggregations and a final
        # sort puts experiments back in that order.
        experiment_enum = pl.Enum(list(ImpactAnalyzer.default_ia_experiments.keys()))

        result = (
            pl.LazyFrame(
                {
                    "Experiment": ImpactAnalyzer.default_ia_experiments.keys(),
                    "Test": [v[1] for v in ImpactAnalyzer.default_ia_experiments.values()],
                    "Control": [v[0] for v in ImpactAnalyzer.default_ia_experiments.values()],
                },
            )
            .with_columns(pl.col("Experiment").cast(experiment_enum))
            .join(
                control_groups_summary.select(
                    *by_list,
                    pl.exclude(by_column_names).name.suffix("_Test"),
                ),
                how="left",
                left_on="Test",
                right_on="ControlGroup_Test",
            )
            .join(
                control_groups_summary.select(
                    *by_list,
                    pl.exclude(by_column_names).name.suffix("_Control"),
                ),
                how="left",
                left_on=["Control", *by_column_names],
                right_on=["ControlGroup_Control", *by_column_names],
            )
            .with_columns(
                Control_Fraction=pl.col("Impressions_Control")
                / (pl.col("Impressions_Control") + pl.col("Impressions_Test")),
                CTR_Lift=lift_pl("CTR_Test", "CTR_Control"),
            )
        )

        # Value_Lift from Pega_ValueLift is only available in PDC data
        if has_pega_value_lift:
            result = result.with_columns(Value_Lift=pl.col("Pega_ValueLift_Control"))
        else:
            # For VBD data, calculate Value_Lift from ValuePerImpression
            result = result.with_columns(
                Value_Lift=lift_pl(
                    "ValuePerImpression_Test",
                    "ValuePerImpression_Control",
                ),
            )

        return (
            result.drop(cs.starts_with("Pega_"))
            .sort(["Experiment", *by_column_names])
            .with_columns(pl.col("Experiment").cast(pl.String))
        )
