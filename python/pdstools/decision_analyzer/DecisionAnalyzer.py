# python/pdstools/decision_analyzer/DecisionAnalyzer.py
from bisect import bisect_left
from functools import cached_property
from typing import Literal
import logging
import os
import warnings

import polars as pl
import polars.selectors as cs

from .data_read_utils import validate_columns
from .plots import Plot
from .stage_grouping import get_display_name
from .column_schema import (
    DecisionAnalyzer as DecisionAnalyzer_TD,
    ExplainabilityExtract as ExplainabilityExtract_TD,
)
from .utils import (
    SCOPE_HIERARCHY,
    apply_filter,
    determine_extract_type,
    get_table_definition,
    gini_coefficient,
    rename_and_cast_types,
    resolve_aliases,
)
from ..pega_io.File import read_ds_export

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 10_000
"""Default number of unique interactions to sample for resource-intensive analyses."""


class DecisionAnalyzer:
    """Analyze NBA decision data from Explainability Extract or Decision Analyzer exports.

    This class processes raw decision data to create a comprehensive analysis
    framework for NBA (Next-Best-Action). It supports two data source formats:

    - **Explainability Extract (v1)**: Simpler format with actions at the
      arbitration stage. Stages are synthetically derived from ranking.
    - **Decision Analyzer / EEV2 (v2)**: Full pipeline data with real stage
      information, filter component names, and detailed strategy tracking.

    Data can be loaded via class methods or directly:

    - :meth:`from_explainability_extract`: Load from an Explainability Extract file.
    - :meth:`from_decision_analyzer`: Load from a Decision Analyzer (EEV2) file.
    - Direct ``__init__``: Auto-detects format from the data schema.

    Attributes
    ----------
    decision_data : pl.LazyFrame
        Interaction-level decision data (with global filters applied if any).
    extract_type : str
        Either ``"explainability_extract"`` or ``"decision_analyzer"``.
    plot : Plot
        Plot accessor for visualization methods.

    Examples
    --------
    >>> from pdstools import DecisionAnalyzer
    >>> da = DecisionAnalyzer.from_explainability_extract("data/sample_explainability_extract.parquet")
    >>> da.get_overview_stats
    >>> da.plot.sensitivity()
    """

    @classmethod
    def from_explainability_extract(
        cls,
        source: str | os.PathLike,
        **kwargs,
    ) -> "DecisionAnalyzer":
        """Create a DecisionAnalyzer from an Explainability Extract (v1) file.

        Parameters
        ----------
        source : str | os.PathLike
            Path to the Explainability Extract parquet file, or a URL.
        **kwargs
            Additional keyword arguments passed to ``__init__`` (e.g.
            ``sample_size``, ``mandatory_expr``, ``additional_columns``).

        Returns
        -------
        DecisionAnalyzer

        Examples
        --------
        >>> da = DecisionAnalyzer.from_explainability_extract("data/sample_explainability_extract.parquet")
        """
        raw_data = read_ds_export(str(source))
        if raw_data is None:
            raise ValueError(f"Could not read data from {source}")
        return cls(raw_data, **kwargs)

    @classmethod
    def from_decision_analyzer(
        cls,
        source: str | os.PathLike,
        **kwargs,
    ) -> "DecisionAnalyzer":
        """Create a DecisionAnalyzer from a Decision Analyzer / EEV2 (v2) file.

        Parameters
        ----------
        source : str | os.PathLike
            Path to the Decision Analyzer parquet file, or a URL.
        **kwargs
            Additional keyword arguments passed to ``__init__`` (e.g.
            ``sample_size``, ``mandatory_expr``, ``additional_columns``).

        Returns
        -------
        DecisionAnalyzer

        Examples
        --------
        >>> da = DecisionAnalyzer.from_decision_analyzer("data/sample_eev2.parquet")
        """
        raw_data = read_ds_export(str(source))
        if raw_data is None:
            raise ValueError(f"Could not read data from {source}")
        return cls(raw_data, **kwargs)

    # Preferred fields for data filtering, in display order.
    # Subsetting to actual available columns happens in __init__.
    _default_filter_fields = [
        "Decision Time",
        "Channel",
        "Direction",
        "Issue",
        "Group",
        "Action",
        "Treatment",
        "Stage",
        "ModelPositives",
        "ModelEvidence",
    ]

    def __init__(
        self,
        raw_data: pl.LazyFrame,
        level="Stage Group",
        sample_size=DEFAULT_SAMPLE_SIZE,
        mandatory_expr: pl.Expr | None = None,
        additional_columns: dict[str, pl.DataType] | None = None,
    ):
        """Initialize DecisionAnalyzer with raw decision data.

        This class processes raw decision data (Explainability Extract) to create a comprehensive analysis framework
        for NBA. After light preprocessing it allows users to conduct various analysis on the data.

        For more information about Explainability Extract, see:
        https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/explainability-extract.html

        Parameters
        ----------
        raw_data : pl.LazyFrame
            Raw decision data containing interaction-level records from Explainability Extract.
        level : str, default "Stage Group"
            Granularity level for stage analysis. Options:
            - "Stage Group": Groups stages into categories (recommended)
            - "Stage": Individual stage-level analysis
        sample_size : int, default 50000
            Maximum number of unique interactions to sample for analysis. Larger values
            provide more statistical accuracy but slower performance. Minimum 1000.
        mandatory_expr : pl.Expr, optional
            Polars expression to create the `is_mandatory` column used in ranking.
            The expression should return True/False values that get converted to 1/0.
            Actions with is_mandatory=1 get FIRST rank in the ranking function.

            Example: `pl.col("Issue") == "Service"` results in:
            - Service actions: is_mandatory = 1 (ranked first)
            - Non-Service actions: is_mandatory = 0 (ranked by other criteria)

            Other examples:
            - `(pl.col("Group") == "Credit") & (pl.col("Priority") > 0.8)`
            - `pl.col("Action").is_in(["CriticalAction1", "CriticalAction2"])`
        additional_columns : dict[str, pl.DataType], optional
            Additional columns to include in processing beyond the standard table definition.
            Dictionary mapping column names to their polars data types.

            Example: additional_columns = {"non_standard_column" : pl.Utf8}

        Notes
        -----
        The ranking function orders actions by: is_mandatory (desc) → Priority (desc) →
        StageOrder (desc) → pyIssue → pyGroup → pyName.

        If mandatory_expr is None, all actions get is_mandatory=0.
        The expression gets applied as: `raw_data.with_columns(is_mandatory=mandatory_expr)`

        Examples
        --------
        Basic usage:
        >>> analyzer = DecisionAnalyzer(raw_data)

        Make retention actions mandatory:
        >>> mandatory = pl.col("Issue") == "Retention"
        >>> decision_analyzer = DecisionAnalyzer(raw_data, mandatory_expr=mandatory)
        """
        self.plot = Plot(self)
        self.level = level
        self.sample_size = sample_size
        self._thresholding_cache: dict[tuple[str, tuple[int, ...]], pl.DataFrame] = {}
        self._sensitivity_cache: dict[int, pl.LazyFrame] = {}
        # Normalize alternative column names (e.g. "Issue" → "Issue")
        raw_data = resolve_aliases(raw_data, DecisionAnalyzer_TD, ExplainabilityExtract_TD)
        # pxEngagement Stage present?
        self.extract_type = determine_extract_type(raw_data)
        # Get table definition and add any additional columns to it
        table_def = get_table_definition(self.extract_type)
        if additional_columns:
            for col_name, col_type in additional_columns.items():
                table_def[col_name] = {
                    "display_name": col_name,
                    "default": True,
                    "type": col_type,
                }
        # all columns are present?
        validation_result, validation_error = validate_columns(raw_data, table_def)
        self.validation_error = validation_error if not validation_result else None
        if not validation_result and validation_error is not None:
            warnings.warn(validation_error, UserWarning)
        # Bail out early if critical columns are missing — cleanup_raw_data
        # would crash with an opaque ColumnNotFoundError otherwise.
        # Check using display names; account for both raw keys and display
        # names already present in the data (rename hasn't happened yet).
        critical_display_names = {"Interaction ID", "Issue", "Group", "Action"}
        available = set(raw_data.collect_schema().names())
        # Build the set of display names that will be available after renaming
        available_after_rename = set()
        for raw_key, config in table_def.items():
            if raw_key in available or config["display_name"] in available:
                available_after_rename.add(config["display_name"])
        missing_critical = critical_display_names - available_after_rename
        if missing_critical:
            raise ValueError(
                f"Cannot construct DecisionAnalyzer: critical columns missing "
                f"from the data: {', '.join(sorted(missing_critical))}"
            )

        # cast datatypes
        raw_data = rename_and_cast_types(df=raw_data, table_definition=table_def).sort("Interaction ID")
        if mandatory_expr is not None:
            raw_data = raw_data.with_columns(is_mandatory=mandatory_expr)
        else:
            raw_data = raw_data.with_columns(is_mandatory=pl.lit(0))
        self.unfiltered_raw_decision_data = self.cleanup_raw_data(raw_data)
        self.resetGlobalDataFilters()
        available_columns = set(self.unfiltered_raw_decision_data.collect_schema().names())
        self.fields_for_data_filtering = [f for f in self._default_filter_fields if f in available_columns]
        self.preaggregation_columns = {
            "Issue",
            "Group",
            "Action",
            "Treatment",
            "Channel",
            "Direction",
            "Component Name",
            # "Model Control Group",
            "day",
        }
        self.preaggregation_columns = self.preaggregation_columns.intersection(available_columns)

        self.max_win_rank = 5
        self.AvailableNBADStages = ["Arbitration", "Output"]

        if self.extract_type == "explainability_extract":
            columns_to_remove = {"Component Name"}
            self.preaggregation_columns -= columns_to_remove

            self.NBADStages_FilterView = self.AvailableNBADStages
            self.NBADStages_RemainingView = self.AvailableNBADStages

        elif self.extract_type == "decision_analyzer":
            stage_df = self.unfiltered_raw_decision_data.group_by(self.level).agg(pl.min("Stage Order")).collect()
            if "Arbitration" not in stage_df[self.level] and self.level == "Stage Group":
                arb = pl.DataFrame(
                    {self.level: "Arbitration", "Stage Order": 3800},
                    schema=stage_df.schema,
                )
                stage_df = pl.concat([stage_df, arb])
            stage_df = stage_df.sort("Stage Order")
            self.AvailableNBADStages = stage_df.get_column(self.level).to_list()

    @property
    def available_levels(self) -> list[str]:
        """Stage granularity levels available for this dataset.

        Returns ``["Stage Group", "Stage"]`` for Decision Analyzer (v2) data
        when both columns are present, or ``["Stage Group"]`` for
        Explainability Extract (v1) data where only synthetic stages exist.
        """
        if self.extract_type == "explainability_extract":
            return ["Stage Group"]
        available = set(self.unfiltered_raw_decision_data.collect_schema().names())
        levels = []
        if "Stage Group" in available:
            levels.append("Stage Group")
        if "Stage" in available:
            levels.append("Stage")
        return levels if levels else ["Stage Group"]

    def set_level(self, level: str):
        """Switch the stage granularity level used for all analyses.

        Recomputes the available stages for the new level and invalidates
        all cached properties so subsequent queries use the new granularity.

        Parameters
        ----------
        level : str
            ``"Stage Group"`` or ``"Stage"``.
        """
        if level == self.level:
            return
        valid_levels = set(self.available_levels)
        if level not in valid_levels:
            raise ValueError(f"level must be one of {sorted(valid_levels)}, got '{level}'")
        self.level = level
        self._recompute_available_stages()
        self._invalidate_cached_properties()

    def _recompute_available_stages(self):
        """Derive ``AvailableNBADStages`` from the data for the current level."""
        if self.extract_type == "explainability_extract":
            self.AvailableNBADStages = ["Arbitration", "Output"]
            return

        stage_df = self.unfiltered_raw_decision_data.group_by(self.level).agg(pl.min("Stage Order")).collect()
        if "Arbitration" not in stage_df[self.level].to_list() and self.level == "Stage Group":
            arb = pl.DataFrame(
                {self.level: "Arbitration", "Stage Order": 3800},
                schema=stage_df.schema,
            )
            stage_df = pl.concat([stage_df, arb])
        stage_df = stage_df.sort("Stage Order")
        self.AvailableNBADStages = stage_df.get_column(self.level).to_list()

    @cached_property
    def color_mappings(self) -> dict[str, dict[str, str]]:
        """Compute consistent color mappings for all categorical dimensions.

        Color assignments are based on all unique values in the full dataset
        (before sampling), sorted alphabetically. This ensures colors remain
        consistent throughout the session regardless of filtering.

        Returns
        -------
        dict[str, dict[str, str]]
            Nested dictionary mapping dimension names to color dictionaries.
            Example: {
                "Issue": {"Retention": "#001F5F", "Sales": "#10A5AC"},
                "Group": {"CreditCards": "#001F5F", "Loans": "#10A5AC"},
            }

        Notes
        -----
        Uses @cached_property so computation happens once on first access.
        Colors are assigned from the Pega colorway using modulo indexing.

        See Also
        --------
        pdstools.utils.color_mapping.create_categorical_color_mappings
            Generic utility for creating color mappings in any Streamlit app.
        """
        from ..utils.color_mapping import create_categorical_color_mappings
        from ..utils.pega_template import colorway

        categorical_columns = [
            "Issue",
            "Group",
            "Action",
            "Treatment",
            "Channel",
            "Direction",
            "Stage Group",
            "Stage",
        ]

        return create_categorical_color_mappings(self.decision_data, categorical_columns, colorway)

    @cached_property
    def stages_from_arbitration_down(self):
        """All stages from Arbitration onward, respecting the current level.

        At "Stage Group" level this slices from the literal "Arbitration"
        entry.  At "Stage" level it finds stages whose Stage Order is >=
        the Arbitration group order (3800) using the stage_to_group_mapping.
        """
        stages = self.AvailableNBADStages
        if "Arbitration" in stages:
            return stages[stages.index("Arbitration") :]
        # At "Stage" level: use the group mapping to find stages in or
        # after the Arbitration group.
        mapping = self.stage_to_group_mapping
        if mapping:
            arb_stages = {s for s, g in mapping.items() if g == "Arbitration"}
            for i, s in enumerate(stages):
                if s in arb_stages:
                    return stages[i:]
        # Fallback: return all stages
        return stages

    @cached_property
    def stages_with_propensity(self):
        """Infer which stages have meaningful propensity scores from the data.

        Examines the sample data to determine which stages have non-null, non-default
        propensity values. Returns stages where propensity-based classification makes sense.
        """
        # Check which stages have meaningful propensity values in the sample data
        # Propensity = 0.5 is the default for new/untested models, so we look for diversity
        propensity_by_stage = (
            self.sample.group_by(self.level)
            .agg(
                [
                    pl.col("Propensity").drop_nulls().count().alias("prop_count"),
                    pl.col("Propensity").drop_nulls().n_unique().alias("prop_unique"),
                    # Check if there are non-0.5 propensity values
                    (pl.col("Propensity").drop_nulls() != 0.5).sum().alias("prop_non_default"),
                ]
            )
            .collect()
        )

        # Stages have propensity if they have diverse propensity scores (not just 0.5)
        stages_with_scores = (
            propensity_by_stage.filter(
                (pl.col("prop_count") > 0) & ((pl.col("prop_unique") > 1) | (pl.col("prop_non_default") > 0))
            )
            .get_column(self.level)
            .to_list()
        )

        if stages_with_scores:
            return stages_with_scores

        # Fallback: use stages_from_arbitration_down (Arbitration and later)
        return self.stages_from_arbitration_down

    @cached_property
    def propensity_validation_warning(self) -> str | None:
        """Validate propensity values and return warning message if issues detected.

        Checks for:
        1. Invalid propensities (> 1.0) - mathematically impossible for probabilities
        2. Unusually high propensities (> 0.1) - uncommon for typical marketing interactions

        Returns None if validation passes or propensity data is not available.
        Uses sample data for efficiency.
        """
        # Edge case: No propensity column
        if "Propensity" not in self.sample.collect_schema().names():
            return None

        # Edge case: No stages with meaningful propensity
        if not self.stages_with_propensity:
            return None

        # Filter to stages with propensity and compute statistics
        propensity_data = (
            self.sample.filter(pl.col(self.level).is_in(self.stages_with_propensity))
            .select(pl.col("Propensity"), pl.col(self.level))
            .filter(pl.col("Propensity").is_not_null() & pl.col("Propensity").is_finite())
            .collect()
        )

        # Edge case: No valid propensity data
        total_count = propensity_data.height
        if total_count == 0:
            return None

        # Compute statistics
        stats = propensity_data.select(
            pl.col("Propensity").quantile(0.95).alias("p95"),
            pl.col("Propensity").max().alias("max"),
            (pl.col("Propensity") > 1.0).sum().alias("invalid_count"),
            (pl.col("Propensity") > 0.1).sum().alias("high_count"),
        ).row(0, named=True)

        # Build warning messages
        warnings = []

        # Check for invalid propensities (> 1.0)
        if stats["invalid_count"] > 0:
            invalid_pct = 100 * stats["invalid_count"] / total_count
            # Get stages with invalid values
            invalid_stages = (
                propensity_data.filter(pl.col("Propensity") > 1.0)
                .select(pl.col(self.level).unique())
                .get_column(self.level)
                .to_list()
            )
            stage_list = (
                ", ".join(invalid_stages)
                if len(invalid_stages) <= 3
                else f"{', '.join(invalid_stages[:3])}, and {len(invalid_stages) - 3} more"
            )

            warnings.append(
                f"⚠️ **Invalid propensity values detected:**\n\n"
                f"• {stats['invalid_count']:,} records ({invalid_pct:.2f}%) have propensities > 1.0\n\n"
                f"• Maximum propensity: {stats['max']:.2f}\n\n"
                f"• Affected stages: {stage_list}\n\n"
                f"Propensities should be between 0 and 1. Please check your model calibration or data extraction process."
            )

        # Check for unusually high propensities (> 10%)
        elif stats["high_count"] > 0:  # Only show if no invalid values
            high_pct = 100 * stats["high_count"] / total_count
            # Only warn if a significant portion has high values
            if high_pct > 5:  # More than 5% of records
                high_stages = (
                    propensity_data.filter(pl.col("Propensity") > 0.1)
                    .select(pl.col(self.level).unique())
                    .get_column(self.level)
                    .to_list()
                )
                stage_list = (
                    ", ".join(high_stages)
                    if len(high_stages) <= 3
                    else f"{', '.join(high_stages[:3])}, and {len(high_stages) - 3} more"
                )

                warnings.append(
                    f"ℹ️ **Unusually high propensities detected:**\n\n"
                    f"• {high_pct:.1f}% of records have propensities > 10%\n\n"
                    f"• 95th percentile propensity: {stats['p95'] * 100:.1f}%\n\n"
                    f"• Affected stages: {stage_list}\n\n"
                    f"This is unusual for typical marketing interactions (usually < 1%). This might indicate:\n\n"
                    f"• Different model calibration approach\n\n"
                    f"• High-intent channels or contexts\n\n"
                    f"• Potential data quality issues\n\n"
                    f"Consider reviewing your model configuration if this seems unexpected."
                )

        return warnings[0] if warnings else None

    @cached_property
    def arbitration_stage(self):
        return self.sample.filter(pl.col(self.level).is_in(self.stages_from_arbitration_down))

    @property
    def num_sample_interactions(self) -> int:
        """
        Number of unique interactions in the sample.
        Automatically triggers sampling if not yet calculated.
        """
        if not hasattr(self, "_num_sample_interactions"):
            # Trigger sample calculation to set _num_sample_interactions
            _ = self.sample
        return self._num_sample_interactions  # type: ignore[attr-defined]

    def _invalidate_cached_properties(self):
        """Resets the properties of the class. Needed for global filters."""
        cls = type(self)
        cached = {
            attr
            for attr in list(self.__dict__.keys())
            if (descriptor := getattr(cls, attr, None))
            if isinstance(descriptor, cached_property)
        }
        for attr in cached:
            del self.__dict__[attr]
        # Also reset num_sample_interactions so it gets recalculated with new filters
        if hasattr(self, "_num_sample_interactions"):
            delattr(self, "_num_sample_interactions")
        self._thresholding_cache.clear()
        self._sensitivity_cache.clear()

    def applyGlobalDataFilters(self, filters: pl.Expr | list[pl.Expr] | None = None):
        """
        Apply a global set of filters
        """
        self._invalidate_cached_properties()
        if filters is not None:
            self.decision_data = apply_filter(self.unfiltered_raw_decision_data, filters)

    def resetGlobalDataFilters(self):
        self.decision_data = self.unfiltered_raw_decision_data
        self._invalidate_cached_properties()

    @cached_property
    def getPreaggregatedFilterView(self):
        """Pre-aggregates the full dataset over customers and interactions providing
        a view of what is filtered at a stage.

        This pre-aggregation is pretty similar to what "VBD" does to interaction
        history. It aggregates over individual customers and interactions giving
        summary statistics that are sufficient to drive most of the analyses
        (but not all). The results of this pre-aggregation are much smaller
        than the original data and is expected to easily fit in memory. We therefore
        use polars caching to efficiently cache this.

        This "filter" view keeps the same organization as the decision analyzer data
        in that it records the actions that get filtered out at stages. From this
        a "remaining" view is easily derived.
        """
        num_samples = 1  # TODO: when > 1 this breaks the .explode() I'm doing in some places(thresholding analysis e.g.), need to solve
        stats_cols = ["Decision Time", "Value", "Propensity", "Priority"]
        exprs = [
            pl.col("Interaction ID").filter(pl.col("Rank") <= i).count().alias(f"Win_at_rank{i}")
            for i in range(1, self.max_win_rank + 1)
        ] + [
            pl.min(*stats_cols).name.suffix("_min"),
            pl.max(*stats_cols).name.suffix("_max"),
            pl.col("Propensity", "Priority").sample(n=num_samples, with_replacement=True, shuffle=True),
            pl.col("Interaction ID").unique().alias("Interaction_IDs"),
            pl.col("Interaction ID").n_unique().alias("Decisions"),
        ]

        self.preaggregated_decision_data_filterview = (
            self.decision_data.group_by(self.preaggregation_columns.union({self.level, "Stage Order", "Record Type"}))
            .agg(exprs)
            .collect()  # materialize to cache the expensive aggregation
            .lazy()  # re-wrap so downstream consumers stay lazy
        )
        return self.preaggregated_decision_data_filterview

    @cached_property
    def getPreaggregatedRemainingView(self):
        """Pre-aggregates the full dataset over customers and interactions providing a view of remaining offers.

        This pre-aggregation builds on the filter view and aggregates over
        the stages remaining.
        """
        self.preaggregated_decision_data_remainingview = (
            self.aggregate_remaining_per_stage(
                self.getPreaggregatedFilterView,
                list(self.preaggregation_columns),
                [
                    pl.col("Interaction_IDs").flatten().unique().count().alias("Decisions"),
                    pl.min("Decision Time_min"),
                    pl.max("Decision Time_max"),
                    pl.min("Value_min"),
                    pl.max("Value_max"),
                    # pl.col("Propensity").sample(
                    #     n=num_samples, with_replacement=True, shuffle=True
                    # ),  # a list sample values - for distribution plots
                    pl.first("Propensity"),
                    pl.min("Propensity_min"),
                    pl.max("Propensity_max"),
                    # pl.col("Priority")
                    # .sample(n=num_samples, with_replacement=True, shuffle=True)
                    # .alias("Priority"),
                    pl.first("Priority"),
                    pl.min("Priority_min"),
                    pl.max("Priority_max"),
                ]
                + [pl.sum(f"Win_at_rank{i}") for i in range(1, self.max_win_rank + 1)],
            )
            .collect()  # type: ignore[union-attribute]  # materialize to cache
            .lazy()  # re-wrap so downstream consumers stay lazy
        )
        return self.preaggregated_decision_data_remainingview

    @cached_property
    def sample(self):
        """Hash-based deterministic sample of interactions for resource-intensive analyses.

        Selects up to ``sample_size`` unique interactions using a hash of
        Interaction ID. All actions within a selected interaction are kept.
        If fewer interactions exist than ``sample_size``, no sampling is performed.

        When the ``--sample`` CLI flag is active, this operates on the
        already-reduced dataset, so two layers of sampling may apply.
        """
        needed_columns = [
            "Interaction ID",
            "Channel",
            "Direction",
            "Issue",
            "Group",
            "Stage Group",
            "Stage",
            "Stage Order",
            "Rank",
            "Priority",
            "Propensity",
            "Value",
            "Context Weight",
            "Levers",
            "Subject ID",
            "Record Type",
            "Action",
            "day",
            "is_mandatory",
        ]

        # Filter to only keep columns that exist in the data
        available_cols = set(self.decision_data.collect_schema().names())
        columns_to_keep = [col for col in needed_columns if col in available_cols]

        total_interaction_count = (
            self.decision_data.set_sorted("Interaction ID")
            .select(pl.n_unique("Interaction ID").alias("unique_count"))
            .collect()
            .item()
        )
        logger.debug("Sampling from %d total interactions", total_interaction_count)
        # Set num_sample_interactions attribute - use sample_size if we have more interactions than sample_size
        self.sample_size = min(total_interaction_count, self.sample_size)
        target_sample_size = self.sample_size
        sample_rate = min(1.0, target_sample_size / max(1, total_interaction_count))

        # Use hash-based sampling for efficiency - this is deterministic per interaction ID
        # but doesn't require collecting all unique IDs first, 15x faster than collecting 50k interactions first.
        try:
            df = (
                self.decision_data.select(columns_to_keep)
                .with_columns([(pl.col("Interaction ID").hash() % 1000 < 1000 * sample_rate).alias("_sample")])
                .filter(pl.col("_sample"))
                .drop("_sample")
                .collect()
                .shrink_to_fit()  # reclaim unused memory (DataFrame-only method)
                .lazy()  # re-wrap so downstream consumers stay lazy
            )
        except Exception as e:
            if "maximum length reached" in str(e).lower():
                raise RuntimeError(
                    "Dataset exceeds Polars' 32-bit row limit (2^32 rows). "
                    "Install the 64-bit runtime with: uv pip install 'polars[rt64]'"
                ) from e
            raise

        return df

    @property
    def filtered_sample(self):
        """Sample data with page-level filters applied.

        Reads filter expressions from st.session_state.page_channel_expr if available.
        Falls back to unfiltered sample if no page filters are set or not in Streamlit context.

        This property is not cached because it depends on mutable session_state.
        Page-level code should cache the result locally if needed for performance.

        Returns
        -------
        pl.LazyFrame
            Sampled data with page filters applied, or unfiltered sample if no filters.
        """
        try:
            import streamlit as st

            channel_expr = st.session_state.get("page_channel_expr", None)
            if channel_expr is not None:
                return apply_filter(self.sample, [channel_expr])
        except ImportError:
            pass  # Not in Streamlit context

        return self.sample

    def getAvailableFieldsForFiltering(self, categoricalOnly=False):
        if not categoricalOnly:
            return list(self.fields_for_data_filtering)

        string_cols = set(self.decision_data.select(cs.string(include_categorical=True)).collect_schema().keys())
        return [f for f in self.fields_for_data_filtering if f in string_cols]

    def cleanup_raw_data(self, df: pl.LazyFrame):
        """This method cleans up the raw data we read from parquet/S3/whatever.

        This likely needs to change as and when we get closer to product, to
        match what comes out of Pega. It does some modest type casting and
        potentially changing back some of the temporary column names that have
        been added to generate more data.
        """

        available = set(df.collect_schema().names())

        if "day" not in available:
            df = df.with_columns(day=pl.col("Decision Time").dt.date())

        # Build ranking columns — only include columns that actually exist
        ranking_cols: list[str | pl.Expr] = ["is_mandatory"]
        if "Priority" in available:
            ranking_cols.append("Priority")
        if "Stage Order" in available:
            ranking_cols.append("Stage Order")
        ranking_cols.extend(
            [
                pl.col("Issue").rank() * -1,
                pl.col("Group").rank() * -1,
                pl.col("Action").rank() * -1,
            ]
        )

        df = df.with_columns(
            Rank=pl.struct(ranking_cols).rank(descending=True, method="ordinal", seed=1).over("Interaction ID")
        )

        if self.extract_type == "explainability_extract":
            df = df.with_columns(
                pl.when(pl.col("Rank") == 1).then(pl.lit("Output")).otherwise(pl.lit("Arbitration")).alias(self.level),
                pl.when(pl.col("Rank") == 1).then(pl.lit(3800)).otherwise(pl.lit(10000)).alias("Stage Order"),
                pl.when(pl.col("Rank") == 1)
                .then(pl.lit("OUTPUT"))
                .otherwise(pl.lit("FILTERED_OUT"))
                .alias("Record Type"),
            )

        # Apply user-friendly display names to stage columns
        stage_cols = [c for c in ("Stage Group", "Stage") if c in available]
        if stage_cols:
            df = df.with_columns(
                [
                    pl.col(c).cast(pl.Utf8).map_elements(get_display_name, return_dtype=pl.Utf8).cast(pl.Categorical)
                    for c in stage_cols
                ]
            )

        preproc_df = (
            df.with_columns(pl.col(pl.Categorical).cast(pl.Utf8))
            .with_columns(
                pl.col(self.level).cast(pl.Categorical),
            )
            .filter(
                pl.col("Action").is_not_null()
            )  # Why do we have null pyName values? this takes too much processing time
        )
        return preproc_df

    def getPossibleScopeValues(self):
        available = set(self.decision_data.collect_schema().names())
        return [col for col in SCOPE_HIERARCHY if col in available]

    def getPossibleStageValues(self):
        options = self.AvailableNBADStages
        # TODO figure out how to get the actual possible values, should be available from the enum directly
        # [
        #     stage
        #     for stage in self.NBADStages_FilterView
        #     if stage in df['pxEgagementStage'].categories
        # ]
        return options

    @property
    def stage_to_group_mapping(self) -> dict[str, str]:
        """Map each Stage name to its Stage Group.

        Only meaningful when ``level == "Stage"`` and both columns exist.
        Returns an empty dict otherwise (including v1 / explainability data).
        """
        if self.level != "Stage":
            return {}
        available = set(self.unfiltered_raw_decision_data.collect_schema().names())
        if "Stage Group" not in available or "Stage" not in available:
            return {}
        mapping_df = self.unfiltered_raw_decision_data.select(["Stage", "Stage Group"]).unique().collect()
        return dict(
            zip(
                mapping_df.get_column("Stage").to_list(),
                mapping_df.get_column("Stage Group").to_list(),
            )
        )

    def getDistributionData(
        self,
        stage: str,
        grouping_levels: str | list[str],
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        distribution_data = (
            apply_filter(self.getPreaggregatedRemainingView, additional_filters)
            .filter(pl.col(self.level) == stage)
            .group_by(grouping_levels)
            .agg(pl.sum("Decisions"))
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
        )

        return distribution_data

    # import streamlit as st
    # @st.cache_data
    def getFunnelData(
        self, scope, additional_filters: pl.Expr | list[pl.Expr] | None = None
    ) -> tuple[pl.LazyFrame, pl.DataFrame]:
        # Apply filtering once to the pre-aggregated view
        filtered_df = apply_filter(self.getPreaggregatedFilterView, additional_filters)

        interaction_count_expr = (
            apply_filter(self.decision_data, additional_filters).select("Interaction ID").unique().count()
        )

        # Compute remaining actions funnel
        # Count total action occurrences (not unique interactions) for granularity-independent metric
        funnelData = self.aggregate_remaining_per_stage(
            df=filtered_df,
            group_by_columns=[scope],
            aggregations=[
                pl.sum("Decisions").alias("action_occurrences"),  # Total action occurrences for this scope value
                pl.col("Interaction_IDs")
                .flatten()
                .unique()
                .count()
                .alias("interaction_count_for_scope"),  # Unique interactions with this scope value
            ],
        ).filter(pl.col("action_occurrences") > 0)

        # Compute filtered funnel view
        filtered_funnel = (
            filtered_df.filter(pl.col("Record Type") == "FILTERED_OUT")
            .group_by([self.level, scope])
            .agg(
                action_occurrences=pl.sum("Decisions"),
                interaction_count_for_scope=pl.col("Interaction_IDs").flatten().unique().count(),
            )
            .collect()
        )

        interaction_count = interaction_count_expr.collect().item()
        # Calculate metrics:
        # - actions_per_interaction: total action occurrences / total interactions (granularity-independent)
        # - penetration_pct: percentage of interactions that have at least one action of this scope value
        metrics_expr = (
            pl.lit(interaction_count).alias("interaction_count"),
            (pl.col("action_occurrences") / pl.lit(interaction_count)).alias("actions_per_interaction"),
            ((pl.col("interaction_count_for_scope") / pl.lit(interaction_count)) * 100).alias("penetration_pct"),
        )

        return funnelData.with_columns(metrics_expr), filtered_funnel.with_columns(metrics_expr)

    def getFilterComponentData(self, top_n, additional_filters: pl.Expr | list[pl.Expr] | None = None) -> pl.DataFrame:
        group_cols = [self.level, "Component Name"]
        available = set(self.getPreaggregatedFilterView.collect_schema().names())
        if "Component Type" in available:
            group_cols.append("Component Type")

        stages_actions_df = (
            apply_filter(self.getPreaggregatedFilterView, additional_filters)
            .filter(pl.col(self.level) != "Output")
            .group_by(group_cols)
            .agg(pl.col("Interaction_IDs").flatten().unique().count().alias("Filtered Decisions"))
            .collect()
        )
        result = pl.concat(
            pl.collect_all(
                [
                    x.lazy().top_k(top_n, by="Filtered Decisions").sort("Filtered Decisions", descending=False)
                    for x in stages_actions_df.partition_by(self.level)
                ]
            )
        ).with_columns(pl.col(pl.Categorical).cast(pl.Utf8))

        return result

    def getComponentActionImpact(
        self,
        top_n: int = 10,
        scope: str = "Action",
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Per-component breakdown of which items are filtered and how many.

        For each component, returns the top-N items (at the chosen scope
        granularity) it filters out. The scope controls whether the breakdown
        is at Issue, Group, or Action level.

        Parameters
        ----------
        top_n : int, default 10
            Maximum number of items to return per component.
        scope : str, default "Action"
            Granularity level: ``"Issue"``, ``"Group"``, or ``"Action"``.
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters to apply before aggregation.

        Returns
        -------
        pl.DataFrame
            Columns include pxComponentName, StageGroup, scope columns, and
            Filtered Decisions. Sorted by component then descending count.
        """
        filtered_data = apply_filter(self.decision_data, additional_filters).filter(
            pl.col("Record Type") == "FILTERED_OUT"
        )

        # Build scope columns up to and including the requested level
        scope_hierarchy = ["Issue", "Group", "Action"]
        scope_idx = scope_hierarchy.index(scope) if scope in scope_hierarchy else 2
        scope_cols = scope_hierarchy[: scope_idx + 1]

        group_cols = ["Component Name", self.level] + scope_cols
        available = set(filtered_data.collect_schema().names())
        group_cols = [c for c in group_cols if c in available]

        impact_df = filtered_data.group_by(group_cols).agg(pl.len().alias("Filtered Decisions")).collect()

        # Top-N items per component
        result = pl.concat(
            pl.collect_all(
                [
                    part.lazy().top_k(top_n, by="Filtered Decisions").sort("Filtered Decisions", descending=True)
                    for part in impact_df.partition_by("Component Name")
                ]
            )
        ).with_columns(pl.col(pl.Categorical).cast(pl.Utf8))

        return result.sort(["Component Name", "Filtered Decisions"], descending=[False, True])

    def getComponentDrilldown(
        self,
        component_name: str,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Deep-dive into a single filter component showing dropped actions and
        their potential value.

        Since scoring columns (Priority, Value, Propensity) are typically null
        on FILTERED_OUT rows, this method derives the action's "potential value"
        by looking up average scores from rows where the same action survives
        (non-null Priority/Value). This gives the "value of what's being
        dropped" perspective.

        Parameters
        ----------
        component_name : str
            The pxComponentName to drill into.
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters to apply before aggregation.

        Returns
        -------
        pl.DataFrame
            Columns: pyIssue, pyGroup, pyName, Filtered Decisions,
            avg_Priority, avg_Value, avg_Propensity, pxComponentType (if
            available). Sorted by Filtered Decisions descending.
        """
        base = apply_filter(self.decision_data, additional_filters)
        available = set(base.collect_schema().names())

        # Filtered rows for this component
        filtered_rows = base.filter(
            (pl.col("Record Type") == "FILTERED_OUT") & (pl.col("Component Name") == component_name)
        )

        group_cols = ["Issue", "Group", "Action"]
        agg_exprs = [pl.len().alias("Filtered Decisions")]
        if "Component Type" in available:
            group_cols.append("Component Type")

        filtered_agg = filtered_rows.group_by(group_cols).agg(agg_exprs).collect()

        # Reference scores from surviving rows (non-null Priority)
        score_cols = ["Priority", "Value", "Propensity"]
        present_scores = [c for c in score_cols if c in available]

        if present_scores:
            score_aggs = [pl.col(c).mean().alias(f"avg_{c}") for c in present_scores]
            reference_scores = (
                base.filter(pl.col("Priority").is_not_null())
                .group_by(["Issue", "Group", "Action"])
                .agg(score_aggs)
                .collect()
            )
            result = filtered_agg.join(
                reference_scores,
                on=["Issue", "Group", "Action"],
                how="left",
            )
        else:
            result = filtered_agg

        return result.with_columns(pl.col(pl.Categorical).cast(pl.Utf8)).sort("Filtered Decisions", descending=True)

    def reRank(
        self,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
        overrides: list[pl.Expr] = [],
    ) -> pl.LazyFrame:
        """
        Calculates prio and rank for all PVCL combinations
        """
        rank_exprs = [
            pl.struct(
                [
                    "is_mandatory",
                    x,
                    "Stage Order",
                    pl.col("Issue").rank() * -1,
                    pl.col("Group").rank() * -1,
                    pl.col("Action").rank() * -1,
                ]
            )
            .rank(descending=True, method="ordinal", seed=1)
            .over(["Interaction ID"])
            .cast(pl.Int16)
            .alias(f"rank_{x.split('_')[1]}")
            for x in ["prio_PVCL", "prio_VCL", "prio_PCL", "prio_PVL", "prio_PVC"]
        ]

        # Fill missing PVCL components with 1.0 (neutral for multiplication).
        available = set(self.sample.collect_schema().names())
        pvcl_defaults = {
            "Propensity": 1.0,
            "Value": 1.0,
            "Context Weight": 1.0,
            "Levers": 1.0,
        }
        fill_exprs = []
        for col_name, default in pvcl_defaults.items():
            if col_name in available:
                fill_exprs.append(pl.col(col_name).fill_null(default))
            else:
                fill_exprs.append(pl.lit(default).alias(col_name))

        rank_df = (
            apply_filter(
                self.sample.with_columns(fill_exprs),
                additional_filters,
            )
            .with_columns(overrides)
            .filter(pl.col("Priority").is_not_null())
            .with_columns(
                prio_PVCL=(pl.col("Propensity") * pl.col("Value") * pl.col("Context Weight") * pl.col("Levers")),
                prio_VCL=(pl.col("Value") * pl.col("Context Weight") * pl.col("Levers")),
                prio_PCL=(pl.col("Propensity") * pl.col("Context Weight") * pl.col("Levers")),
                prio_PVL=(pl.col("Propensity") * pl.col("Value") * pl.col("Levers")),
                prio_PVC=(pl.col("Propensity") * pl.col("Value") * pl.col("Context Weight")),
            )
            .with_columns(*rank_exprs)
        )

        return rank_df

    # TODO consider making his more generic, dropping the win_rank argument and
    # creating a larger result set for all possible ranks, with filtering only
    # in the UI.
    def get_win_loss_distribution_data(self, level, win_rank, additional_filters=None):
        win_col = f"Win_at_rank{win_rank}"
        group_level_win_losses = (
            apply_filter(self.getPreaggregatedRemainingView, additional_filters)
            .filter(pl.col(self.level) == "Arbitration")
            .group_by(level)
            .agg(Wins=pl.sum(win_col), Decisions=pl.sum("Decisions"))
            .with_columns(Losses=pl.col("Decisions") - pl.col("Wins"))
            .with_columns(
                Wins=pl.col("Wins") / pl.sum("Wins"),
                Losses=pl.col("Losses") / pl.sum("Losses"),
            )
        )

        group_level_win_losses = group_level_win_losses.unpivot(
            index=level,
            on=["Wins", "Losses"],
            variable_name="Status",
            value_name="Percentage",
        ).sort(["Status", "Percentage"], descending=True)

        return group_level_win_losses

    def get_optionality_data(self, df):
        """
        Finding the average number of actions per stage without trend analysis.
        We have to go back to the interaction level data, no way to
        use pre-aggregations unfortunately.
        """
        total_interactions = df.select("Interaction ID").collect().unique().height
        expr = [
            pl.len().alias("nOffers"),
            pl.col("Propensity").filter(pl.col("Propensity") < 0.5).max().alias("bestPropensity"),
        ]
        per_offer_count_and_stage = (
            self.aggregate_remaining_per_stage(
                df=df,
                group_by_columns=["Interaction ID"],
                aggregations=expr,
            )
            .group_by(["nOffers", self.level])
            .agg(Interactions=pl.len(), AverageBestPropensity=pl.mean("bestPropensity"))
        )
        schema = per_offer_count_and_stage.collect_schema()
        zero_actions = (
            per_offer_count_and_stage.group_by(self.level)
            .agg(interaction_count=pl.sum("Interactions"))
            .with_columns(
                Interactions=(total_interactions - pl.col("interaction_count")).cast(schema["Interactions"]),
                AverageBestPropensity=pl.lit(0.0).cast(schema["AverageBestPropensity"]),
                nOffers=pl.lit(0).cast(schema["nOffers"]),
            )
            .drop("interaction_count")
        )
        optionality_data = pl.concat(
            [
                per_offer_count_and_stage,
                zero_actions.select(per_offer_count_and_stage.collect_schema().names()),
            ]
        ).sort("nOffers", descending=True)

        return optionality_data

    # @cached_property
    def get_optionality_data_with_trend(self, df=None):
        """
        Finding the average number of actions per stage with trend analysis.
        We have to go back to the interaction level data, no way to
        use pre-aggregations unfortunately.
        """
        if df is None:
            df = self.sample
        expr = [
            pl.len().alias("nOffers"),
            pl.col("Propensity").filter(pl.col("Propensity") < 0.5).max().alias("bestPropensity"),
        ]
        optionality_data = (
            self.aggregate_remaining_per_stage(
                df=df,
                group_by_columns=["Interaction ID", "day"],
                aggregations=expr,
            )
            .group_by(["nOffers", self.level, "day"])
            .agg(Interactions=pl.len(), AverageBestPropensity=pl.mean("bestPropensity"))
            .sort("nOffers", descending=True)
        )
        return optionality_data

    # @cached_property
    def get_optionality_funnel(self, df=None):
        if df is None:
            df = self.sample
        optionality_funnel = (
            self.get_optionality_data(df)
            .with_columns(
                pl.when(pl.col("nOffers") >= 7)
                .then(pl.lit("7+"))
                .otherwise(pl.col("nOffers").cast(str))
                .alias("available_actions")
            )
            .with_columns(
                pl.col("available_actions").cast(pl.Enum(["0", "1", "2", "3", "4", "5", "6", "7+"])),
            )
            .group_by([self.level, "available_actions"])
            .agg(pl.sum("Interactions"))
            .sort([self.level, "available_actions"])
        )
        return optionality_funnel

    def getActionVariationData(self, stage, color_by=None):
        """Get action variation data, optionally broken down by a categorical dimension.

        Args:
            stage: The stage to analyze
            color_by: Optional categorical column to break down the variation by.
                     Can use "Channel/Direction" to combine Channel and Direction columns.
        """
        # Handle combined Channel/Direction column
        needs_channel_direction = color_by == "Channel/Direction"
        if needs_channel_direction:
            grouping_levels = ["Channel", "Direction", "Action"]
        else:
            grouping_levels = ["Action"] if color_by is None else [color_by, "Action"]

        # Get distribution data grouped by Action (and optionally color_by dimension)
        distribution = self.getDistributionData(
            stage=stage,
            grouping_levels=grouping_levels,
        ).collect()

        # Create combined Channel/Direction column if needed
        if needs_channel_direction:
            distribution = distribution.with_columns(
                pl.concat_str("Channel", "Direction", separator="/").alias("Channel/Direction")
            )
            color_by_col = "Channel/Direction"
        else:
            color_by_col = color_by

        # Compute cumulative stats within each group if color_by is specified
        if color_by is not None:
            # Define final columns to keep
            final_columns = [
                "ActionIndex",
                "Action",
                "Decisions",
                "cumDecisions",
                "DecisionsFraction",
                "ActionsFraction",
                color_by_col,
            ]

            data = (
                distribution.sort([color_by_col, "Decisions"], descending=[False, True])
                .with_columns(cumDecisions=pl.col("Decisions").cum_sum().over(color_by_col).cast(pl.Int32))
                .with_columns(
                    DecisionsFraction=(pl.col("cumDecisions") / pl.col("Decisions").sum().over(color_by_col)),
                    Decisions=pl.col("Decisions").cast(pl.Int32),
                )
                .with_columns(ActionIndex=pl.int_range(pl.len()).over(color_by_col) + 1)
                .with_columns(
                    ActionsFraction=pl.col("ActionIndex") / pl.col("ActionIndex").max().over(color_by_col),
                    ActionIndex=pl.col("ActionIndex").cast(pl.UInt32),
                    Decisions=pl.col("Decisions").cast(pl.UInt32),
                    cumDecisions=pl.col("cumDecisions").cast(pl.UInt32),
                )
                .select(final_columns)
            )

            # Create zero points for each unique group value
            unique_groups = data[color_by_col].unique()
            zero_points = (
                pl.DataFrame(
                    {
                        color_by_col: unique_groups,
                        "ActionIndex": [0] * len(unique_groups),
                        "Action": [""] * len(unique_groups),
                        "Decisions": [0] * len(unique_groups),
                        "cumDecisions": [0] * len(unique_groups),
                        "DecisionsFraction": [0.0] * len(unique_groups),
                        "ActionsFraction": [0.0] * len(unique_groups),
                    }
                )
                .with_columns(
                    pl.col("ActionIndex").cast(pl.UInt32),
                    pl.col("Decisions").cast(pl.UInt32),
                    pl.col("cumDecisions").cast(pl.UInt32),
                )
                .select(final_columns)
            )

            result = pl.concat([zero_points, data]).sort([color_by_col, "ActionIndex"])
        else:
            data = (
                distribution.with_columns(cumDecisions=pl.col("Decisions").cum_sum().cast(pl.Int32))
                .with_columns(
                    DecisionsFraction=(pl.col("cumDecisions") / pl.sum("Decisions")),
                    Decisions=pl.col("Decisions").cast(pl.Int32),
                )
                .with_row_index("ActionIndex", 1)
                .with_columns(
                    ActionsFraction=pl.col("ActionIndex") / pl.len(),
                    ActionIndex=pl.col("ActionIndex").cast(pl.UInt32),
                    Decisions=pl.col("Decisions").cast(pl.UInt32),
                    cumDecisions=pl.col("cumDecisions").cast(pl.UInt32),
                )
            )

            # Single zero point for non-grouped data
            zero_point = pl.DataFrame(
                {
                    "ActionIndex": 0,
                    "Action": "",
                    "Decisions": 0,
                    "cumDecisions": 0,
                    "DecisionsFraction": 0.0,
                    "ActionsFraction": 0.0,
                }
            ).with_columns(
                pl.col("ActionIndex").cast(pl.UInt32),
                pl.col("Decisions").cast(pl.UInt32),
                pl.col("cumDecisions").cast(pl.UInt32),
            )

            result = pl.concat([zero_point, data])

        return result.lazy()

    # TODO: figure out how to main standard stage order, for now simply solved by sorting on counts
    def getABTestResults(self):
        tbl = (
            self.getPreaggregatedRemainingView.group_by(
                # TODO: we should include all the IA properties but they're not populated currently
                [self.level, "Model Control Group"]
            )
            .agg(pl.len())
            .collect()
            .pivot(
                index=self.level,
                values="len",
                columns=["Model Control Group"],
                sort_columns=True,
            )
            .with_columns((pl.col("Control") / (pl.col("Test") + pl.col("Control"))).alias("Control Percentage"))
            .sort("Test", descending=True)
        )
        return tbl

    def getThresholdingData(self, fld, quantile_range=range(10, 100, 10)):
        cache_key = (fld, tuple(quantile_range))
        if cache_key in self._thresholding_cache:
            return self._thresholding_cache[cache_key]

        thresholds_wide = (
            # Note: runs on pre-aggregated data - maybe should be using _min/_max for filtering instead
            self.getPreaggregatedFilterView.filter(pl.col(self.level).is_in(self.stages_from_arbitration_down))
            .select(
                # TODO can probably code this up more efficiently
                [pl.col(fld).explode().quantile(q / 100.0).alias(f"p{q}") for q in quantile_range]
                + [
                    ((pl.col(fld).explode()) < (pl.col(fld).explode().quantile(q / 100.0))).sum().alias(f"n{q}")
                    for q in quantile_range
                ]
                + [pl.lit("Arbitration").alias(self.level)]
            )
            .collect()
        )
        thresholds_long = (
            thresholds_wide.select([self.level] + [f"n{q}" for q in quantile_range])
            .unpivot(
                index=self.level,
                on=cs.numeric(),
                variable_name="Decile",
                value_name="Count",
            )
            .with_columns(pl.col("Decile").str.replace("n", "p"))
            .join(
                thresholds_wide.select([self.level] + [f"p{q}" for q in quantile_range]).unpivot(
                    index=self.level,
                    on=cs.numeric(),
                    variable_name="Decile",
                    value_name="Threshold",
                ),
                on=[self.level, "Decile"],
            )
            .sort([self.level, "Threshold"])
        ).filter(pl.col(self.level) == "Arbitration")
        self._thresholding_cache[cache_key] = thresholds_long
        return thresholds_long

    def priority_component_distribution(self, component, granularity, stage=None, additional_filters=None):
        """Data for a single component's distribution, grouped by granularity.

        Parameters
        ----------
        component : str
            Column name of the component to analyze.
        granularity : str
            Column to group by (e.g. "Issue", "Group", "Action").
        stage : str, optional
            Filter to actions remaining at this stage. If None, uses all
            rows with non-null Priority.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample (e.g. channel filter).
        """
        cols = [granularity, component]
        df = self._remaining_at_stage(stage, additional_filters=additional_filters)
        return df.select(cols).sort(granularity)

    def all_components_distribution(self, granularity, stage=None, additional_filters=None):
        """Data for the overview panel: all prioritization components at once.

        Parameters
        ----------
        granularity : str
            Column to group by.
        stage : str, optional
            Filter to actions remaining at this stage.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample (e.g. channel filter).
        """
        from .utils import PRIO_COMPONENTS

        available = set(self.sample.collect_schema().names())
        cols = [c for c in PRIO_COMPONENTS if c in available]
        df = self._remaining_at_stage(stage, additional_filters=additional_filters)
        return df.select([granularity] + cols).sort(granularity)

    def _remaining_at_stage(self, stage=None, additional_filters=None):
        """Return sample rows remaining at *stage*.

        Uses the ``aggregate_remaining_per_stage`` logic: rows whose stage
        order is >= the selected stage are "remaining" there.  If *stage*
        is None, falls back to rows with non-null Priority.
        """
        base = apply_filter(self.sample, additional_filters)
        if stage is None:
            return base.filter(pl.col("Priority").is_not_null())
        stage_idx = self.AvailableNBADStages.index(stage) if stage in self.AvailableNBADStages else 0
        remaining_stages = self.AvailableNBADStages[stage_idx:]
        return base.filter(pl.col(self.level).is_in(remaining_stages))

    def aggregate_remaining_per_stage(
        self, df: pl.LazyFrame, group_by_columns: list[str], aggregations: list = []
    ) -> pl.LazyFrame:
        """
        Workhorse function to convert the raw Decision Analyzer data (filter view) to
        the aggregates remaining per stage, ensuring all stages are represented.
        """
        stage_orders = (
            df.group_by(self.level)
            .agg(pl.min("Stage Order"))
            .with_columns(
                pl.col(self.level)
                .cast(pl.Utf8)
                .cast(
                    pl.Enum(self.AvailableNBADStages)
                )  # weird polars behaviour(version: 1.29), try removing in later patches
            )
        )

        def aggregate_over_remaining_stages(df, stage, remaining_stages):
            return (
                df.filter(pl.col(self.level).is_in(remaining_stages))
                .group_by(group_by_columns)
                .agg(aggregations)
                .with_columns(pl.lit(stage).alias(self.level))
            )

        aggs = {
            stage: aggregate_over_remaining_stages(df, stage, self.AvailableNBADStages[i:])
            for (i, stage) in enumerate(self.AvailableNBADStages)
        }
        remaining_view = (
            pl.concat(aggs.values())
            .with_columns(pl.col(self.level).cast(stage_orders.collect_schema()[self.level]))
            .join(stage_orders, on=self.level, how="left")
        )

        return remaining_view

    def filtered_action_counts(
        self,
        groupby_cols: list,
        propensityTH: float | None = None,
        priorityTH: float | None = None,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Return action counts from the sample, optionally classified by propensity/priority thresholds.

        Parameters
        ----------
        groupby_cols : list
            Column names to group by.
        propensityTH : float, optional
            Propensity threshold for classifying offers.
        priorityTH : float, optional
            Priority threshold for classifying offers.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample (e.g. channel filter).

        Returns
        -------
        pl.LazyFrame
            Aggregated action counts per group, with quality buckets when
            both thresholds are provided.
        """
        df = apply_filter(self.sample, additional_filters)
        additional_cols = ["Action", "Propensity"]
        required_cols = list(set(groupby_cols + additional_cols))
        for col in required_cols:
            if col not in df.collect_schema().names():
                raise ValueError(f"Column '{col}' not found in the dataframe.")

        if propensityTH is None or priorityTH is None:
            return df.group_by(groupby_cols).agg(
                no_of_offers=pl.count("Action"),
            )

        # Only classify actions by propensity/priority at stages where propensity is available
        # Actions filtered before ActionPropensity stage don't have meaningful propensity scores
        has_propensity = pl.col(self.level).is_in(self.stages_with_propensity)

        propensity_classifying_expr = [
            pl.col("Action")
            .filter(has_propensity & (pl.col("Propensity") == 0.5) & (pl.col(self.level) != "Output"))
            .count()
            .alias("new_models"),
            pl.col("Action")
            .filter(has_propensity & (pl.col("Propensity") < propensityTH) & (pl.col("Propensity") != 0.5))
            .count()
            .alias("poor_propensity_offers"),
            pl.col("Action")
            .filter(has_propensity & (pl.col("Priority") < priorityTH) & (pl.col("Propensity") != 0.5))
            .count()
            .alias("poor_priority_offers"),
            pl.col("Action")
            .filter(
                has_propensity
                & (pl.col("Propensity") >= propensityTH)
                & (pl.col("Propensity") != 0.5)
                & (pl.col("Priority") >= priorityTH)
            )
            .count()
            .alias("good_offers"),
        ]
        return df.group_by(groupby_cols).agg(no_of_offers=pl.count("Action"), *propensity_classifying_expr)

    def get_offer_quality(self, df, group_by):
        """
        Given a dataframe with filtered action counts at stages.
        Flips it to usual VF view by doing a rolling sum over stages.

        Parameters
        ----------
        df : pl.LazyFrame
            Decision Analyzer style filtered action counts dataframe.
        groupby_cols : list
            The list of column names to group by([self.level, "Interaction ID"]).

        Returns
        -------
        pl.LazyFrame
            Value Finder style, available action counts per group_by category
        """
        # Get all unique customers/interactions from the sample to ensure we count those with no actions
        if isinstance(group_by, str):
            group_by = [group_by]

        # Get the full set of customers from the FIRST stage (where all customers start)
        # This ensures we track all customers throughout the pipeline
        first_stage = self.AvailableNBADStages[0]
        all_customers = self.sample.filter(pl.col(self.level) == first_stage).select(group_by).unique().collect().lazy()

        dfs = []
        for i, stage in enumerate(self.AvailableNBADStages):
            # Get aggregated counts for this stage (only interactions WITH actions)
            stage_df = (
                # TODO refactor to use the remaining view aggregator (aggregate_remaining_per_stage)
                df.filter(pl.col(self.level).is_in(self.AvailableNBADStages[i:]))
                .group_by(group_by)
                .agg(
                    pl.sum(
                        "no_of_offers",
                        "new_models",
                        "poor_propensity_offers",
                        "poor_priority_offers",
                        "good_offers",
                    )
                )
                .with_columns(pl.lit(stage).alias(self.level))
            )

            # Left join with all customers to include those with no actions (fill with 0s)
            # Cast stage_df level column to categorical to match join keys
            stage_df = stage_df.with_columns(pl.col(self.level).cast(pl.Categorical))

            # Add stage to all customers and join
            stage_customers = all_customers.with_columns(pl.lit(stage).cast(pl.Categorical).alias(self.level))
            stage_df = stage_customers.join(stage_df, on=group_by + [self.level], how="left").with_columns(
                pl.col(
                    "no_of_offers", "new_models", "poor_propensity_offers", "poor_priority_offers", "good_offers"
                ).fill_null(0)
            )

            dfs.append(stage_df)
        stage_df = pl.concat(dfs)

        # Determine if stage has propensity scores available (ActionPropensity and later)
        has_propensity = pl.col(self.level).is_in(self.stages_with_propensity)

        stage_df = stage_df.with_columns(
            has_no_offers=pl.when(pl.col("no_of_offers") == 0).then(pl.lit(1)).otherwise(pl.lit(0)),
            # For stages WITH propensity: classify as relevant (green) or irrelevant (orange)
            atleast_one_relevant_action=pl.when(has_propensity & (pl.col("good_offers") >= 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
            only_irrelevant_actions=pl.when(
                has_propensity & (pl.col("good_offers") == 0) & (pl.col("no_of_offers") > 0)
            )
            .then(pl.lit(1))
            .otherwise(0),
            # For stages WITHOUT propensity: just show as having actions (blue)
            atleast_one_action=pl.when(~has_propensity & (pl.col("no_of_offers") > 0))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
        )
        return stage_df

    @cached_property
    def get_overview_stats(self):
        """Creates an overview from the full (filtered) dataset.

        Aggregate metrics (Decisions, Customers, Actions, Channels, Duration)
        are computed over ``decision_data`` so they reflect the true counts.
        Only the average-offers-per-stage KPI uses the sample (it requires
        interaction-level optionality analysis that would be too expensive on
        the full data).
        """

        nOffersPerStage = (
            self.get_optionality_data(self.sample)
            .group_by(self.level)
            .agg(pl.col("nOffers").mean().round().cast(pl.Int16))
            .collect()
        )

        def _offer_counts(stage):
            stage_values = nOffersPerStage[self.level].to_list()
            return (
                (nOffersPerStage.filter(pl.col(self.level) == stage).select("nOffers").item())
                if stage in stage_values
                else 0
            )

        # Use the full dataset for headline counts (not the sample)
        full_data = self.decision_data

        kpis = (
            (
                self.getPreaggregatedFilterView.select(
                    pl.n_unique("Action").alias("Actions"),
                    (
                        pl.struct("Channel", "Direction").n_unique()
                        if "Direction" in self.getPreaggregatedFilterView.collect_schema().names()
                        else pl.n_unique("Channel")
                    ).alias("Channels"),
                    ((pl.max("Decision Time_max") - pl.min("Decision Time_min")) + pl.duration(days=1)).alias(
                        "Duration"
                    ),
                    pl.min("Decision Time_min").cast(pl.Date).alias("StartDate"),
                )
            )
            .collect()
            .hstack(
                full_data.select(
                    (
                        pl.n_unique("Subject ID")
                        if "Subject ID" in full_data.collect_schema().names()
                        else pl.n_unique("Interaction ID")
                    ).alias("Customers"),
                    pl.n_unique("Interaction ID").alias("Decisions"),
                ).collect()
            )
            .hstack(
                pl.DataFrame(
                    {
                        "avgOffersAtArbitration": _offer_counts("Arbitration"),
                        "avgAvailable": _offer_counts("AvailableActions"),
                    }
                )
            )
        )

        return {k: kpis[k].item() for k in kpis.columns}

    def get_sensitivity(self, win_rank=1, filters=None, additional_filters=None):
        """Global or local sensitivity of the prioritization factors.

        Parameters
        ----------
        win_rank : int
            Maximum rank to be considered a winner.
        filters : pl.Expr, optional
            Selected offers, only used in local sensitivity analysis.
            When ``None`` (global), results are cached by ``win_rank``.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample before re-ranking
            (e.g. channel filter).  When set, caching is bypassed.

        Returns
        -------
        pl.LazyFrame
        """
        is_global_sensitivity = filters is None
        if is_global_sensitivity and additional_filters is None:
            if win_rank in self._sensitivity_cache:
                return self._sensitivity_cache[win_rank]
            filters = pl.col("Rank") <= win_rank
        elif is_global_sensitivity:
            filters = pl.col("Rank") <= win_rank

        sensitivity = (
            apply_filter(
                self.reRank(additional_filters=additional_filters), filters
            )  # don't put filters in rerank function, we need to filter after reranking!
            # .filter(pl.col("rank_PVCL") <= win_rank)
            .select(
                [
                    pl.col("Interaction ID")
                    .filter(pl.col(x) <= win_rank)
                    .n_unique()
                    .cast(pl.Int32)  # thinks they are unsigned int, 33-34 returns big number
                    .alias((f"{x.split('_')[1]}_win_count"))  # calculating win_counts of different combinations
                    for x in [
                        "rank_PVCL",
                        "rank_VCL",
                        "rank_PCL",
                        "rank_PVL",
                        "rank_PVC",
                    ]
                ]
            )
            .with_columns(
                [
                    (pl.col("PVCL_win_count") - pl.col(x)).alias(x)
                    for x in [
                        "VCL_win_count",
                        "PCL_win_count",
                        "PVL_win_count",
                        "PVC_win_count",
                    ]
                ]
            )
            .rename(
                {
                    "PVCL_win_count": "Priority",
                    "VCL_win_count": "Propensity",
                    "PCL_win_count": "Value",
                    "PVL_win_count": "Context Weights",
                    "PVC_win_count": "Levers",
                }
            )
            .unpivot(variable_name="Factor", value_name="Influence")
        )
        if is_global_sensitivity:
            sensitivity = sensitivity.with_columns(Influence=pl.col("Influence").abs())
            if additional_filters is None:
                self._sensitivity_cache[win_rank] = sensitivity
        return sensitivity

    def get_offer_variability_stats(self, stage):
        offer_variability_data = self.getActionVariationData(stage)
        return {
            "n90": bisect_left(
                offer_variability_data.select("DecisionsFraction").collect()["DecisionsFraction"],
                0.9,
            )
            - 1,  # first one is dummy
            "gini": 1.0
            - gini_coefficient(
                offer_variability_data.select(["ActionsFraction", "DecisionsFraction"]).collect(),
                "ActionsFraction",
                "DecisionsFraction",
            ),
        }

    def get_winning_or_losing_interactions(self, win_rank, group_filter, win: bool, additional_filters=None):
        if win:
            rank_filter = pl.col("Rank") <= win_rank
        else:
            rank_filter = pl.col("Rank") > win_rank
        return (
            apply_filter(apply_filter(self.sample, additional_filters), group_filter)
            .filter(rank_filter & (pl.col(self.level).is_in(self.stages_from_arbitration_down)))
            .select(pl.col("Interaction ID").unique())
        )

    def winning_from(self, interactions, win_rank, groupby_cols, top_k, additional_filters=None):
        winning_from = (
            apply_filter(self.sample, additional_filters)
            .filter(
                pl.col("Rank") > win_rank
            )  # TODO generalize this to any stage from Arbitration up but excluding Final
            .join(
                interactions,
                on="Interaction ID",
                how="inner",
            )
            .group_by(groupby_cols)
            .agg(Decisions=pl.len())
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
            .head(top_k)
        )
        return winning_from

    def losing_to(self, interactions, win_rank, groupby_cols, top_k, additional_filters=None):
        return (
            apply_filter(self.sample, additional_filters)
            .filter(pl.col("Rank") <= win_rank)
            .join(
                interactions,
                on="Interaction ID",
                how="inner",
            )
            .group_by(groupby_cols)
            .agg(Decisions=pl.len())
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
            .head(top_k)
        )

    def get_win_distribution_data(
        self,
        lever_condition: pl.Expr,
        lever_value: float | None = None,
        all_interactions: int | None = None,
    ) -> pl.DataFrame:
        """
        Calculate win distribution data for business lever analysis.

        This method generates distribution data showing how actions perform in
        arbitration decisions, both in baseline conditions and optionally with
        lever adjustments applied.

        Parameters
        ----------
        lever_condition : pl.Expr
            Polars expression defining which actions to apply the lever to.
            Example: pl.col("Action") == "SpecificAction" or
                    (pl.col("Issue") == "Service") & (pl.col("Group") == "Cards")
        lever_value : float, optional
            The lever multiplier value to apply to selected actions.
            If None, returns baseline distribution only.
            If provided, returns both original and lever-adjusted win counts.
        all_interactions : int, optional
            Total number of interactions to calculate "no winner" count.
            If provided, enables calculation of interactions without any winner.
            If None, "no winner" data is not calculated.

        Returns
        -------
        pl.DataFrame
            DataFrame containing win distribution with columns:
            - pyIssue, pyGroup, pyName: Action identifiers
            - original_win_count: Number of rank-1 wins in baseline scenario
            - new_win_count: Number of rank-1 wins after lever adjustment (only if lever_value provided)
            - n_decisions_survived_to_arbitration: Number of arbitration decisions the action participated in
            - selected_action: "Selected" for actions matching lever_condition, "Rest" for others
            - no_winner_count: Number of interactions without any winner (only if all_interactions provided)

        Notes
        -----
        - Only includes actions that survive to arbitration stage
        - Win counts represent rank-1 (first place) finishes in arbitration decisions
        - This is a zero-sum analysis: boosting selected actions suppresses others
        - Results are sorted by win count (new_win_count if available, else original_win_count)
        - When all_interactions is provided, "no winner" represents interactions without any rank-1 winner

        Examples
        --------
        Get baseline distribution for a specific action:
        >>> lever_cond = pl.col("Action") == "MyAction"
        >>> baseline = decision_analyzer.get_win_distribution_data(lever_cond)

        Get distribution with 2x lever applied to service actions:
        >>> lever_cond = pl.col("Issue") == "Service"
        >>> with_lever = decision_analyzer.get_win_distribution_data(lever_cond, 2.0)

        Get distribution with no winner count:
        >>> total_interactions = 10000
        >>> with_no_winner = decision_analyzer.get_win_distribution_data(lever_cond, 2.0, total_interactions)
        """
        if lever_value is None:
            # Return baseline distribution only
            original_winners = self.reRank(
                additional_filters=pl.col(self.level).is_in(self.stages_from_arbitration_down),
            ).select(["Issue", "Group", "Action"] + ["Interaction ID", "Rank"])

            result: pl.DataFrame = (  # type: ignore[assignment]
                original_winners.group_by(["Issue", "Group", "Action"])
                .agg(
                    original_win_count=pl.col("Rank").filter(pl.col("Rank") == 1).len(),
                    n_decisions_survived_to_arbitration=pl.col("Interaction ID").n_unique(),
                )
                .with_columns(
                    selected_action=pl.when(lever_condition).then(pl.lit("Selected")).otherwise(pl.lit("Rest"))
                )
                .collect()
                .sort("original_win_count", descending=True)
            )

            # Add no winner count if all_interactions is provided
            if all_interactions is not None:
                winner_df: pl.DataFrame = original_winners.select("Interaction ID").collect()  # type: ignore[assignment]
                interactions_with_winners = winner_df.n_unique()
                no_winner_count = max(0, all_interactions - interactions_with_winners)  # Ensure non-negative

                # Create a row with the same data types as the result
                no_winner_data = {
                    "Issue": ["No Winner"],
                    "Group": ["No Winner"],
                    "Action": ["No Winner"],
                    "original_win_count": [no_winner_count],
                    "n_decisions_survived_to_arbitration": [0],
                    "selected_action": ["No Winner"],
                }

                # Cast to match result schema
                no_winner_row = pl.DataFrame(no_winner_data).cast({k: v for k, v in result.schema.items()})

                result = pl.concat([result, no_winner_row])

            return result
        else:
            # Return both baseline and lever-adjusted distribution
            recalculated_winners = self.reRank(
                overrides=[
                    (pl.when(lever_condition).then(pl.lit(lever_value)).otherwise(pl.col("Levers"))).alias("Levers")
                ],
                additional_filters=pl.col(self.level).is_in(self.stages_from_arbitration_down),
            ).select(["Issue", "Group", "Action"] + ["Interaction ID", "Rank", "rank_PVCL"])

            result_lf = (
                recalculated_winners.group_by(["Issue", "Group", "Action"])
                .agg(
                    original_win_count=pl.col("Rank").filter(pl.col("Rank") == 1).len(),
                    new_win_count=pl.col("rank_PVCL").filter(pl.col("rank_PVCL") == 1).len(),
                    n_decisions_survived_to_arbitration=pl.col("Interaction ID").n_unique(),
                )
                .with_columns(
                    selected_action=pl.when(lever_condition).then(pl.lit("Selected")).otherwise(pl.lit("Rest"))
                )
            )
            result: pl.DataFrame = result_lf.collect().sort("new_win_count", descending=True)  # type: ignore[assignment]

            # Add no winner count if all_interactions is provided
            if all_interactions is not None:
                # Calculate no winner count based on new ranking
                new_winner_df: pl.DataFrame = (  # type: ignore[assignment]
                    recalculated_winners.filter(pl.col("rank_PVCL") == 1).select("Interaction ID").collect()
                )
                interactions_with_new_winners = new_winner_df.n_unique()
                no_winner_count = max(0, all_interactions - interactions_with_new_winners)  # Ensure non-negative

                # Create a row with the same data types as the result
                no_winner_data = {
                    "Issue": ["No Winner"],
                    "Group": ["No Winner"],
                    "Action": ["No Winner"],
                    "original_win_count": [0],  # No winner has no original wins
                    "new_win_count": [no_winner_count],
                    "n_decisions_survived_to_arbitration": [0],
                    "selected_action": ["No Winner"],
                }

                # Cast to match result schema
                no_winner_row = pl.DataFrame(no_winner_data).cast({k: v for k, v in result.schema.items()})

                result = pl.concat([result, no_winner_row])

            return result

    def get_trend_data(
        self,
        stage: str = "AvailableActions",
        scope: Literal["Group", "Issue", "Action"] | None = "Group",
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.DataFrame:
        stages = self.AvailableNBADStages[self.AvailableNBADStages.index(stage) :]
        group_by = ["day"] if scope is None else ["day", scope]

        trend_data = (
            apply_filter(self.sample, additional_filters)
            .filter(pl.col(self.level).is_in(stages))
            .group_by(group_by)
            .agg(pl.n_unique("Interaction ID").alias("Decisions"))
            .sort(group_by)
        )

        return trend_data

    def find_lever_value(
        self,
        lever_condition: pl.Expr,
        target_win_percentage: float,
        win_rank: int = 1,
        low: float = 0,
        high: float = 100,
        precision: float = 0.01,
        ranking_stages: list[str] | None = None,
    ) -> float:
        """
        Binary search algorithm to find lever value needed to achieve a desired win percentage.

        Parameters
        ----------
        lever_condition : pl.Expr
            Polars expression that defines which actions should receive the lever
        target_win_percentage : float
            The desired win percentage (0-100)
        win_rank : int, default 1
            Consider actions winning if they rank <= this value
        low : float, default 0
            Lower bound for lever search range
        high : float, default 100
            Upper bound for lever search range
        precision : float, default 0.01
            Search precision - smaller values give more accurate results
        ranking_stages : list[str], optional
            List of stages to include in analysis. Defaults to ["Arbitration"]

        Returns
        -------
        float
            The lever value needed to achieve the target win percentage

        Raises
        ------
        ValueError
            If the target win percentage cannot be achieved within the search range
        """
        if ranking_stages is None:
            ranking_stages = ["Arbitration"]

        def _calculate_action_win_percentage(lever: float) -> float:
            """Calculate win percentage for a given lever value"""
            ranked_df = self.reRank(
                overrides=[(pl.when(lever_condition).then(pl.lit(lever)).otherwise(pl.col("Levers"))).alias("Levers")],
                additional_filters=pl.col(self.level).is_in(self.stages_from_arbitration_down),
            ).filter(pl.col("rank_PVCL") <= win_rank)

            selected_wins_df: pl.DataFrame = (  # type: ignore[assignment]
                ranked_df.filter(lever_condition).select("Interaction ID").collect()
            )
            selected_wins = selected_wins_df.height
            selected_total_df: pl.DataFrame = ranked_df.select("Interaction ID").collect()  # type: ignore[assignment]  # noqa: E501
            selected_total = selected_total_df.height
            percentage = (selected_wins / selected_total) * 100
            return percentage

        beginning_high = high
        beginning_low = low

        # Check if target is achievable within bounds
        low_percentage = _calculate_action_win_percentage(beginning_low)
        high_percentage = _calculate_action_win_percentage(beginning_high)

        if target_win_percentage < low_percentage:
            raise ValueError(
                f"Target {target_win_percentage}% is too low. Even at lever {beginning_low}, your actions win in {low_percentage:.1f}% of interactions at arbitration."
                "You might have interactions where only your selected actions survive until arbitration. So they will win no matter what."
            )
        elif target_win_percentage > high_percentage:
            raise ValueError(
                f"Target {target_win_percentage}% is too high. Even at lever {beginning_high}, you only get {high_percentage:.1f}%. "
                f"You can increase the search range."
            )

        while high - low > precision:
            mid = (low + high) / 2

            current_win_percentage = _calculate_action_win_percentage(mid)

            if current_win_percentage < target_win_percentage:
                low = mid
            else:
                high = mid

        final_lever = (low + high) / 2
        return final_lever
