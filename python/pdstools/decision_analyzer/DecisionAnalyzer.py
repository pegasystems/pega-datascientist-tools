# python/pdstools/decision_analyzer/DecisionAnalyzer.py
from functools import cached_property
import logging
import os
import warnings

import polars as pl
import polars.selectors as cs

from .data_read_utils import validate_columns
from ._aggregates import Aggregates
from ._scoring import Scoring
from .plots import Plot
from .stage_grouping import DISPLAY_NAME_LOOKUP
from .column_schema import (
    DecisionAnalyzer as DecisionAnalyzer_TD,
    ExplainabilityExtract as ExplainabilityExtract_TD,
)
from .utils import (
    SCOPE_HIERARCHY,
    apply_filter,
    determine_extract_type,
    get_table_definition,
    rename_and_cast_types,
    resolve_aliases,
)
from ..pega_io.File import read_ds_export

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 10_000
"""Default number of unique interactions to sample for resource-intensive analyses."""

MANDATORY_PRIORITY_THRESHOLD = 4_999_999
"""Priority threshold at or above which actions are treated as mandatory by the
arbitration engine. Mandatory actions bypass normal ranking and always land in
the top slot. Used to auto-detect mandatory rows when no explicit
``mandatory_expr`` is supplied to :class:`DecisionAnalyzer`."""


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
    aggregates : Aggregates
        Accessor for aggregation queries (funnel, distribution,
        optionality, action variation, …).
    scoring : Scoring
        Accessor for re-ranking, sensitivity, win/loss and lever
        analysis.

    Examples
    --------
    >>> from pdstools import DecisionAnalyzer
    >>> da = DecisionAnalyzer.from_explainability_extract("data/sample_explainability_extract.parquet")
    >>> da.overview_stats
    >>> da.plot.sensitivity()
    >>> da.aggregates.get_funnel_data(scope="Action")
    >>> da.scoring.get_sensitivity()
    """

    # Lazily-set attribute populated when ``sample`` is first accessed.
    # Declared here so type-checkers see it on the class.
    _num_sample_interactions: int

    @classmethod
    def from_explainability_extract(
        cls,
        source: str | os.PathLike,
        *,
        level: str = "Stage Group",
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        mandatory_expr: pl.Expr | None = None,
        additional_columns: dict[str, pl.DataType] | None = None,
        num_samples: int = 1,
    ) -> "DecisionAnalyzer":
        """Create a DecisionAnalyzer from an Explainability Extract (v1) file.

        Parameters
        ----------
        source : str | os.PathLike
            Path to the Explainability Extract parquet file, or a URL.
        level, sample_size, mandatory_expr, additional_columns, num_samples
            See :meth:`__init__` for details.

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
        return cls(
            raw_data,
            level=level,
            sample_size=sample_size,
            mandatory_expr=mandatory_expr,
            additional_columns=additional_columns,
            num_samples=num_samples,
        )

    @classmethod
    def from_decision_analyzer(
        cls,
        source: str | os.PathLike,
        *,
        level: str = "Stage Group",
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        mandatory_expr: pl.Expr | None = None,
        additional_columns: dict[str, pl.DataType] | None = None,
        num_samples: int = 1,
    ) -> "DecisionAnalyzer":
        """Create a DecisionAnalyzer from a Decision Analyzer / EEV2 (v2) file.

        Parameters
        ----------
        source : str | os.PathLike
            Path to the Decision Analyzer parquet file, or a URL.
        level, sample_size, mandatory_expr, additional_columns, num_samples
            See :meth:`__init__` for details.

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
        return cls(
            raw_data,
            level=level,
            sample_size=sample_size,
            mandatory_expr=mandatory_expr,
            additional_columns=additional_columns,
            num_samples=num_samples,
        )

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
        *,
        level: str = "Stage Group",
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        mandatory_expr: pl.Expr | None = None,
        additional_columns: dict[str, pl.DataType] | None = None,
        num_samples: int = 1,
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
        num_samples : int, default 1
            Number of Propensity/Priority values to sample per preaggregation group.
            Higher values produce smoother quantile estimates in thresholding analysis
            at the cost of a proportionally larger pre-aggregated cache. Values above
            10–20 rarely add meaningful precision.
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

            When ``None`` (the default), mandatory rows are auto-detected from
            the ``Priority`` column using
            ``pl.col("Priority") >= MANDATORY_PRIORITY_THRESHOLD`` — mirroring
            how the arbitration engine treats high-priority actions.
        additional_columns : dict[str, pl.DataType], optional
            Additional columns to include in processing beyond the standard table definition.
            Dictionary mapping column names to their polars data types.

            Example: additional_columns = {"non_standard_column" : pl.Utf8}

        Notes
        -----
        The ranking function orders actions by: is_mandatory (desc) → Priority (desc) →
        StageOrder (desc) → Issue → Group → Action.

        If mandatory_expr is None, mandatory rows are auto-detected from
        ``Priority`` using :data:`MANDATORY_PRIORITY_THRESHOLD`. If the
        ``Priority`` column is unavailable, all rows are treated as
        non-mandatory.
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
        self.aggregates = Aggregates(self)
        self.scoring = Scoring(self)
        self.level = level
        self.sample_size = sample_size
        self._num_samples = num_samples
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
        # Bail out early if critical columns are missing — _cleanup_raw_data
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
        elif "Priority" in raw_data.collect_schema().names():
            raw_data = raw_data.with_columns(
                is_mandatory=(pl.col("Priority").fill_null(0) >= MANDATORY_PRIORITY_THRESHOLD).cast(pl.Int8)
            )
        else:
            raw_data = raw_data.with_columns(is_mandatory=pl.lit(0))
        self.decision_data = self._cleanup_raw_data(raw_data)
        available_columns = set(self.decision_data.collect_schema().names())
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

        # Validate that the requested level exists in the data; fall back to
        # the first available level if not (e.g. user passed default
        # "Stage Group" but data only has a "Stage" column).
        available_levels = self.available_levels
        if self.level not in available_levels:
            logger.info(
                "Requested level '%s' not present in data; falling back to '%s'.",
                self.level,
                available_levels[0],
            )
            self.level = available_levels[0]

        if self.extract_type == "explainability_extract":
            columns_to_remove = {"Component Name"}
            self.preaggregation_columns -= columns_to_remove

            self.NBADStages_FilterView = self.AvailableNBADStages
            self.NBADStages_RemainingView = self.AvailableNBADStages

        elif self.extract_type == "decision_analyzer":
            self._recompute_available_stages()

    @property
    def available_levels(self) -> list[str]:
        """Stage granularity levels available for this dataset.

        Returns ``["Stage Group", "Stage"]`` for Decision Analyzer (v2) data
        when both columns are present, or ``["Stage Group"]`` for
        Explainability Extract (v1) data where only synthetic stages exist.
        """
        if self.extract_type == "explainability_extract":
            return ["Stage Group"]
        available = set(self.decision_data.collect_schema().names())
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
        """Derive ``AvailableNBADStages`` from the data for the current level.

        At "Stage Group" level, synthetically injects "Arbitration" if it
        has no data rows (it is used as an anchor point by many analyses).

        At "Stage" level, detects Stage Groups that have no individual
        stages represented in the data and inserts the group name as a
        placeholder so the full pipeline is visible.
        """
        if self.extract_type == "explainability_extract":
            self.AvailableNBADStages = ["Arbitration", "Output"]
            return

        stage_df = self.decision_data.group_by(self.level).agg(pl.min("Stage Order")).collect()

        if self.level == "Stage Group":
            if "Arbitration" not in stage_df[self.level].to_list():
                arb = pl.DataFrame(
                    {self.level: "Arbitration", "Stage Order": 3800},
                    schema=stage_df.schema,
                )
                stage_df = pl.concat([stage_df, arb])
        elif self.level == "Stage":
            # Build the Stage Group pipeline and inject placeholders for
            # groups that have no individual stages in the data.
            available = set(self.decision_data.collect_schema().names())
            if "Stage Group" in available:
                group_df = self.decision_data.group_by("Stage Group").agg(pl.col("Stage Order").min()).collect()
                if "Arbitration" not in group_df["Stage Group"].to_list():
                    group_df = pl.concat(
                        [
                            group_df,
                            pl.DataFrame(
                                {"Stage Group": "Arbitration", "Stage Order": 3800},
                                schema=group_df.schema,
                            ),
                        ]
                    )
                group_df = group_df.sort("Stage Order")
                all_groups = group_df["Stage Group"].to_list()
                group_orders = dict(
                    zip(group_df["Stage Group"].to_list(), group_df["Stage Order"].to_list(), strict=False)
                )
                mapping = self.stage_to_group_mapping
                covered_groups = set(mapping.values())
                for grp in all_groups:
                    if grp not in covered_groups:
                        placeholder = pl.DataFrame(
                            {self.level: grp, "Stage Order": group_orders[grp]},
                            schema=stage_df.schema,
                        )
                        stage_df = pl.concat([stage_df, placeholder])

        stage_df = stage_df.sort("Stage Order")
        self.AvailableNBADStages = stage_df.get_column(self.level).to_list()

    @cached_property
    def mandatory_actions(self) -> set[str]:
        """Set of action names flagged as mandatory in the current data.

        Mandatory actions bypass normal arbitration and always rank in the
        top slot. Auto-detected from ``Priority`` (see
        :data:`MANDATORY_PRIORITY_THRESHOLD`) unless an explicit
        ``mandatory_expr`` was supplied at construction.

        Returns
        -------
        set[str]
            Distinct ``Action`` values where ``is_mandatory`` is truthy.
            Empty when no mandatory rows or no ``Action`` / ``is_mandatory``
            column is available.
        """
        schema_names = self.decision_data.collect_schema().names()
        if "is_mandatory" not in schema_names or "Action" not in schema_names:
            return set()
        names = (
            self.decision_data.filter(pl.col("is_mandatory") > 0)
            .select(pl.col("Action").unique())
            .collect()
            .get_column("Action")
            .to_list()
        )
        return {n for n in names if n is not None}

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
        messages: list[str] = []

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

            messages.append(
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

                messages.append(
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

        return messages[0] if messages else None

    @cached_property
    def arbitration_stage(self) -> pl.LazyFrame:
        """Sample rows remaining at or after the Arbitration stage."""
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
        return self._num_sample_interactions

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

    @cached_property
    def preaggregated_filter_view(self):
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
        # Number of Propensity/Priority values sampled per preaggregation group.
        # Each group contributes exactly num_samples values (sampled with
        # replacement), so equal weighting per group is preserved regardless of
        # num_samples. Higher values reduce quantile variance at the cost of a
        # proportionally larger pre-aggregated cache.
        num_samples = self._num_samples
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
    def preaggregated_remaining_view(self):
        """Pre-aggregates the full dataset over customers and interactions providing a view of remaining offers.

        This pre-aggregation builds on the filter view and aggregates over
        the stages remaining.
        """
        self.preaggregated_decision_data_remainingview = (
            self.aggregates.aggregate_remaining_per_stage(
                self.preaggregated_filter_view,
                list(self.preaggregation_columns),
                [
                    pl.col("Interaction_IDs")
                    .list.explode(keep_nulls=False, empty_as_null=False)
                    .unique()
                    .count()
                    .alias("Decisions"),
                    pl.min("Decision Time_min"),
                    pl.max("Decision Time_max"),
                    pl.min("Value_min"),
                    pl.max("Value_max"),
                    pl.col("Propensity")
                    .list.explode(keep_nulls=False, empty_as_null=False)
                    .sample(n=self._num_samples, with_replacement=True, shuffle=True),
                    pl.min("Propensity_min"),
                    pl.max("Propensity_max"),
                    pl.col("Priority")
                    .list.explode(keep_nulls=False, empty_as_null=False)
                    .sample(n=self._num_samples, with_replacement=True, shuffle=True),
                    pl.min("Priority_min"),
                    pl.max("Priority_max"),
                ]
                + [pl.sum(f"Win_at_rank{i}") for i in range(1, self.max_win_rank + 1)],
            )
            .collect()  # materialize to cache
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
            "Component Name",
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
        effective_sample_size = min(total_interaction_count, self.sample_size)
        self._num_sample_interactions = effective_sample_size
        target_sample_size = effective_sample_size
        sample_rate = min(1.0, target_sample_size / max(1, total_interaction_count))

        # Use hash-based sampling for efficiency - this is deterministic per interaction ID
        # but doesn't require collecting all unique IDs first, 15x faster than collecting 50k interactions first.
        # Compare hash directly against a scaled UInt64 threshold to avoid
        # modular-arithmetic collisions (hash % small_N can collapse to few buckets).
        try:
            base = self.decision_data.select(columns_to_keep)
            if sample_rate < 1.0:
                hash_threshold = int((2**64 - 1) * sample_rate)
                base = base.filter(pl.col("Interaction ID").hash() < hash_threshold)
            df = (
                base.collect()
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

    def filtered(self, filters: list[pl.Expr] | pl.Expr | None = None) -> pl.LazyFrame:
        """Return ``self.sample`` with the given filter expressions applied.

        Parameters
        ----------
        filters : list[pl.Expr] | pl.Expr | None, default None
            Filter expressions to AND together. ``None`` or an empty list
            returns the sample unchanged. Apps should construct this list
            from their own state (for Streamlit pages, see
            :func:`pdstools.app.decision_analyzer.da_streamlit_utils.collect_page_filters`);
            the library deliberately does not read UI state.

        Returns
        -------
        pl.LazyFrame
            The (possibly filtered) sample.
        """
        if filters is None:
            return self.sample
        if isinstance(filters, list) and not filters:
            return self.sample
        return apply_filter(self.sample, filters)

    def get_available_fields_for_filtering(self, *, categorical_only: bool = False) -> list[str]:
        """Return column names available for data filtering.

        Parameters
        ----------
        categorical_only : bool, default False
            If True, return only string/categorical columns.
        """
        if not categorical_only:
            return list(self.fields_for_data_filtering)

        string_cols = set(self.decision_data.select(cs.string(include_categorical=True)).collect_schema().keys())
        return [f for f in self.fields_for_data_filtering if f in string_cols]

    def _cleanup_raw_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
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
                    pl.col(c)
                    .cast(pl.Utf8)
                    .replace_strict(DISPLAY_NAME_LOOKUP, default=pl.col(c).cast(pl.Utf8), return_dtype=pl.Utf8)
                    .cast(pl.Categorical)
                    for c in stage_cols
                ]
            )

        preproc_df = (
            df.with_columns(pl.col(pl.Categorical).cast(pl.Utf8))
            .with_columns(
                pl.col(self.level).cast(pl.Categorical),
            )
            .filter(pl.col("Action").is_not_null())  # Drop rows with null Action; this takes some processing time
        )
        return preproc_df

    def get_possible_scope_values(self) -> list[str]:
        """Return scope hierarchy columns present in the data (e.g. Issue, Group, Action)."""
        available = set(self.decision_data.collect_schema().names())
        return [col for col in SCOPE_HIERARCHY if col in available]

    def get_possible_stage_values(self) -> list[str]:
        """Return the list of available stage values for the current level."""
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
        available = set(self.decision_data.collect_schema().names())
        if "Stage Group" not in available or "Stage" not in available:
            return {}
        mapping_df = self.decision_data.select(["Stage", "Stage Group"]).unique().collect()
        return dict(
            zip(
                mapping_df.get_column("Stage").to_list(),
                mapping_df.get_column("Stage Group").to_list(),
                strict=False,
            )
        )

    # TODO consider making this more generic by returning all rank views in one pass.

    @cached_property
    def overview_stats(self) -> dict[str, object]:
        """Creates an overview from the full (filtered) dataset.

        Aggregate metrics (Decisions, Customers, Actions, Channels, Duration)
        are computed over ``decision_data`` so they reflect the true counts.
        Only the average-offers-per-stage KPI uses the sample (it requires
        interaction-level optionality analysis that would be too expensive on
        the full data).
        """

        nOffersPerStage = (
            self.aggregates.get_optionality_data(self.sample)
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
                self.preaggregated_filter_view.select(
                    pl.n_unique("Action").alias("Actions"),
                    (
                        pl.struct("Channel", "Direction").n_unique()
                        if "Direction" in self.preaggregated_filter_view.collect_schema().names()
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
