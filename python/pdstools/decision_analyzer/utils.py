# python/pdstools/decision_analyzer/utils.py
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..utils.cdh_utils import parse_pega_date_time_formats
from .column_schema import (
    DecisionAnalyzer,
    ExplainabilityExtract,
)

# As long as this is run once, anywhere, it's enabled globally.
# Putting it here AND in the Home.py file should therefore be enough,
# because every other file imports from utils.py (hence running this part too.)
pl.enable_string_cache()


@dataclass
class ColumnResolver:
    """Resolves column mappings between raw data and a standardized schema.

    Raw decision data can come from multiple sources with different schemas:
    - Explainability Extract vs Decision Analyzer exports
    - Inbound vs Outbound channel data

    For example, channel information may appear as:
    - 'Channel' (already using the display name)
    - 'pyChannel' (an alias for the display name)
    - 'Primary_ContainerPayload_Channel' (raw name needing rename)
    - Both raw key and display_name present (conflict requiring resolution)

    This class normalizes these variations by:
    - Mapping raw column names to standardized display names
    - Resolving conflicts when both raw and display_name columns exist
    - Building the final schema with consistent column names

    Attributes
    ----------
    table_definition : dict
        Column definitions with 'display_name', 'default', and 'type' keys
    raw_columns : set[str]
        Column names present in the raw data
    """

    table_definition: dict
    raw_columns: set[str]

    # Results populated by resolve()
    rename_mapping: dict[str, str] = field(default_factory=dict, init=False)
    type_mapping: dict[str, type[pl.DataType]] = field(default_factory=dict, init=False)
    columns_to_drop: list[str] = field(default_factory=list, init=False)
    final_columns: list[str] = field(default_factory=list, init=False)
    _resolved: bool = field(default=False, init=False)

    def __post_init__(self):
        self.resolve()

    def resolve(self) -> "ColumnResolver":
        """Resolve all column mappings and conflicts.

        Returns
        -------
        ColumnResolver
            Self, for method chaining
        """
        if self._resolved:
            return self

        resolved_targets: dict[str, str] = {}  # display_name -> actual column name
        columns_needing_rename: dict[str, str] = {}  # raw_col -> display_name

        for raw_col, config in self.table_definition.items():
            display_name = config["display_name"]
            raw_exists = raw_col in self.raw_columns
            target_exists = display_name in self.raw_columns
            needs_rename = raw_col != display_name

            if not raw_exists and not target_exists:
                continue

            # Only cast default columns; non-default columns have unreliable type definitions.
            if raw_exists and target_exists and needs_rename:
                # Conflict: both exist - prefer the already-correctly-named column
                self.columns_to_drop.append(raw_col)
                resolved_targets[display_name] = display_name
                if config["default"]:
                    self.type_mapping[display_name] = config["type"]
            elif raw_exists:
                if needs_rename:
                    columns_needing_rename[raw_col] = display_name
                resolved_targets[display_name] = raw_col
                if config["default"]:
                    self.type_mapping[raw_col] = config["type"]
            elif target_exists:
                resolved_targets[display_name] = display_name
                if config["default"]:
                    self.type_mapping[display_name] = config["type"]

        remaining_columns = self.raw_columns - set(self.columns_to_drop)
        for raw_col, display_name in columns_needing_rename.items():
            if raw_col in remaining_columns:
                self.rename_mapping[raw_col] = display_name

        for raw_col, config in self.table_definition.items():
            display_name = config["display_name"]
            if display_name in resolved_targets and display_name not in self.final_columns:
                self.final_columns.append(display_name)

        self._resolved = True
        return self

    def get_missing_columns(self) -> list[str]:
        """Get list of required columns missing from the raw data.

        Returns
        -------
        list[str]
            Column names that are marked as default but not found in raw data
        """
        missing = []
        for raw_col, config in self.table_definition.items():
            if not config["default"]:
                continue
            display_name = config["display_name"]
            if raw_col not in self.raw_columns and display_name not in self.raw_columns:
                missing.append(raw_col)
        return missing


# Scope hierarchy for plots and UI dropdowns.
# Keys are the display names used in the data after renaming.
SCOPE_HIERARCHY = ["Issue", "Group", "Action"]

# Priority factors (PVCL) used in arbitration scoring and sensitivity analysis.
PRIO_FACTORS = ["Propensity", "Value", "Context Weight", "Levers"]

# All prioritization components including the computed Priority itself.
PRIO_COMPONENTS = PRIO_FACTORS + ["Priority"]


def apply_filter(df: pl.LazyFrame, filters: pl.Expr | list[pl.Expr] | None = None):
    """
    Apply a global set of filters. Kept outside of the DecisionData class as
    this is really more of a utility function, not bound to that class at all.
    """

    def _apply(item, df):
        col_diff = set(item.meta.root_names()) - set(df.collect_schema().names())
        if len(col_diff) == 0:
            df = df.filter(item)
        else:
            from polars.exceptions import ColumnNotFoundError

            raise ColumnNotFoundError(col_diff)
        return df

    if filters is None:
        return df
    elif isinstance(filters, pl.Expr):
        df = _apply(filters, df)

    elif isinstance(filters, list) and all(isinstance(i, pl.Expr) for i in filters):
        for item in filters:
            df = _apply(item, df)
    else:
        raise ValueError(f"Filters should be a pl.Expr or a list of pl.Expr. Got {type(filters)}")

    return df


def area_under_curve(df: pl.DataFrame, col_x: str, col_y: str):
    return (
        df.with_columns(
            ((pl.col(col_y) + pl.col(col_y).shift(1)) / 2).alias("MeanY"),
            (pl.col(col_x) - pl.col(col_x).shift(1)).alias("DeltaX"),
        )
        .drop_nulls()
        .select((pl.col("DeltaX") * pl.col("MeanY")).sum())
        .item()
    )


def gini_coefficient(df: pl.DataFrame, col_x: str, col_y: str):
    return area_under_curve(df, col_x, col_y) * 2 - 1


def get_first_level_stats(interaction_data: pl.LazyFrame, filters: list[pl.Expr] | None = None):
    """Returns first-level stats of a dataframe for the filter summary.

    Shows unique actions (Issue/Group/Action combinations), unique
    interactions (decisions), and total rows so users understand the
    impact of their filters.
    """
    action_path = ["Issue", "Group", "Action"]
    schema = apply_filter(interaction_data, filters).collect_schema()
    has_interaction_id = "Interaction ID" in schema.names()

    select_exprs = [
        pl.struct(action_path).n_unique().alias("Actions"),
        pl.len().alias("row_count"),
    ]
    if has_interaction_id:
        select_exprs.append(pl.n_unique("Interaction ID").alias("interaction_count"))

    counts = apply_filter(interaction_data, filters).select(select_exprs).collect()

    stats = {
        "Actions": counts.get_column("Actions").item(),
        "Rows": counts.get_column("row_count").item(),
    }
    if has_interaction_id:
        stats["Interactions"] = counts.get_column("interaction_count").item()
    return stats


def resolve_aliases(
    df: pl.LazyFrame,
    *table_definitions: dict,
) -> pl.LazyFrame:
    """Rename alias columns to their canonical raw key names before validation.

    Scans all table definitions for ``aliases`` entries. If an alias is found
    in the data but neither the raw key nor the display_name is present, the
    column is renamed to the raw key so downstream processing can find it.

    Parameters
    ----------
    df : pl.LazyFrame
        Raw data that may use alternative column names.
    *table_definitions : dict
        One or more table definition dicts (DecisionAnalyzer, ExplainabilityExtract).

    Returns
    -------
    pl.LazyFrame
        Data with alias columns renamed to canonical raw key names.
    """
    raw_cols = set(df.collect_schema().names())
    renames: dict[str, str] = {}

    # Collect all raw keys across all table definitions so we never rename
    # a column that is itself a canonical raw key in another definition.
    all_raw_keys = set()
    for table_def in table_definitions:
        all_raw_keys.update(table_def.keys())

    for table_def in table_definitions:
        for canonical, config in table_def.items():
            aliases = config.get("aliases", [])
            if not aliases:
                continue
            # Only rename if neither the canonical raw key nor the display_name are present
            display_name = config["display_name"]
            if canonical in raw_cols or display_name in raw_cols:
                continue
            for alias in aliases:
                if alias in raw_cols and alias not in renames:
                    # Don't rename if the alias is itself a raw key in another table definition
                    if alias in all_raw_keys and alias != canonical:
                        continue
                    renames[alias] = canonical
                    break

    return df.rename(renames) if renames else df


def determine_extract_type(raw_data):
    """Detect whether the data is a Decision Analyzer (v2) or Explainability Extract (v1).

    V2 data must have both a strategy name column *and* stage pipeline columns
    (``Stage_pyStageGroup`` / ``Stage Group``). Data that has strategy names
    but no stage information (e.g. pre-aggregated or anonymized exports) is
    treated as v1 so the synthetic-stage fallback is used.
    """
    available = set(raw_data.collect_schema().names())

    strategy_config = DecisionAnalyzer.get("pxStrategyName", {})
    strategy_names = {"pxStrategyName"}
    if strategy_config:
        strategy_names.add(strategy_config["display_name"])
        strategy_names.update(strategy_config.get("aliases", []))

    has_strategy = bool(strategy_names & available)

    stage_config = DecisionAnalyzer.get("Stage_pyStageGroup", {})
    stage_names = {"Stage_pyStageGroup"}
    if stage_config:
        stage_names.add(stage_config["display_name"])
        stage_names.update(stage_config.get("aliases", []))

    has_stages = bool(stage_names & available)

    return "decision_analyzer" if (has_strategy and has_stages) else "explainability_extract"


def rename_and_cast_types(
    df: pl.LazyFrame,
    table_definition: dict,
) -> pl.LazyFrame:
    """Rename columns and cast data types based on table definition.

    Performs a single-pass rename from raw column keys to display names,
    then casts types for default columns.

    Parameters
    ----------
    df : pl.LazyFrame
        The input dataframe to process
    table_definition : dict
        Dictionary containing column definitions with 'display_name', 'default',
        and 'type' keys

    Returns
    -------
    pl.LazyFrame
        Processed dataframe with renamed columns and cast types
    """
    resolver = ColumnResolver(
        table_definition=table_definition,
        raw_columns=set(df.collect_schema().names()),
    )

    if resolver.columns_to_drop:
        df = df.drop(resolver.columns_to_drop)

    df = _cast_columns(df, resolver.type_mapping)

    return df.rename(resolver.rename_mapping).select(resolver.final_columns)


def _cast_columns(df: pl.LazyFrame, type_mapping: dict[str, type[pl.DataType]]) -> pl.LazyFrame:
    """Cast columns to their target types.

    Parameters
    ----------
    df : pl.LazyFrame
        The dataframe to process
    type_mapping : dict[str, type[pl.DataType]]
        Mapping of column names to their target types

    Returns
    -------
    pl.LazyFrame
        Dataframe with columns cast to target types
    """
    schema = df.collect_schema()
    for col_name, target_type in type_mapping.items():
        if col_name not in schema.names():
            continue
        current_type = schema[col_name]
        if current_type != target_type:
            if target_type == pl.Datetime:
                df = df.with_columns(parse_pega_date_time_formats(col_name))
            else:
                df = df.with_columns(pl.col(col_name).cast(target_type))
    return df


def get_table_definition(table: str):
    mapping = {
        "decision_analyzer": DecisionAnalyzer,
        "explainability_extract": ExplainabilityExtract,
    }
    if table not in mapping:
        raise ValueError(f"Unknown table: {table}")
    return mapping[table]


def create_hierarchical_selectors(
    data: pl.LazyFrame,
    selected_issue: str | None = None,
    selected_group: str | None = None,
    selected_action: str | None = None,
) -> dict[str, dict[str, list[str] | int]]:
    """
    Create hierarchical filter options and calculate indices for selectbox widgets.

    Args:
        data: LazyFrame with hierarchical data (should be pre-filtered to desired stage)
        selected_issue: Currently selected issue (optional)
        selected_group: Currently selected group (optional)
        selected_action: Currently selected action (optional)

    Returns:
        dict with structure:
        {
            "issues": {"options": [...], "index": 0},
            "groups": {"options": ["All", ...], "index": 0},
            "actions": {"options": ["All", ...], "index": 0}
        }
    """

    # Step 1: Get all available issues
    issues_df: pl.DataFrame = data.select("Issue").unique().collect()  # type: ignore[assignment]
    available_issues = issues_df.get_column("Issue").to_list()
    issue_index = 0
    if selected_issue and selected_issue in available_issues:
        issue_index = available_issues.index(selected_issue)

    # Use the selected issue (or first one if none selected)
    current_issue = selected_issue if selected_issue in available_issues else available_issues[0]

    # Step 2: Get groups for current issue
    filtered_by_issue = data.filter(pl.col("Issue") == current_issue)
    groups_df: pl.DataFrame = (
        filtered_by_issue.select("Group").unique().collect()  # type: ignore[assignment]
    )
    available_groups = groups_df.get_column("Group").to_list()
    group_options = ["All"] + available_groups
    group_index = 0  # Default to "All"
    if selected_group and selected_group in group_options:
        group_index = group_options.index(selected_group)

    # Use the selected group (or "All" if none selected/invalid)
    current_group = selected_group if selected_group in group_options else "All"

    # Step 3: Get actions for current issue+group
    if current_group == "All":
        filtered_by_issue_group = filtered_by_issue
    else:
        filtered_by_issue_group = filtered_by_issue.filter(pl.col("Group") == current_group)

    actions_df: pl.DataFrame = (
        filtered_by_issue_group.select("Action").unique().collect()  # type: ignore[assignment]
    )
    available_actions = actions_df.get_column("Action").to_list()
    action_options = ["All"] + available_actions
    action_index = 0  # Default to "All"
    if selected_action and selected_action in action_options:
        action_index = action_options.index(selected_action)

    return {
        "issues": {"options": available_issues, "index": issue_index},
        "groups": {"options": group_options, "index": group_index},
        "actions": {"options": action_options, "index": action_index},
    }


def get_scope_config(
    selected_issue: str, selected_group: str, selected_action: str
) -> dict[str, str | pl.Expr | list[str]]:
    """
    Generate scope configuration for lever application and plotting based on user selections.

    Parameters
    ----------
    selected_issue : str
        Selected issue value from dropdown (can be "All")
    selected_group : str
        Selected group value from dropdown (can be "All")
    selected_action : str
        Selected action value from dropdown (can be "All")

    Returns
    -------
    dict[str, str | pl.Expr | list[str]]
        Configuration dictionary containing:
        - level: "Action", "Group", or "Issue" indicating scope level
        - lever_condition: Polars expression for filtering selected actions
        - group_cols: List of column names for grouping operations
        - x_col: Column name to use for x-axis in plots
        - selected_value: The actual selected value for highlighting
        - plot_title_prefix: Prefix for plot titles
    """
    if selected_action != "All":
        return {
            "level": "Action",
            "lever_condition": pl.col("Action") == selected_action,
            "group_cols": ["Issue", "Group", "Action"],
            "x_col": "Action",
            "selected_value": selected_action,
            "plot_title_prefix": "Win Count by Action",
        }
    elif selected_group != "All":
        return {
            "level": "Group",
            "lever_condition": (pl.col("Issue") == selected_issue) & (pl.col("Group") == selected_group),
            "group_cols": ["Issue", "Group"],
            "x_col": "Group",
            "selected_value": selected_group,
            "plot_title_prefix": "Win Count by Group",
        }
    else:
        return {
            "level": "Issue",
            "lever_condition": pl.col("Issue") == selected_issue,
            "group_cols": ["Issue"],
            "x_col": "Issue",
            "selected_value": selected_issue,
            "plot_title_prefix": "Win Count by Issue",
        }


logger = logging.getLogger(__name__)

_INTERACTION_ID_RAW_KEY = "pxInteractionID"


def _get_interaction_id_candidates() -> list[str]:
    """Build the set of possible interaction ID column names from the schema.

    Collects the raw key, display name, and aliases from both table
    definitions so this stays in sync with ``column_schema.py``.
    """
    candidates: list[str] = [_INTERACTION_ID_RAW_KEY]
    for table_def in (DecisionAnalyzer, ExplainabilityExtract):
        config = table_def.get(_INTERACTION_ID_RAW_KEY)
        if config is None:
            continue
        display = config["display_name"]
        if display not in candidates:
            candidates.append(display)
        for alias in config.get("aliases", []):
            if alias not in candidates:
                candidates.append(alias)
    return candidates


def _find_interaction_id_column(columns: set[str]) -> str:
    """Return the first matching interaction ID column name from the data."""
    for candidate in _get_interaction_id_candidates():
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Cannot sample: no interaction ID column found. Looked for: {', '.join(_get_interaction_id_candidates())}"
    )


def _determine_output_directory(source_path: str | None, output_dir: str | None) -> Path:
    """Determine the best output directory for cached/sampled files.

    Priority order:
    1. If output_dir is explicitly provided, use that
    2. Otherwise, if source_path is a file and its directory is writeable, use that
    3. Otherwise, fall back to current directory

    Parameters
    ----------
    source_path : str | None
        Path to the source file.
    output_dir : str | None
        Explicitly requested output directory (takes precedence if provided).

    Returns
    -------
    Path
        Directory to use for output.
    """
    # Explicit output_dir takes precedence
    if output_dir:
        return Path(output_dir)

    # Try to use source file's directory if it's a file and writeable
    if source_path:
        source = Path(source_path)
        if source.is_file():
            parent_dir = source.parent
            # Check if directory is writeable
            if parent_dir.exists() and parent_dir.is_dir():
                try:
                    # Test writeability by checking permissions
                    if os.access(parent_dir, os.W_OK):
                        logger.info("Using source file directory for output: %s", parent_dir)
                        return parent_dir
                    else:
                        logger.debug("Source directory %s is not writeable, using fallback", parent_dir)
                except Exception as e:
                    logger.debug("Error checking writeability of %s: %s", parent_dir, e)

    # Fall back to current directory
    return Path(".")


def sample_interactions(
    df: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    id_column: str | None = None,
    use_random: bool = False,
    total_interactions: int | None = None,
) -> pl.LazyFrame:
    """Sample interactions from a LazyFrame before ingestion.

    By default, uses deterministic hash-based filtering so the same data and limit
    always produce the same sample. When sampling already-sampled data, uses random
    sampling to avoid bias from repeated deterministic sampling.

    All rows belonging to a selected interaction are kept (stratified on interaction ID).

    Exactly one of *n* or *fraction* must be provided.

    Parameters
    ----------
    df : pl.LazyFrame
        Raw data to sample from.
    n : int, optional
        Maximum number of unique interactions to keep.
    fraction : float, optional
        Fraction of interactions to keep (0.0–1.0).
    id_column : str, optional
        Name of the interaction ID column. Auto-detected if not given.
    use_random : bool, default False
        If True, use random sampling instead of deterministic hash-based sampling.
        This should be set when sampling already-sampled data to avoid bias.
    total_interactions : int, optional
        Pre-computed total number of unique interactions. If provided, avoids
        an expensive full-data scan to count them.

    Returns
    -------
    pl.LazyFrame
        Filtered LazyFrame containing only the sampled interactions.
    """
    if (n is None) == (fraction is None):
        raise ValueError("Exactly one of 'n' or 'fraction' must be provided.")

    available = set(df.collect_schema().names())
    if id_column is None:
        id_column = _find_interaction_id_column(available)

    # Random sampling (for already-sampled data or when explicitly requested)
    # This must be checked BEFORE hash-based sampling to avoid double-hashing bias
    if use_random:
        # Random sampling requires collecting unique IDs
        unique_ids_df = df.select(id_column).unique().collect()
        total = len(unique_ids_df)

        if fraction is not None:
            # Convert fraction to target count
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"fraction must be in (0, 1], got {fraction}")
            target_n = int(total * fraction)
            if target_n >= total:
                logger.info("Fraction %.2f yields %d >= %d total, skipping.", fraction, target_n, total)
                return df
            logger.info(
                "sample_interactions: RANDOM sampling, target=%d of %d (%.1f%%)", target_n, total, fraction * 100
            )
            sampled_ids_df = unique_ids_df.sample(n=target_n, shuffle=True, seed=None)
        else:
            # n-based random sampling
            assert n is not None
            if total <= n:
                logger.info("Data has %d interactions (≤ requested %d), skipping.", total, n)
                return df
            logger.info("sample_interactions: RANDOM sampling, target=%d of %d", n, total)
            sampled_ids_df = unique_ids_df.sample(n=n, shuffle=True, seed=None)

        return df.join(sampled_ids_df.lazy(), on=id_column, how="semi")

    # Hash-based sampling (for fresh data, no double-hashing)
    # For fraction-based sampling, apply filter lazily
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        threshold = int(fraction * 10_000)
        logger.info(
            "sample_interactions: LAZY hash filter, fraction=%.2f, threshold=%d",
            fraction,
            threshold,
        )
        return df.filter(pl.col(id_column).hash() % 10_000 < threshold)

    # n-based hash sampling
    assert n is not None

    # Deterministic hash-based sampling for n interactions
    # If we know the total, compute an exact threshold
    if total_interactions is not None:
        if total_interactions <= n:
            logger.info("Data has %d interactions (≤ requested %d), skipping.", total_interactions, n)
            return df
        threshold = int((n / total_interactions) * 10_000)
        logger.info(
            "sample_interactions: LAZY hash filter, n=%d of %d, threshold=%d",
            n,
            total_interactions,
            threshold,
        )
        return df.filter(pl.col(id_column).hash() % 10_000 < threshold)

    # No total known — estimate unique interactions from a small sample of data.
    # Reading a 1% hash slice is much cheaper than scanning all rows for unique IDs.
    logger.info("Estimating interaction count from 1%% sample...")
    sample_slice = df.filter(pl.col(id_column).hash() % 100 < 1)
    sample_unique = sample_slice.select(pl.n_unique(id_column)).collect().item()
    estimated_total = sample_unique * 100  # Scale up from 1% sample

    if estimated_total <= n:
        logger.info("Estimated %d interactions (≤ requested %d), skipping sampling.", estimated_total, n)
        return df

    # Add 10% buffer to threshold to compensate for estimation variance
    raw_threshold = int((n / estimated_total) * 10_000)
    threshold = max(1, min(int(raw_threshold * 1.1), 10_000))
    logger.info(
        "sample_interactions: LAZY hash filter (estimated), n=%d, est_total=%d, threshold=%d",
        n,
        estimated_total,
        threshold,
    )
    return df.filter(pl.col(id_column).hash() % 10_000 < threshold)


def prepare_and_save(
    df: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str | None = None,
    source_path: str | None = None,
) -> tuple[pl.LazyFrame, Path | None]:
    """Prepare data for analysis by sampling or caching, and persist as parquet.

    **Sampling mode** (when n or fraction provided):
    Writes ``decision_analyzer_sample_<count>.parquet`` into *output_dir*
    (defaults to the current working directory). Returns a LazyFrame scanning
    the written file plus the file path.

    **Caching mode** (when neither n nor fraction provided):
    Writes ``decision_analyzer_cache_<count>.parquet`` into *output_dir*
    with 100% sample metadata. Useful for caching non-parquet sources (CSV,
    JSON, ZIP) for faster reloading.

    The parquet file includes metadata tracking:
    - Original source file path
    - Sample percentage relative to original data (100% for caching mode)
    - Whether percentage was calculated exactly or approximated

    If sampling is requested but the data is smaller than the requested sample,
    sampling is skipped and the original LazyFrame is returned unchanged
    (no file is written).

    Parameters
    ----------
    df : pl.LazyFrame
        Raw data to process.
    n : int, optional
        Maximum number of unique interactions to keep (sampling mode).
    fraction : float, optional
        Fraction of interactions to keep 0.0–1.0 (sampling mode).
    output_dir : str, optional
        Directory for the output parquet file. If not provided, defaults to the
        source file's directory (when source is a file and directory is writeable),
        otherwise current directory ``"."``.
    source_path : str, optional
        Path to the original source file for metadata tracking and determining
        output directory.

    Returns
    -------
    tuple[pl.LazyFrame, Path | None]
        The (possibly sampled/cached) LazyFrame and the path to the written
        parquet file, or ``None`` when no file was written.

    Examples
    --------
    Sample data and save with metadata:

    >>> df = pl.scan_parquet("large_data.parquet")
    >>> sampled, path = prepare_and_save(
    ...     df,
    ...     n=100000,
    ...     source_path="large_data.parquet"
    ... )
    >>> print(path)
    decision_analyzer_sample_100k.parquet

    Cache non-parquet data:

    >>> df = pl.scan_csv("export.csv")
    >>> cached, path = prepare_and_save(
    ...     df,
    ...     source_path="export.csv"
    ... )
    >>> print(path)
    decision_analyzer_cache_87k.parquet

    Read metadata from a prepared file:

    >>> import polars as pl
    >>> metadata = pl.read_parquet_metadata("decision_analyzer_sample_100k.parquet")
    >>> print(metadata["pdstools:source_file"])
    large_data.parquet
    >>> print(metadata["pdstools:sample_percentage"])
    10.0
    """
    # Determine mode: sampling or caching
    is_sampling = (n is not None) or (fraction is not None)

    # Skip caching if no source path provided
    if not is_sampling and not source_path:
        return df, None

    available = set(df.collect_schema().names())
    id_column = _find_interaction_id_column(available)

    # Step 1: Read source metadata if available
    source_metadata = None
    original_source = source_path or "unknown"
    if source_path:
        source_metadata = _read_source_metadata(source_path)
        if source_metadata:
            # Inherit original source from chained sampling
            original_source = source_metadata["source_file"]

    # Step 2: Process data based on mode
    if is_sampling:
        # Detect if source was already sampled (use random sampling to avoid hash bias)
        use_random = source_metadata is not None
        if use_random:
            logger.info("Source is already sampled - using random sampling to avoid hash-based bias")

        # Perform sampling — fully lazy for hash-based, collects IDs only for random
        sampled_lf = sample_interactions(
            df,
            n=n,
            fraction=fraction,
            id_column=id_column,
            use_random=use_random,
        )

        # Check if sampling was actually needed (when n-based sampling is requested)
        if n is not None:
            actual_count = sampled_lf.select(pl.n_unique(id_column)).collect().item()
            if actual_count <= n:
                logger.info("Data has %d interactions (≤ requested %d), skipping file write.", actual_count, n)
                return df, None
    else:
        sampled_lf = df

    # Step 3: Determine output directory and write via sink_parquet (streaming)
    prefix = "decision_analyzer_sample_" if is_sampling else "decision_analyzer_cache_"
    dest = _determine_output_directory(source_path, output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Use a temporary name first, rename after we know the count
    import uuid

    tmp_name = f"{prefix}{uuid.uuid4().hex[:8]}.parquet"
    tmp_path = dest / tmp_name

    logger.info("Streaming data to %s", tmp_path)
    try:
        sampled_lf.sink_parquet(tmp_path)
    except Exception:
        # sink_parquet may not support all query plans; fall back to collect
        logger.info("sink_parquet failed, falling back to collect + write_parquet")
        sampled_lf.collect(streaming=True).write_parquet(tmp_path)

    # Step 4: Read back to get the actual count and compute metadata
    result_lf = pl.scan_parquet(tmp_path)
    processed_count = result_lf.select(pl.n_unique(id_column)).collect().item()

    # Calculate sample percentage (approximate for hash-based large data)
    if is_sampling:
        if fraction is not None:
            sample_percentage = fraction * 100.0
            method = "exact"
        elif n is not None:
            # We don't know the exact total without a full scan, so approximate
            sample_percentage = 0.0
            method = "approximated"
        else:
            sample_percentage = 0.0
            method = "unknown"
    else:
        sample_percentage = 100.0
        method = "exact"

    # Step 5: Apply inheritance if sampling a sample
    if source_metadata and is_sampling:
        source_pct = source_metadata["sample_percentage"]
        assert isinstance(source_pct, float)
        sample_percentage = (sample_percentage * source_pct) / 100.0
        if source_metadata["method"] == "approximated":
            method = "approximated"

    formatted_count = format_count_for_filename(processed_count)
    logger.info(
        "Processed %d interactions%s",
        processed_count,
        f" ({sample_percentage:.1f}% of original)" if is_sampling and sample_percentage < 100 else "",
    )

    # Step 6: Rename to final path with count
    base_path = dest / f"{prefix}{formatted_count}.parquet"
    out_path = base_path
    counter = 1
    while out_path.exists():
        logger.info("File %s already exists, adding suffix", out_path.name)
        out_path = dest / f"{prefix}{formatted_count}_{counter}.parquet"
        counter += 1

    tmp_path.rename(out_path)

    # Step 7: Write metadata by re-reading and re-writing with metadata
    # (parquet metadata can only be set at write time)
    metadata = {
        "pdstools:source_file": original_source,
        "pdstools:sample_percentage": f"{sample_percentage:.2f}",
        "pdstools:sample_percentage_method": method,
    }
    logger.info("Writing metadata to %s", out_path)
    result_df = pl.read_parquet(out_path)
    result_df.write_parquet(out_path, metadata=metadata)

    return pl.scan_parquet(out_path), out_path


def parse_sample_flag(value: str) -> dict[str, int | float]:
    """Parse the ``--sample`` CLI flag value into keyword arguments.

    Delegates to :func:`pdstools.utils.streamlit_utils.parse_sample_spec`.
    """
    from pdstools.utils.streamlit_utils import parse_sample_spec

    return parse_sample_spec(value)


def format_count_for_filename(count: int) -> str:
    """Format an interaction count for use in filenames.

    Uses human-readable abbreviations with 2 significant figures.

    Parameters
    ----------
    count : int
        Number of interactions.

    Returns
    -------
    str
        Formatted count (e.g., "87k", "1.2M", "2.5B").

    Examples
    --------
    >>> format_count_for_filename(42)
    '42'
    >>> format_count_for_filename(1500)
    '1.5k'
    >>> format_count_for_filename(87432)
    '87k'
    >>> format_count_for_filename(1234567)
    '1.2M'
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        # Thousands
        value = count / 1000
        rounded = round(value)

        if rounded >= 1000:
            # Transition to millions (e.g., 999,500 → 1M)
            millions = count / 1_000_000
            rounded_m = round(millions)
            if rounded_m >= 10:
                return f"{rounded_m}M"
            elif rounded_m >= 1:
                # For values 1-9M, show as integer if it rounds cleanly
                return f"{rounded_m}M"
            else:
                formatted = f"{millions:.1f}".rstrip("0").rstrip(".")
                return f"{formatted}M"
        elif rounded >= 10:
            return f"{rounded}k"
        else:
            # Use 1 decimal for < 10
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}k"
    elif count < 1_000_000_000:
        # Millions
        value = count / 1_000_000
        rounded = round(value)

        if rounded >= 1000:
            # Transition to billions
            billions = count / 1_000_000_000
            rounded_b = round(billions)
            if rounded_b >= 10:
                return f"{rounded_b}B"
            elif rounded_b >= 1:
                # For values 1-9B, show as integer if it rounds cleanly
                return f"{rounded_b}B"
            else:
                formatted = f"{billions:.1f}".rstrip("0").rstrip(".")
                return f"{formatted}B"
        elif rounded >= 10:
            return f"{rounded}M"
        else:
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}M"
    else:
        # Billions
        value = count / 1_000_000_000
        rounded = round(value)

        if rounded >= 10:
            return f"{rounded}B"
        else:
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}B"


def should_cache_source(source_path: str | None) -> bool:
    """Return True if source should be cached as parquet.

    Caching is beneficial for non-parquet sources (CSV, JSON, ZIP, directories)
    but unnecessary for single parquet files which are already optimized.

    Parameters
    ----------
    source_path : str | None
        Path to the source file or directory.

    Returns
    -------
    bool
        True if source should be cached, False otherwise.

    Examples
    --------
    >>> should_cache_source("/data/export.csv")
    True
    >>> should_cache_source("/data/export.parquet")
    False
    >>> should_cache_source(None)
    False
    """
    if not source_path:
        return False
    path = Path(source_path)
    # Skip if source is already a single parquet file
    if path.is_file() and path.suffix == ".parquet":
        return False
    return True


def _read_source_metadata(source_path: str) -> dict[str, str | float] | None:
    """Read pdstools metadata from a parquet file if it exists.

    Parameters
    ----------
    source_path : str
        Path to the parquet file to check.

    Returns
    -------
    dict or None
        Dictionary with keys: source_file, sample_percentage, method
        Returns None if file doesn't exist, is not parquet, or lacks metadata.
    """
    try:
        metadata = pl.read_parquet_metadata(source_path)

        # Check if this file has our metadata
        source_file = metadata.get("pdstools:source_file")
        if source_file is None:
            return None

        return {
            "source_file": source_file,
            "sample_percentage": float(metadata.get("pdstools:sample_percentage", "0")),
            "method": metadata.get("pdstools:sample_percentage_method", "unknown"),
        }
    except Exception:
        # File doesn't exist, not a parquet, or other read error
        return None
