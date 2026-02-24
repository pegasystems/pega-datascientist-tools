# python/pdstools/decision_analyzer/utils.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type, Union

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
    table_definition : Dict
        Column definitions with 'display_name', 'default', and 'type' keys
    raw_columns : Set[str]
        Column names present in the raw data
    """

    table_definition: Dict
    raw_columns: Set[str]

    # Results populated by resolve()
    rename_mapping: Dict[str, str] = field(default_factory=dict, init=False)
    type_mapping: Dict[str, Type[pl.DataType]] = field(default_factory=dict, init=False)
    columns_to_drop: List[str] = field(default_factory=list, init=False)
    final_columns: List[str] = field(default_factory=list, init=False)
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

        resolved_targets: Dict[str, str] = {}  # display_name -> actual column name
        columns_needing_rename: Dict[str, str] = {}  # raw_col -> display_name

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
            if (
                display_name in resolved_targets
                and display_name not in self.final_columns
            ):
                self.final_columns.append(display_name)

        self._resolved = True
        return self

    def get_missing_columns(self) -> List[str]:
        """Get list of required columns missing from the raw data.

        Returns
        -------
        List[str]
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


def apply_filter(
    df: pl.LazyFrame, filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None
):
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
        raise ValueError(
            f"Filters should be a pl.Expr or a list of pl.Expr. Got {type(filters)}"
        )

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


def get_first_level_stats(
    interaction_data: pl.LazyFrame, filters: List[pl.Expr] = None
):
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
    *table_definitions: Dict,
) -> pl.LazyFrame:
    """Rename alias columns to their canonical raw key names before validation.

    Scans all table definitions for ``aliases`` entries. If an alias is found
    in the data but neither the raw key nor the display_name is present, the
    column is renamed to the raw key so downstream processing can find it.

    Parameters
    ----------
    df : pl.LazyFrame
        Raw data that may use alternative column names.
    *table_definitions : Dict
        One or more table definition dicts (DecisionAnalyzer, ExplainabilityExtract).

    Returns
    -------
    pl.LazyFrame
        Data with alias columns renamed to canonical raw key names.
    """
    raw_cols = set(df.collect_schema().names())
    renames: Dict[str, str] = {}

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

    The heuristic is: if any column name matches the raw key, display name, or
    aliases for the ``pxStrategyName`` entry in the DecisionAnalyzer table
    definition, the data is v2.
    """
    strategy_config = DecisionAnalyzer.get("pxStrategyName", {})
    strategy_names = {"pxStrategyName"}
    if strategy_config:
        strategy_names.add(strategy_config["display_name"])
        strategy_names.update(strategy_config.get("aliases", []))

    available = set(raw_data.collect_schema().names())
    return (
        "decision_analyzer" if strategy_names & available else "explainability_extract"
    )


def rename_and_cast_types(
    df: pl.LazyFrame,
    table_definition: Dict,
) -> pl.LazyFrame:
    """Rename columns and cast data types based on table definition.

    Performs a single-pass rename from raw column keys to display names,
    then casts types for default columns.

    Parameters
    ----------
    df : pl.LazyFrame
        The input dataframe to process
    table_definition : Dict
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


def _cast_columns(
    df: pl.LazyFrame, type_mapping: Dict[str, Type[pl.DataType]]
) -> pl.LazyFrame:
    """Cast columns to their target types.

    Parameters
    ----------
    df : pl.LazyFrame
        The dataframe to process
    type_mapping : Dict[str, Type[pl.DataType]]
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
    selected_issue: Optional[str] = None,
    selected_group: Optional[str] = None,
    selected_action: Optional[str] = None,
) -> Dict[str, Dict[str, Union[List[str], int]]]:
    """
    Create hierarchical filter options and calculate indices for selectbox widgets.

    Args:
        data: LazyFrame with hierarchical data (should be pre-filtered to desired stage)
        selected_issue: Currently selected issue (optional)
        selected_group: Currently selected group (optional)
        selected_action: Currently selected action (optional)

    Returns:
        Dict with structure:
        {
            "issues": {"options": [...], "index": 0},
            "groups": {"options": ["All", ...], "index": 0},
            "actions": {"options": ["All", ...], "index": 0}
        }
    """

    # Step 1: Get all available issues
    available_issues = (
        data.select("Issue").unique().collect().get_column("Issue").to_list()
    )
    issue_index = 0
    if selected_issue and selected_issue in available_issues:
        issue_index = available_issues.index(selected_issue)

    # Use the selected issue (or first one if none selected)
    current_issue = (
        selected_issue if selected_issue in available_issues else available_issues[0]
    )

    # Step 2: Get groups for current issue
    filtered_by_issue = data.filter(pl.col("Issue") == current_issue)
    available_groups = (
        filtered_by_issue.select("Group")
        .unique()
        .collect()
        .get_column("Group")
        .to_list()
    )
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
        filtered_by_issue_group = filtered_by_issue.filter(
            pl.col("Group") == current_group
        )

    available_actions = (
        filtered_by_issue_group.select("Action")
        .unique()
        .collect()
        .get_column("Action")
        .to_list()
    )
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
) -> Dict[str, Union[str, pl.Expr, List[str]]]:
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
    Dict[str, Union[str, pl.Expr, List[str]]]
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
            "lever_condition": (pl.col("Issue") == selected_issue)
            & (pl.col("Group") == selected_group),
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
