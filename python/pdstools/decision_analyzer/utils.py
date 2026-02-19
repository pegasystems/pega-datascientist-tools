from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type, Union

import polars as pl

from ..utils.cdh_utils import parse_pega_date_time_formats
from .table_definition import (
    DecisionAnalyzer,
    ExplainabilityExtract,
    # audit_tag_mapping,
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
    - 'pyChannel' (already using the target name), this column doesn't exist on Inbound calls
    - 'Primary_ContainerPayload_Channel' (raw name needing rename),
    this column has the channel information on Inbound calls but it is null on Outbound calls
    - Both columns present (conflict requiring resolution)

    This class normalizes these variations by:
    - Mapping raw column names to standardized target labels
    - Resolving conflicts when both raw and target columns exist
    - Building the final schema with consistent column names

    Attributes
    ----------
    table_definition : Dict
        Column definitions with 'label' (target name), 'default', and 'type' keys
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

        resolved_targets: Dict[str, str] = {}  # target_label -> actual column name
        columns_needing_rename: Dict[str, str] = {}  # raw_col -> target_label

        for raw_col, config in self.table_definition.items():
            target_label = config["label"]
            raw_exists = raw_col in self.raw_columns
            target_exists = target_label in self.raw_columns
            needs_rename = raw_col != target_label

            if not raw_exists and not target_exists:
                continue

            # Only cast default columns; non-default columns have unreliable type definitions.
            if raw_exists and target_exists and needs_rename:
                # Conflict: both exist - prefer the already-correctly-named column
                self.columns_to_drop.append(raw_col)
                resolved_targets[target_label] = target_label
                if config["default"]:
                    self.type_mapping[target_label] = config["type"]
            elif raw_exists:
                if needs_rename:
                    columns_needing_rename[raw_col] = target_label
                resolved_targets[target_label] = raw_col
                if config["default"]:
                    self.type_mapping[raw_col] = config["type"]
            elif target_exists:
                resolved_targets[target_label] = target_label
                if config["default"]:
                    self.type_mapping[target_label] = config["type"]

        remaining_columns = self.raw_columns - set(self.columns_to_drop)
        for raw_col, target_label in columns_needing_rename.items():
            if raw_col in remaining_columns:
                self.rename_mapping[raw_col] = target_label

        for raw_col, config in self.table_definition.items():
            target_label = config["label"]
            if (
                target_label in resolved_targets
                and target_label not in self.final_columns
            ):
                self.final_columns.append(target_label)

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
            target_label = config["label"]
            if raw_col not in self.raw_columns and target_label not in self.raw_columns:
                missing.append(raw_col)
        return missing


# superset, in order
# TODO add treatment and subset against available columns

NBADScope_Mapping = {
    "pyIssue": "Issue",
    "pyGroup": "Group",
    "pyName": "Action",
    # , "Treatment" : "Treatment"
}


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


# TODO refactor this into the DecisionData class and refactor to use the remaining view aggregator (aggregate_remaining_per_stage)
def filtered_action_counts(
    df: pl.LazyFrame,
    groupby_cols: list,
    propensityTH: float = None,
    priorityTH: float = None,
) -> pl.LazyFrame:
    """
    Returns a DataFrame with action counts filtered based on the given propensity and priority thresholds.

    Parameters
    ----------
    df : pl.LazyFrame
        The input dataframe.
    groupby_cols : list
        The list of column names to group by(["pxEngagementStage", "pxInteractionID"]).
    propensityTH : float
        The propensity threshold.
    priorityTH : float
        The priority threshold.

    Returns
    -------
    pl.LazyFrame
        A DataFrame with action counts filtered based on the given propensity and priority thresholds.
    """

    additional_cols = [
        "pyName",
        "Propensity",
        "Priority",
    ]
    required_cols = list(set(groupby_cols + additional_cols))
    # TODO below is a pattern (to check required columns) we probably need all over the place - but maybe at the level of the streamlit pages
    for col in required_cols:
        if col not in df.collect_schema().names():
            raise ValueError(f"Column '{col}' not found in the dataframe.")

    propensity_classifying_expr = [
        pl.col("pyName")
        .where((pl.col("Propensity") == 0.5) & (pl.col("StageGroup") != "Output"))
        .count()
        .alias("new_models"),
        pl.col("pyName")
        .where((pl.col("Propensity") < propensityTH) & (pl.col("Propensity") != 0.5))
        .count()
        .alias("poor_propensity_offers"),
        pl.col("pyName")
        .where((pl.col("Priority") < priorityTH) & (pl.col("Propensity") != 0.5))
        .count()
        .alias("poor_priority_offers"),
        pl.col("pyName")
        .where(
            (pl.col("Propensity") >= propensityTH)
            & (pl.col("Propensity") != 0.5)
            & (pl.col("Priority") >= priorityTH)
        )
        .count()
        .alias("good_offers"),
    ]
    if propensityTH is None or priorityTH is None:
        filtered_action_counts = df.group_by(groupby_cols).agg(
            no_of_offers=pl.count("pyName"),
        )
    else:
        filtered_action_counts = df.group_by(groupby_cols).agg(
            no_of_offers=pl.count("pyName"), *propensity_classifying_expr
        )

    return filtered_action_counts


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


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
def get_first_level_stats(
    interaction_data: pl.LazyFrame, filters: List[pl.Expr] = None
):
    """
    Returns some first level stats of a dataframe. Used to
    show effects of user data filters.
    """
    action_path = [
        "pyIssue",
        "pyGroup",
        "pyName",
    ]  # TODO generalize context keys, match with available data
    counts = (
        apply_filter(interaction_data, filters)
        .select(
            pl.struct(action_path).n_unique().alias("Actions"),
            pl.count().alias("row_count"),
            # pl.col("pyName")
            # .unique()
            # .implode(),  # pyName hardcoded here... and what is this implode for?
        )
        .collect()
    )

    return {
        "Actions": counts.get_column("Actions").item(),
        "Rows": counts.get_column("row_count").item(),
    }


def determine_extract_type(raw_data):
    return (
        "decision_analyzer"
        if "pxStrategyName" in raw_data.collect_schema().names()
        else "explainability_extract"
    )


def rename_and_cast_types(
    df: pl.LazyFrame,
    table_definition: Dict,
) -> pl.LazyFrame:
    """Rename columns and cast data types based on table definition.

    Parameters
    ----------
    df : pl.LazyFrame
        The input dataframe to process
    table_definition : Dict
        Dictionary containing column definitions with 'label', 'default', and 'type' keys

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
        data.select("pyIssue").unique().collect().get_column("pyIssue").to_list()
    )
    issue_index = 0
    if selected_issue and selected_issue in available_issues:
        issue_index = available_issues.index(selected_issue)

    # Use the selected issue (or first one if none selected)
    current_issue = (
        selected_issue if selected_issue in available_issues else available_issues[0]
    )

    # Step 2: Get groups for current issue
    filtered_by_issue = data.filter(pl.col("pyIssue") == current_issue)
    available_groups = (
        filtered_by_issue.select("pyGroup")
        .unique()
        .collect()
        .get_column("pyGroup")
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
            pl.col("pyGroup") == current_group
        )

    available_actions = (
        filtered_by_issue_group.select("pyName")
        .unique()
        .collect()
        .get_column("pyName")
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

    This utility function determines the appropriate scope level (Issue, Group, or Action)
    based on hierarchical user selections and returns configuration needed for both
    lever condition generation and plotting.

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

    Notes
    -----
    The function follows hierarchical logic:
    - If action != "All": Action-level scope
    - Elif group != "All": Group-level scope
    - Else: Issue-level scope

    Examples
    --------
    Action-level selection:
    >>> config = get_scope_config("Service", "Cards", "SpecificAction")
    >>> config["level"]  # "Action"
    >>> config["lever_condition"]  # pl.col("pyName") == "SpecificAction"

    Group-level selection:
    >>> config = get_scope_config("Service", "Cards", "All")
    >>> config["level"]  # "Group"
    >>> config["lever_condition"]  # (pl.col("pyIssue") == "Service") & (pl.col("pyGroup") == "Cards")
    """
    if selected_action != "All":
        return {
            "level": "Action",
            "lever_condition": pl.col("pyName") == selected_action,
            "group_cols": ["pyIssue", "pyGroup", "pyName"],
            "x_col": "pyName",
            "selected_value": selected_action,
            "plot_title_prefix": "Win Count by Action",
        }
    elif selected_group != "All":
        return {
            "level": "Group",
            "lever_condition": (pl.col("pyIssue") == selected_issue)
            & (pl.col("pyGroup") == selected_group),
            "group_cols": ["pyIssue", "pyGroup"],
            "x_col": "pyGroup",
            "selected_value": selected_group,
            "plot_title_prefix": "Win Count by Group",
        }
    else:
        return {
            "level": "Issue",
            "lever_condition": pl.col("pyIssue") == selected_issue,
            "group_cols": ["pyIssue"],
            "x_col": "pyIssue",
            "selected_value": selected_issue,
            "plot_title_prefix": "Win Count by Issue",
        }
