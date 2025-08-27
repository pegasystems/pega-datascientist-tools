import datetime
import subprocess
from typing import Dict, Iterable, List, Optional, Type, Union

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
            raise pl.ColumnNotFoundError(col_diff)
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


def get_git_version_and_date():
    # Get the version tag
    version = (
        subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
        .decode()
        .strip()
        .replace("v", "")
    )

    # Get the date the tag was pushed
    date_str = (
        subprocess.check_output(["git", "log", "-1", "--format=%ai", version])
        .decode()
        .strip()
    )
    date = datetime.datetime.strptime(date_str.split()[0], "%Y-%m-%d")
    return version, date.strftime("%d %b %Y")


def determine_extract_type(raw_data):
    return (
        "decision_analyzer"
        if "pxStrategyName" in raw_data.collect_schema().names()
        else "explainability_extract"
    )


def rename_and_cast_types(
    df: pl.LazyFrame,
    table_definition: Dict,
    include_cols: Optional[Iterable[str]] = None,
) -> pl.LazyFrame:
    """Rename columns and cast data types based on table definition.

    Parameters
    ----------
    df : pl.LazyFrame
        The input dataframe to process
    table_definition : Dict
        Dictionary containing column definitions with 'label', 'default', and 'type' keys
    include_cols : Optional[Iterable[str]], optional
        Additional columns to include beyond default columns

    Returns
    -------
    pl.LazyFrame
        Processed dataframe with renamed columns and cast types
    """
    # df = cdh_utils._polars_capitalize(df)

    type_map = get_schema(
        df,
        table_definition=table_definition,
        include_cols=include_cols or {},
    )
    # cast types
    for name, _type in type_map.items():
        if df.select(name).collect_schema().dtypes()[0] != _type:
            if _type == pl.Datetime:
                df = df.with_columns(parse_pega_date_time_formats(name))
            else:
                df = df.with_columns(pl.col(name).cast(_type))
    # rename
    name_dict = {}
    for col, properties in table_definition.items():
        name_dict[col] = properties["label"]

    return df.rename(name_dict).select(list(name_dict.values()))


def get_table_definition(table: str):
    mapping = {
        "decision_analyzer": DecisionAnalyzer,
        "explainability_extract": ExplainabilityExtract,
    }
    if table not in mapping:
        raise ValueError(f"Unknown table: {table}")
    return mapping[table]


def get_schema(
    df: pl.LazyFrame,
    table_definition: Dict,
    include_cols: Iterable[str],
) -> Dict[str, Type[pl.DataType]]:
    """Build type mapping for dataframe columns based on table definition.

    Parameters
    ----------
    df : pl.LazyFrame
        The input dataframe to analyze
    table_definition : Dict
        Dictionary containing column definitions with 'label', 'default', and 'type' keys
    include_cols : Iterable[str]
        Additional columns to include beyond default columns

    Returns
    -------
    Dict[str, Type[pl.DataType]]
        Mapping of column names to their data types
    """
    type_map: Dict[str, Type[pl.DataType]] = {}
    available_columns = df.collect_schema().names()

    for defined_col, config in table_definition.items():
        # Find matching column in dataframe (try original name, then label)
        actual_col = None
        if defined_col in available_columns:
            actual_col = defined_col
        elif config["label"] in available_columns:
            actual_col = config["label"]

        # Include column if found and is default or explicitly requested
        if actual_col and (config["default"] or actual_col in include_cols):
            type_map[actual_col] = config["type"]

    return type_map


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
