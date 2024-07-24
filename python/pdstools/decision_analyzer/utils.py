import datetime
import subprocess
from bisect import bisect_left
from typing import Dict, Iterable, List, Literal, Optional, Set, Type, Union

import polars as pl
from ..utils.cdh_utils import parsePegaDateTimeFormats

from .table_definition import DecisionAnalyzer, ExplainabilityExtract, TableConfig

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
        col_diff = set(item.meta.root_names()) - set(df.columns)
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

    required_cols = groupby_cols + [
        "pyName",
        "FinalPropensity",  # TODO generalize this stuff
        "pxEngagementStage",
        "Priority",
    ]
    # TODO below is a pattern (to check required columns) we probably need all over the place - but maybe at the level of the streamlit pages
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")

    propensity_classifying_expr = [
        pl.col("pyName")
        .where(
            (pl.col("FinalPropensity") == 0.5)
            & (pl.col("pxEngagementStage") != "Final")
        )
        .count()
        .alias("new_models"),
        pl.col("pyName")
        .where(
            (pl.col("FinalPropensity") < propensityTH)
            & (pl.col("FinalPropensity") != 0.5)
        )
        .count()
        .alias("poor_propensity_offers"),
        pl.col("pyName")
        .where((pl.col("Priority") < priorityTH) & (pl.col("FinalPropensity") != 0.5))
        .count()
        .alias("poor_priority_offers"),
        pl.col("pyName")
        .where(
            (pl.col("FinalPropensity") >= propensityTH)
            & (pl.col("FinalPropensity") != 0.5)
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


# TODO refactor this into the DecisionData class but figure out what we really want
def find_lever_value(
    decision_data,
    action,
    target_win_percentage,
    win_rank,
    low=0,
    high=100,
    precision=0.01,
    ranking_stages=["Arbitration"],
):
    """Binary search algorithm to find lever given a desired win percentage"""

    def _calculate_action_win_percentage(decision_data, action, lever, win_rank):
        ranked_df = decision_data.reRank(
            overrides=[
                (
                    pl.when(pl.col("pyName") == action)
                    .then(pl.lit(lever))
                    .otherwise(pl.col("Weight"))
                ).alias("Weight")
            ]
        )
        action_win_percentage = (
            ranked_df.filter(pl.col("rank_PVCL") <= win_rank)
            .filter(pl.col("pxEngagementStage").is_in(ranking_stages))
            .group_by("rank_PVCL")
            .agg(
                action_rank1=pl.col("pxInteractionID")
                .where(pl.col("pyName") == action)
                .count(),
                action_not_rank1=pl.col("pxInteractionID")
                .where(pl.col("pyName") != action)
                .count(),
            )
            .with_columns(
                percentage=pl.col("action_rank1")
                / (pl.col("action_rank1") + pl.col("action_not_rank1"))
            )
            .select("percentage")
            .collect()
            .row(0)[0]
            * 100
        )
        return action_win_percentage

    beginning_high = high
    beginning_low = low

    while high - low > precision:
        mid = (low + high) / 2

        current_win_percentage = _calculate_action_win_percentage(
            decision_data=decision_data,
            action=action,
            lever=mid,
            win_rank=win_rank,
        )

        if current_win_percentage < target_win_percentage:
            low = mid
        else:
            high = mid

    if mid >= beginning_high - precision:
        return ValueError(
            f"""You need a higher lever than {high} to reach your desired win rate of {target_win_percentage}%.
            You can increase the search range from siderbar.
            """
        )
    elif mid <= (beginning_low + precision):
        return ValueError(
            f"You need a lower lever than {low} to reach your desired win rate of {target_win_percentage}%"
        )

    return (low + high) / 2



def determine_extract_type(raw_data):
    return (
        "decision_analyzer"
        if "pxEngagementStage" in raw_data.columns
        else "explainability_extract"
    )


## From PDSTOOLS V4 (we can use pdstools.utils.process once v4 is published)
def process(
    df: pl.LazyFrame,
    table: Literal["decision_analyzer", "explainability_extract"],
    subset: bool = True,
    include_cols: Optional[Iterable[str]] = None,
    drop_cols: Optional[Iterable[str]] = None,
    raise_on_unknown: bool = True,
) -> pl.LazyFrame:
    # df = cdh_utils._polars_capitalize(df)
    table_definition = get_table_definition(table)

    type_map = get_schema(
        df,
        table_definition=table_definition,
        include_cols=include_cols or {},
        drop_cols=drop_cols or {},
        subset=subset,
        raise_on_unknown=raise_on_unknown,
    )
    for name, _type in type_map.items():
        if df.select(name).dtypes[0] != _type:
            if _type == pl.Datetime:
                df = df.with_columns(parsePegaDateTimeFormats(name))
            else:
                df = df.with_columns(pl.col(name).cast(_type))
    return df


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
    table_definition: Dict[str, TableConfig],
    include_cols: Iterable[str],
    drop_cols: Iterable[str],
    subset: bool,
    raise_on_unknown: bool = True,
) -> Dict[str, Type[pl.DataType]]:
    type_map: Dict[str, Type[pl.DataType]] = dict()
    checked_columns: Set[str] = set()

    for column, config in table_definition.items():
        df_col = None
        checked_columns = checked_columns.union({column, config["label"]})
        if column in df.columns:
            df_col = column

        elif config["label"] in df.columns:
            df_col = config["label"]

        if df_col is not None and (  # If we've found a matching column
            subset is False  # We don't want to take a subset...
            or (
                df_col not in drop_cols  # The user does not want to drop it
                and (config["default"] or df_col in include_cols)
            )  # And it's either default or part of include cols
        ):
            # Then we make it part of the type map, which we use to filter down
            type_map[df_col] = config["type"]

    unknown_columns = [col for col in df.columns if col not in checked_columns]
    if unknown_columns and raise_on_unknown:
        raise ValueError("Unknown columns found: ", unknown_columns)

    return type_map


## Section from PDSTOOLS V4 ends
