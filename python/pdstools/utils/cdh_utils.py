import datetime
import io
import logging
import re
import tempfile
import warnings
import zipfile
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import polars as pl

from .types import QUERY

F = TypeVar("F", pl.DataFrame, pl.LazyFrame)
if TYPE_CHECKING:  # pragma: no cover
    try:
        import plotly.express as px

        Figure = Union["px.Figure", Any]
    except ImportError:
        Figure = Union[Any]


def _apply_query(df: F, query: Optional[QUERY] = None) -> F:
    if query is None:
        return df

    if isinstance(query, pl.Expr):
        col_names = set(query.meta.root_names())
        query = [query]
    elif isinstance(query, (list, tuple)):
        if not query:
            return df
        if not all(isinstance(expr, pl.Expr) for expr in query):
            raise ValueError(
                "If query is a list or tuple, all items need to be Expressions."
            )
        col_names = {
            root_name for expr in query for root_name in expr.meta.root_names()
        }
    elif isinstance(query, dict):
        if not query:  # Handle empty dict
            return df
        col_names = set(query.keys())
        query = [pl.col(k).is_in(v) for k, v in query.items()]
    else:
        raise ValueError(f"Unsupported query type: {type(query)}")

    # Check if any column names were extracted
    if not col_names:
        raise ValueError("No valid column names found in the query.")

    # Check if all queried columns exist in the DataFrame
    df_columns = set(df.collect_schema().names())
    col_diff = col_names - df_columns
    if col_diff:
        raise ValueError(f"Columns not found: {col_diff}")
    filtered_df = df.filter(query)
    if filtered_df.lazy().select(pl.first().len()).collect().item() == 0:
        raise ValueError("The given query resulted in no more remaining data.")
    return filtered_df


def _combine_queries(existing_query: QUERY, new_query: pl.Expr) -> QUERY:
    if isinstance(existing_query, pl.Expr):
        return existing_query & new_query
    elif isinstance(existing_query, List):
        return existing_query + [new_query]
    elif isinstance(existing_query, Dict):
        # Convert the dictionary to a list of expressions
        existing_exprs = [pl.col(k).is_in(v) for k, v in existing_query.items()]
        return existing_exprs + [new_query]
    else:
        raise ValueError("Unsupported query type")


def default_predictor_categorization(
    x: Union[str, pl.Expr] = pl.col("PredictorName"),
) -> pl.Expr:
    """Function to determine the 'category' of a predictor.

    It is possible to supply a custom function.
    This function can accept an optional column as input
    And as output should be a Polars expression.
    The most straight-forward way to implement this is with
    pl.when().then().otherwise(), which you can chain.

    By default, this function returns "Primary" whenever
    there is no '.' anywhere in the name string,
    otherwise returns the first string before the first period

    Parameters
    ----------
    x: Union[str, pl.Expr], default = pl.col('PredictorName')
        The column to parse

    """
    if isinstance(x, str):
        x = pl.col(x)
    x = x.cast(pl.Utf8) if not isinstance(x, pl.Utf8) else x
    return (
        pl.when(x.str.split(".").list.len() > 1)
        .then(x.str.split(".").list.get(0))
        .otherwise(pl.lit("Primary"))
    ).alias("PredictorCategory")


def _extract_keys(
    df: F,
    key: str = "Name",
    capitalize: bool = True,
) -> F:
    """Extracts keys out of the pyName column

    This is not a lazy operation as we don't know the possible keys
    in advance. For that reason, we select only the key column,
    extract the keys from that, and then collect the resulting dataframe.
    This dataframe is then joined back to the original dataframe.

    This is relatively efficient, but we still do need the whole
    pyName column in memory to do this, so it won't work completely
    lazily from e.g. s3. That's why it only works with eager mode.

    The data in column for which the JSON is extract is normalized a
    little by taking out non-space, non-printable characters. Not just
    ASCII of course. This may be relatively expensive.

    JSON extraction only happens on the unique values so saves a lot
    of time with multiple snapshots of the same models, it also only
    processes rows for which the key column appears to be valid JSON.
    It will break when you "trick" it with malformed JSON.

    Column values for columns that are also encoded in the key column
    will be overwritten with values from the key column, but only for
    rows that are JSON. In previous versions all values were overwritten
    resulting in many nulls.

    Parameters
    ----------
    df: Union[pl.DataFrame, pl.LazyFrame]
        The dataframe to extract the keys from
    key: str
        The column with embedded JSON
    capitalize: bool
        If True (default) normalizes the names of the embedded columns
        otherwise keeps the names as-is.
    """
    # Checking for the 'column is None/Null' case
    if df.collect_schema()[key] != pl.Utf8:
        return df

    # Checking for the 'empty df' of 'not containing JSON' case
    if (
        len(
            df.lazy()
            .select(key)
            .filter(pl.col(key).str.starts_with("{"))
            .head(1)
            .collect()
        )
        == 0
    ):
        return df

    keys_decoded = (
        df.filter(pl.col(key).str.starts_with("{"))
        .select(
            pl.col(key).alias("__original").unique(maintain_order=True),
        )
        .select(
            pl.col("__original"),
            pl.col("__original")
            .cast(pl.Utf8)
            .alias("__keys")
            .str.json_decode(infer_schema_length=None),
            # safe_name("__original").str.json_decode(infer_schema_length=None),
        )
        .unnest("__keys")
        .lazy()
        .collect()
    )
    if capitalize:
        keys_decoded = _polars_capitalize(keys_decoded)

    overlap = set(df.collect_schema().names()).intersection(
        keys_decoded.collect_schema().names()
    )
    return (
        df.join(
            keys_decoded.lazy() if isinstance(df, pl.LazyFrame) else keys_decoded,
            left_on=key,
            right_on="__original",
            coalesce=True,
            suffix="_decoded",
            how="left",
        )
        # Overwrite values from columns that also appear in the decoded keys
        .with_columns(
            [
                pl.when(pl.col(f"{c}_decoded").is_not_null())
                .then(pl.col(f"{c}_decoded"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in overlap
            ]
        )
        .drop([f"{c}_decoded" for c in overlap])
    )


def parse_pega_date_time_formats(
    timestamp_col="SnapshotTime",
    timestamp_fmt: Optional[str] = None,
):
    """Parses Pega DateTime formats.

    Supports commonly used formats:

    - "%Y-%m-%d %H:%M:%S"
    - "%Y%m%dT%H%M%S.%f %Z"
    - "%d-%b-%y"

    Removes timezones, and rounds to seconds, with a 'ns' time unit.

    In the implementation, the third expression uses timestamp_fmt or %Y.
    This is a bit of a hack, because if we pass None, it tries to infer automatically.
    Inferring raises when it can't find an appropriate format, so that's not good.

    Parameters
    ----------
    timestampCol: str, default = 'SnapshotTime'
        The column to parse
    timestamp_fmt: str, default = None
        An optional format to use rather than the default formats
    """

    return (
        pl.coalesce(
            pl.col(timestamp_col).str.to_datetime(
                "%Y-%m-%d %H:%M:%S", strict=False, ambiguous="null"
            ),
            pl.col(timestamp_col).str.to_datetime(
                "%Y%m%dT%H%M%S.%3f %Z", strict=False, ambiguous="null"
            ),
            pl.col(timestamp_col).str.to_datetime(
                "%d-%b-%y", strict=False, ambiguous="null"
            ),
            pl.col(timestamp_col).str.to_datetime(
                "%d%b%Y:%H:%M:%S", strict=False, ambiguous="null"
            ),
            pl.col(timestamp_col).str.to_datetime(
                timestamp_fmt or "%Y", strict=False, ambiguous="null"
            ),
        )
        .dt.replace_time_zone(None)
        .dt.cast_time_unit("ns")
    )


def safe_range_auc(auc: float) -> float:
    """Internal helper to keep auc a safe number between 0.5 and 1.0 always.

    Parameters
    ----------
    auc : float
        The AUC (Area Under the Curve) score

    Returns
    -------
    float
        'Safe' AUC score, between 0.5 and 1.0
    """
    import numpy as np

    if np.isnan(auc):
        return 0.5
    else:
        return 0.5 + np.abs(0.5 - auc)


def auc_from_probs(
    groundtruth: List[int], probs: List[float]
) -> List[float]:  # pragma: no cover
    """Calculates AUC from an array of truth values and predictions.
    Calculates the area under the ROC curve from an array of truth values and
    predictions, making sure to always return a value between 0.5 and 1.0 and
    returns 0.5 when there is just one groundtruth label.

    Parameters
    ----------
    groundtruth : List[int]
        The 'true' values, Positive values must be represented as
        True or 1. Negative values must be represented as False or 0.
    probs : List[float]
        The predictions, as a numeric vector of the same length as groundtruth

    Returns : List[float]
        The AUC as a value between 0.5 and 1.

    Examples:
        >>> auc_from_probs( [1,1,0], [0.6,0.2,0.2])
    """
    import numpy as np

    nlabels = len(np.unique(groundtruth))
    if nlabels < 2:
        return 0.5
    if nlabels > 2:
        raise ValueError("'Groundtruth' has more than two levels.")

    df = pl.DataFrame({"truth": groundtruth, "probs": probs}, strict=False)
    binned = df.group_by(probs="probs").agg(
        [
            (pl.col("truth") == 1).sum().alias("pos"),
            (pl.col("truth") == 0).sum().alias("neg"),
        ]
    )

    return auc_from_bincounts(
        binned.get_column("pos"), binned.get_column("neg"), binned.get_column("probs")
    )


def auc_from_bincounts(
    pos: List[int], neg: List[int], probs: List[float] = None
) -> float:
    """Calculates AUC from counts of positives and negatives directly
    This is an efficient calculation of the area under the ROC curve directly from an array of positives
    and negatives. It makes sure to always return a value between 0.5 and 1.0
    and will return 0.5 when there is just one groundtruth label.

    Parameters
    ----------
    pos : List[int]
        Vector with counts of the positive responses
    neg: List[int]
        Vector with counts of the negative responses
    probs: List[float]
        Optional list with probabilities which will be used to set the order of the bins. If missing defaults to pos/(pos+neg).

    Returns
    -------
    float
        The AUC as a value between 0.5 and 1.

    Examples:
        >>> auc_from_bincounts([3,1,0], [2,0,1])
    """
    import numpy as np

    pos = np.asarray(pos)
    neg = np.asarray(neg)
    if probs is None:
        probs = pos / (pos + neg)

    binorder = np.argsort(probs)[::-1]
    FPR = np.cumsum(neg[binorder]) / np.sum(neg)
    TPR = np.cumsum(pos[binorder]) / np.sum(pos)

    area = (np.diff(FPR, prepend=0)) * (TPR + np.insert(np.roll(TPR, 1)[1:], 0, 0)) / 2
    return safe_range_auc(np.sum(area))


def aucpr_from_probs(
    groundtruth: List[int], probs: List[float]
) -> List[float]:  # pragma: no cover
    """Calculates PR AUC (precision-recall) from an array of truth values and predictions.
    Calculates the area under the PR curve from an array of truth values and
    predictions. Returns 0.0 when there is just one groundtruth label.

    Parameters
    ----------
    groundtruth : List[int]
        The 'true' values, Positive values must be represented as
        True or 1. Negative values must be represented as False or 0.
    probs : List[float]
        The predictions, as a numeric vector of the same length as groundtruth

    Returns : List[float]
        The AUC as a value between 0.5 and 1.

    Examples:
        >>> auc_from_probs( [1,1,0], [0.6,0.2,0.2])
    """
    import numpy as np

    nlabels = len(np.unique(groundtruth))
    if nlabels < 2:
        return 0.0
    if nlabels > 2:
        raise ValueError("'Groundtruth' has more than two levels.")

    df = pl.DataFrame({"truth": groundtruth, "probs": probs})
    binned = df.group_by(probs="probs").agg(
        [
            (pl.col("truth") == 1).sum().alias("pos"),
            (pl.col("truth") == 0).sum().alias("neg"),
        ]
    )

    return aucpr_from_bincounts(
        binned.get_column("pos"), binned.get_column("neg"), binned.get_column("probs")
    )


def aucpr_from_bincounts(
    pos: List[int], neg: List[int], probs: List[float] = None
) -> float:
    """Calculates PR AUC (precision-recall) from counts of positives and negatives directly.
    This is an efficient calculation of the area under the PR curve directly from an
    array of positives and negatives. Returns 0.0 when there is just one
    groundtruth label.

    Parameters
    ----------
    pos : List[int]
        Vector with counts of the positive responses
    neg: List[int]
        Vector with counts of the negative responses
    probs: List[float]
        Optional list with probabilities which will be used to set the order of the bins. If missing defaults to pos/(pos+neg).

    Returns
    -------
    float
        The PR AUC as a value between 0.0 and 1.

    Examples:
        >>> aucpr_from_bincounts([3,1,0], [2,0,1])
    """
    import numpy as np

    pos = np.asarray(pos)
    neg = np.asarray(neg)
    if probs is None:
        o = np.argsort(-(pos / (pos + neg)))
    else:
        o = np.argsort(-np.asarray(probs))
    recall = np.cumsum(pos[o]) / np.sum(pos)
    precision = np.cumsum(pos[o]) / np.cumsum(pos[o] + neg[o])
    prevrecall = np.insert(recall[0 : (len(recall) - 1)], 0, 0)
    prevprecision = np.insert(precision[0 : (len(precision) - 1)], 0, 0)
    area = (recall - prevrecall) * (precision + prevprecision) / 2
    return np.sum(area[1:])


def auc_to_gini(auc: float) -> float:
    """
    Convert AUC performance metric to GINI

    Parameters
    ----------
    auc: float
        The AUC (number between 0.5 and 1)

    Returns
    -------
    float
        GINI metric, a number between 0 and 1

    Examples:
        >>> auc2GINI(0.8232)
    """
    return 2 * safe_range_auc(auc) - 1


def _capitalize(fields: Union[str, Iterable[str]]) -> List[str]:
    """Applies automatic capitalization, aligned with the R couterpart.

    Parameters
    ----------
    fields : list
        A list of names

    Returns
    -------
    fields : list
        The input list, but each value properly capitalized
    """
    capitalize_endwords = [
        "ID",
        "Key",
        "Name",
        "Treatment",
        "Count",
        "Category",
        "Class",
        "Time",
        "DateTime",
        "UpdateTime",
        "ToClass",
        "Version",
        "Predictor",
        "Predictors",
        "Rate",
        "Ratio",
        "Negatives",
        "Positives",
        "Threshold",
        "Error",
        "Importance",
        "Type",
        "Percentage",
        "Index",
        "Symbol",
        "LowerBound",
        "UpperBound",
        "Bins",
        "GroupIndex",
        "ResponseCount",
        "NegativesPercentage",
        "PositivesPercentage",
        "BinPositives",
        "BinNegatives",
        "BinResponseCount",
        "BinSymbol",
        "ResponseCountPercentage",
        "ConfigurationName",
        "Configuration",
        "SMS",
        "Relevant",
        "Proposition",
        "Active",
        "Description",
        "Reference",
        "Date",
        "Performance",
        "Identifier",
        "Component",
        "Prediction",
        "Outcome",
        "Hash",
        "URL",
        "Cap",
        "Template",
        "Issue",
        "Group",
        "Control",
        "Evidence",
        "Propensity",
        "Paid",
        "Subject",
        "Email",
        "Web",
        "Context",
        "Limit",
        "Stage",
        "Omni",
        "Execution",
        "Enabled",
        "Message",
        "Offline",
        "Update",
        "Strategy",
        "ModelTechnique",
    ]
    if not isinstance(fields, list):
        fields = [fields]
    fields = [re.sub("^p(x|y|z)", "", field.lower()) for field in fields]
    fields = list(
        map(lambda x: x.replace("configurationname", "configuration"), fields)
    )
    for word in capitalize_endwords:
        fields = [re.sub(word, word, field, flags=re.I) for field in fields]
        fields = [field[:1].upper() + field[1:] for field in fields]
    return fields


def _polars_capitalize(df: F) -> F:
    cols = df.collect_schema().names()
    renamed_cols = _capitalize(cols)

    def deduplicate(columns: List[str]):
        seen: Dict[str, int] = {}
        new_columns: List[str] = []
        for column in columns:
            if column not in seen:
                seen[column] = 1
            else:
                seen[column] += 1
            if seen[column] == 1:
                new_columns.append(column)
            elif (count := seen[column]) > 1:
                new_columns.append(column + f"_{count}")
            else:
                raise ValueError(f"While deduplicating:{column}")
        return new_columns

    if len(renamed_cols) != len(set(renamed_cols)):
        renamed_cols = deduplicate(renamed_cols)

    return df.rename(
        dict(
            zip(
                cols,
                renamed_cols,
            )
        )
    )


def from_prpc_date_time(
    x: str, return_string: bool = False
) -> Union[datetime.datetime, str]:
    """Convert from a Pega date-time string.

    Parameters
    ----------
    x: str
        String of Pega date-time
    return_string: bool, default=False
        If True it will return the date in string format. If
        False it will return in datetime type

    Returns
    -------
    Union[datetime.datetime, str]
        The converted date in datetime format or string.

    Examples:
        >>> fromPRPCDateTime("20180316T134127.847 GMT")
        >>> fromPRPCDateTime("20180316T134127.847 GMT", True)
        >>> fromPRPCDateTime("20180316T184127.846")
        >>> fromPRPCDateTime("20180316T184127.846", True)
    """
    import pytz

    timezonesplits = x.split(" ")

    if len(timezonesplits) > 1:
        x = timezonesplits[0]

    if "." in x:
        date_no_frac, frac_sec = x.split(".")
        # TODO: obtain only 3 decimals
        if len(frac_sec) > 3:
            frac_sec = frac_sec[:3]
        elif len(frac_sec) < 3:
            frac_sec = "{:<03d}".format(int(frac_sec))
    else:
        date_no_frac = x

    dt = datetime.datetime.strptime(date_no_frac, "%Y%m%dT%H%M%S")

    if len(timezonesplits) > 1:
        dt = dt.replace(tzinfo=pytz.timezone(timezonesplits[1]))

    if "." in x:
        dt = dt.replace(microsecond=int(frac_sec))

    if return_string:
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        return dt


# TODO: Polars doesn't like time zones like GMT+0200


def to_prpc_date_time(dt: datetime.datetime) -> str:
    """Convert to a Pega date-time string

    Parameters
    ----------
    x: datetime.datetime
        A datetime object

    Returns
    -------
    str
        A string representation in the format used by Pega

    Examples:
        >>> toPRPCDateTime(datetime.datetime.now())
    """
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.strftime("%Y%m%dT%H%M%S.%f")[:-3] + dt.strftime(" GMT%z")


def weighted_average_polars(
    vals: Union[str, pl.Expr], weights: Union[str, pl.Expr]
) -> pl.Expr:
    if isinstance(vals, str):
        vals = pl.col(vals)
    if isinstance(weights, str):
        weights = pl.col(weights)

    return (
        (vals * weights)
        .filter(vals.is_not_nan() & vals.is_infinite().not_() & weights.is_not_null())
        .sum()
    ) / weights.filter(
        vals.is_not_nan() & vals.is_infinite().not_() & weights.is_not_null()
    ).sum()


def weighted_performance_polars() -> pl.Expr:
    """Polars function to return a weighted performance"""
    return weighted_average_polars("Performance", "ResponseCount").fill_nan(0.5)


def overlap_lists_polars(col: pl.Series, row_validity: pl.Series) -> List[float]:
    """Calculate the overlap of each of the elements (must be a list) with all the others"""
    nrows = col.len()
    average_overlap = []
    for i in range(nrows):
        set_i = set(col[i].to_list())
        overlap_w_other_channels = []
        for j in range(nrows):
            if not row_validity[i] or not row_validity[j]:
                continue
            if i != j:
                set_j = set(col[j].to_list())
                intersection = set_i & set_j
                overlap_w_other_channels += [len(intersection)]
        if len(overlap_w_other_channels) > 0:
            average_overlap += [
                sum(overlap_w_other_channels)
                / len(overlap_w_other_channels)
                / len(set_i)
            ]
        else:
            average_overlap += [float("nan")]
    return average_overlap


def z_ratio(
    pos_col: pl.Expr = pl.col("BinPositives"), neg_col: pl.Expr = pl.col("BinNegatives")
) -> pl.Expr:
    """Calculates the Z-Ratio for predictor bins.

    The Z-ratio is a measure of how the propensity in a bin differs from the average,
    but takes into account the size of the bin and thus is statistically more relevant.
    It represents the number of standard deviations from the avreage,
    so centers around 0. The wider the spread, the better the predictor is.

    To recreate the OOTB ZRatios from the datamart, use in a group_by.
    See `examples`.

    Parameters
    ----------
    posCol: pl.Expr
        The (Polars) column of the bin positives
    negCol: pl.Expr
        The (Polars) column of the bin positives

    Examples
    --------
    >>> df.group_by(['ModelID', 'PredictorName']).agg([zRatio()]).explode()
    """

    def get_fracs(pos_col=pl.col("BinPositives"), neg_col=pl.col("BinNegatives")):
        return pos_col / pos_col.sum(), neg_col / neg_col.sum()

    def z_ratio_impl(
        pos_fraction_col=pl.col("posFraction"),
        neg_fraction_col=pl.col("negFraction"),
        positives_col=pl.sum("BinPositives"),
        negatives_col=pl.sum("BinNegatives"),
    ):
        return (
            (pos_fraction_col - neg_fraction_col)
            / (
                (pos_fraction_col * (1 - pos_fraction_col) / positives_col)
                + (neg_fraction_col * (1 - neg_fraction_col) / negatives_col)
            ).sqrt()
        ).alias("ZRatio")

    return z_ratio_impl(*get_fracs(pos_col, neg_col), pos_col.sum(), neg_col.sum())


def lift(
    pos_col: pl.Expr = pl.col("BinPositives"), neg_col: pl.Expr = pl.col("BinNegatives")
) -> pl.Expr:
    """Calculates the Lift for predictor bins.

    The Lift is the ratio of the propensity in a particular bin over the average
    propensity. So a value of 1 is the average, larger than 1 means higher
    propensity, smaller means lower propensity.

    Parameters
    ----------
    posCol: pl.Expr
        The (Polars) column of the bin positives
    negCol: pl.Expr
        The (Polars) column of the bin positives

    Examples
    --------
    >>> df.group_by(['ModelID', 'PredictorName']).agg([lift()]).explode()
    """

    def lift_impl(bin_pos, bin_neg, total_pos, total_neg):
        return (
            # TODO not sure how polars (mis)behaves when there are no positives at all
            # I would hope for a NaN but base python doesn't do that. Polars perhaps.
            # Stijn: It does have proper None value support, may work like you say
            bin_pos * (total_pos + total_neg) / ((bin_pos + bin_neg) * total_pos)
        ).alias("Lift")

    return lift_impl(pos_col, neg_col, pos_col.sum(), neg_col.sum())


def log_odds(
    positives=pl.col("Positives"),
    negatives=pl.col("ResponseCount") - pl.col("Positives"),
):
    N = positives.count()
    return (
        (
            ((positives + 1 / N).log() - (positives + 1).sum().log())
            - ((negatives + 1 / N).log() - (negatives + 1).sum().log())
        )
        .round(2)
        .alias("LogOdds")
    )


# TODO: reconsider this. Feature importance now stored in datamart
# perhaps we should not bother to calculate it ourselves.
def feature_importance(over=["PredictorName", "ModelID"]):
    var_imp = weighted_average_polars(
        log_odds(
            pl.col("BinPositives"), pl.col("BinResponseCount") - pl.col("BinPositives")
        ),
        "BinResponseCount",
    ).alias("FeatureImportance")
    if over is not None:
        var_imp = var_imp.over(over)
    return var_imp


def _apply_schema_types(df: F, definition, verbose=False, **timestamp_opts) -> F:
    """
    This function is used to convert the data types of columns in a DataFrame to a desired types.
    The desired types are defined in a `PegaDefaultTables` class.

    Parameters
    ----------
    df : pl.LazyFrame
        The DataFrame whose columns' data types need to be converted.
    definition : PegaDefaultTables
        A `PegaDefaultTables` object that contains the desired data types for the columns.
    verbose : bool
        If True, the function will print a message when a column is not in the default table schema.
    timestamp_opts : str
        Additional arguments for timestamp parsing.

    Returns
    -------
    List
        A list with polars expressions for casting data types.
    """

    def get_mapping(columns, reverse=False):
        if not reverse:
            return dict(zip(columns, _capitalize(columns)))
        else:
            return dict(zip(_capitalize(columns), columns))

    schema = df.collect_schema()
    named = get_mapping(schema.names())
    typed = get_mapping(
        [col for col in dir(definition) if not col.startswith("__")], reverse=True
    )

    types = []
    for col, renamedCol in named.items():
        try:
            new_type = getattr(definition, typed[renamedCol])
            original_type = schema[col].base_type()
            if original_type == pl.Null:
                if verbose:
                    warnings.warn(f"Warning: {col} column is Null data type.")
            elif original_type != new_type:
                if (
                    original_type == pl.Categorical
                    and new_type in pl.selectors.numeric()
                ):
                    types.append(pl.col(col).cast(pl.Utf8).cast(new_type))
                elif new_type == pl.Datetime and original_type != pl.Date:
                    types.append(parse_pega_date_time_formats(col, **timestamp_opts))
                else:
                    types.append(pl.col(col).cast(new_type))
        except Exception:
            if verbose:  # pragma: no cover
                warnings.warn(
                    f"Column {col} not in default table schema, can't set type."
                )
    return df.with_columns(types)


def gains_table(df, value: str, index=None, by=None):
    """Calculates cumulative gains from any data frame.

    The cumulative gains are the cumulative values expressed
    as a percentage vs the size of the population, also expressed
    as a percentage.

    Parameters
    ----------
    df: pl.DataFrame
        The (Polars) dataframe with the raw values
    value: str
        The name of the field with the values (plotted on y-axis)
    index = None
        Optional name of the field for the x-axis. If not passed in
        all records are used and weighted equally.
    by = None
        Grouping field(s), can also be None

    Returns
    -------
    pl.DataFrame
        A (Polars) dataframe with cum_x and cum_y columns and optionally
        the grouping column(s). Values for cum_x and cum_y are relative
        so expressed as values 0-1.

    Examples
    --------
    >>> gains_data = gains_table(df, 'ResponseCount', by=['Channel','Direction])
    """

    sort_expr = pl.col(value) if index is None else pl.col(value) / pl.col(index)
    index_expr = (
        (pl.int_range(1, pl.len() + 1) / pl.len())
        if index is None
        else (pl.cum_sum(index) / pl.sum(index))
    )

    if by is None:
        gains_df = pl.concat(
            [
                pl.DataFrame(data={"cum_x": [0.0], "cum_y": [0.0]}).lazy(),
                df.lazy()
                .sort(sort_expr, descending=True)
                .select(
                    index_expr.cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value)).cast(pl.Float64).alias("cum_y"),
                ),
            ]
        )
    else:
        by_as_list = by if isinstance(by, list) else [by]
        sort_expr = by_as_list + [sort_expr]
        gains_df = (
            df.lazy()
            .sort(sort_expr, descending=True)
            .select(
                by_as_list
                + [
                    index_expr.over(by).cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value))
                    .over(by)
                    .cast(pl.Float64)
                    .alias("cum_y"),
                ]
            )
        )
        # Add entry for the (0,0) point
        gains_df = pl.concat(
            [gains_df.group_by(by).agg(cum_x=pl.lit(0.0), cum_y=pl.lit(0.0)), gains_df]
        ).sort(by_as_list + ["cum_x"])

    return gains_df.collect()


def lazy_sample(df: F, n_rows: int, with_replacement: bool = True) -> F:
    if with_replacement:
        return df.select(pl.all().sample(n=n_rows, with_replacement=with_replacement))

    from functools import partial

    import numpy as np

    def sample_it(s: pl.Series, n) -> pl.Series:
        s_len = s.len()
        if s_len < n:
            return pl.Series(values=[True] * s_len, dtype=pl.Boolean)
        else:
            return pl.Series(
                values=np.random.binomial(1, n / s_len, s_len),
                dtype=pl.Boolean,
            )

    func = partial(sample_it, n=n_rows)
    return (
        df.with_columns(pl.first().map_batches(func).alias("_sample"))
        .filter(pl.col("_sample"))
        .drop("_sample")
    )


# TODO: perhaps the color / plot utils should move into a separate file
def legend_color_order(fig):
    """Orders legend colors alphabetically in order to provide pega color
    consistency among different categories"""

    colorway = [
        "#001F5F",  # dark blue
        "#10A5AC",
        "#F76923",  # orange
        "#661D34",  # wine
        "#86CAC6",  # mint
        "#005154",  # forest
        "#86CAC6",  # mint
        "#5F67B9",  # violet
        "#FFC836",  # yellow
        "#E63690",  # pink
        "#AC1361",  # berry
        "#63666F",  # dark grey
        "#A7A9B4",  # medium grey
        "#D0D1DB",  # light grey
    ]
    colors = []
    for trace in fig.data:
        if trace.legendgroup is not None:
            colors.append(trace.legendgroup)
    colors.sort()

    # https://github.com/pegasystems/pega-datascientist-tools/issues/201
    if len(colors) >= len(colorway):
        return fig

    indexed_colors = {k: v for v, k in enumerate(colors)}
    for trace in fig.data:
        if trace.legendgroup is not None:
            try:
                trace.marker.color = colorway[indexed_colors[trace.legendgroup]]
                trace.line.color = colorway[indexed_colors[trace.legendgroup]]
            except AttributeError:  # pragma: no cover
                pass

    return fig


def process_files_to_bytes(
    file_paths: List[Union[str, Path]], base_file_name: Union[str, Path]
) -> Tuple[bytes, str]:
    """
    Processes a list of file paths, returning file content as bytes and a corresponding file name.
    Useful for zipping muliple model reports and the byte object is used for downloading files in
    Streamlit app.

    This function handles three scenarios:
    1. Single file: Returns the file's content as bytes and the provided base file name.
    2. Multiple files: Creates a zip file containing all files, returns the zip file's content as bytes
       and a generated zip file name.
    3. No files: Returns empty bytes and an empty string.

    Parameters
    ----------
    file_paths : List[Union[str, Path]]
        A list of file paths to process. Can be empty, contain a single path, or multiple paths.
    base_file_name : Union[str, Path]
        The base name to use for the output file. For a single file, this name is returned as is.
        For multiple files, this is used as part of the generated zip file name.

    Returns
    -------
    Tuple[bytes, str]
        A tuple containing:
        - bytes: The content of the single file or the created zip file, or empty bytes if no files.
        - str: The file name (either base_file_name or a generated zip file name), or an empty string if no files.
    """
    path_list: List[Path] = [Path(fp) for fp in file_paths]
    base_file_name = Path(base_file_name)

    if not path_list:
        return b"", ""

    if len(path_list) == 1:
        try:
            with path_list[0].open("rb") as file:
                return file.read(), base_file_name.name
        except IOError as e:
            print(f"Error reading file {path_list[0]}: {e}")
            return b"", ""

    # Multiple files
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, "w") as zipf:
        for file_path in path_list:
            try:
                zipf.write(
                    file_path,
                    file_path.name,
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            except IOError as e:
                print(f"Error adding file {file_path} to zip: {e}")

    time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    zip_file_name = f"{base_file_name.stem}_{time}.zip"
    in_memory_zip.seek(0)
    return in_memory_zip.getvalue(), zip_file_name


def get_latest_pdstools_version():
    import requests

    try:
        response = requests.get("https://pypi.org/pypi/pdstools/json")
        return response.json()["info"]["version"]
    except Exception:
        return None


def setup_logger():
    """Returns a logger and log buffer in root level"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_buffer = StringIO()
    handler = logging.StreamHandler(log_buffer)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, log_buffer


def create_working_and_temp_dir(
    name: Optional[str] = None,
    working_dir: Optional[PathLike] = None,
) -> Tuple[Path, Path]:
    """Creates a working directory for saving files and a temp_dir"""
    # Create a temporary directory in working_dir
    working_dir = Path(working_dir) if working_dir else Path.cwd()
    working_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_name = (
        tempfile.mkdtemp(prefix=f"tmp_{name}_", dir=working_dir)
        if name
        else tempfile.mkdtemp(prefix="tmp_", dir=working_dir)
    )
    return working_dir, Path(temp_dir_name)
