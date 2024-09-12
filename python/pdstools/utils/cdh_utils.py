# -*- coding: utf-8 -*-
"""
cdhtools: Data Science add-ons for Pega.

Various utilities to access and manipulate data from Pega for purposes
of data analysis, reporting and monitoring.
"""

import datetime
import io
import logging
import re
import warnings
import zipfile
import tempfile
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import polars as pl
import pytz
import requests

from .errors import NotEagerError
from .table_definitions import PegaDefaultTables
from .types import any_frame


def defaultPredictorCategorization(
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


# TODO: we could make below much more efficient. We're now making everything
# JSON then extracting. We could first detect if there is any JSON at all
# ( startswith("{").any() ). Also could do the JSON extract on just the unique pyNames
# that do start with a {. This could then be merged back with the original data.


def _extract_keys(
    df: any_frame,
    col="Name",
    capitalize=True,
    import_strategy="eager",
) -> any_frame:
    """Extracts keys out of the pyName column

    This is not a lazy operation as we don't know the possible keys
    in advance. For that reason, we select only the pyName column,
    extract the keys from that, and then collect the resulting dataframe.
    This dataframe is then joined back to the original dataframe.

    This is relatively efficient, but we still do need the whole
    pyName column in memory to do this, so it won't work completely
    lazily from e.g. s3. That's why it only works with eager mode.

    The data in column for which the JSON is extract is normalized a
    little by taking out non-space, non-printable characters. Not just
    ASCII of course. This may be relatively expensive, see notes in
    code about ideas to speed up.

    Parameters
    ----------
    df: Union[pl.DataFrame, pl.LazyFrame]
        The dataframe to extract the keys from
    """
    # Checking for the 'column is None/Null' case
    if df.collect_schema()[col] != pl.Utf8:
        return df

    if import_strategy != "eager":
        raise NotEagerError("Extracting keys")

    # Checking for the 'empty df' case
    if len(df.select(col).head(1).collect()) == 0:
        return df

    def safeName():
        return (
            pl.when(~pl.col(col).cast(pl.Utf8).str.starts_with("{"))
            .then(
                pl.concat_str(
                    [
                        pl.lit('{"pyName":"'),
                        # we need to protect the string in the extract column
                        # against very wild content like emoticons that break
                        # the json parsing
                        # see https://www.regular-expressions.info/posixbrackets.html
                        pl.col(col).str.replace_all(
                            r"[^\p{L}\p{Nl}\p{Nd}\p{P}\p{Z}]", ""
                        ),
                        pl.lit('"}'),
                    ]
                )
            )
            .otherwise(pl.col(col).cast(pl.Utf8))
        ).alias("tempName")

    series = (
        df.select(
            safeName().str.json_decode(infer_schema_length=None),
        )
        .unnest("tempName")
        .lazy()
        .collect()
    )
    if not capitalize:
        return df.with_columns(series)
    return df.with_columns(_polarsCapitalize(series))


def parsePegaDateTimeFormats(
    timestampCol="SnapshotTime",
    timestamp_fmt: str = None,
    strict_conversion: bool = True,
):
    """Parses Pega DateTime formats.

    Supports the two most commonly used formats:

    - "%Y-%m-%d %H:%M:%S"
    - "%Y%m%dT%H%M%S.%f %Z"

    If you want to parse a different timezone, then

    Removes timezones, and rounds to seconds, with a 'ns' time unit.

    Parameters
    ----------
    timestampCol: str, default = 'SnapshotTime'
        The column to parse
    timestamp_fmt: str, default = None
        An optional format to use rather than the default formats
    strict_conversion: bool, default = True
        Whether to error on incorrect parses or just return Null values
    """
    if timestamp_fmt is not None:
        return pl.col(timestampCol).str.strptime(
            pl.Datetime,
            timestamp_fmt,
            strict=strict_conversion,
        )
    else:
        return (
            pl.when((pl.col(timestampCol).str.slice(4, 1) == pl.lit("-")))
            .then(
                pl.col(timestampCol)
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                .dt.cast_time_unit("ns")
            )
            .otherwise(
                pl.col(timestampCol)
                .str.strptime(pl.Datetime, "%Y%m%dT%H%M%S.%3f %Z", strict=False)
                .dt.replace_time_zone(None)
                .dt.cast_time_unit("ns")
            )
        ).alias(timestampCol)


def getTypeMapping(df, definition, verbose=False, **timestamp_opts):
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

    def getMapping(columns, reverse=False):
        if not reverse:
            return dict(zip(columns, _capitalize(columns)))
        else:
            return dict(zip(_capitalize(columns), columns))

    named = getMapping(df.collect_schema().names())
    typed = getMapping(
        [col for col in dir(definition) if not col.startswith("__")], reverse=True
    )

    types = []
    for col, renamedCol in named.items():
        try:
            new_type = getattr(definition, typed[renamedCol])
            original_type = df.collect_schema()[col].base_type()
            if original_type == pl.Null:
                if verbose:
                    warnings.warn(f"Warning: {col} column is Null data type.")
            elif original_type != new_type:
                if original_type == pl.Categorical and new_type in pl.NUMERIC_DTYPES:
                    types.append(pl.col(col).cast(pl.Utf8).cast(new_type))
                elif new_type == pl.Datetime and original_type != pl.Date:
                    types.append(parsePegaDateTimeFormats(col, **timestamp_opts))
                else:
                    types.append(pl.col(col).cast(new_type))
        except:
            if verbose:
                warnings.warn(
                    f"Column {col} not in default table schema, can't set type."
                )
    return types


def set_types(df, table="infer", verbose=False, **timestamp_opts):
    if table == "infer":
        table = inferTableDefinition(df)

    if table == "pyValueFinder":
        definition = PegaDefaultTables.pyValueFinder()
    elif table == "ADMModelSnapshot":
        definition = PegaDefaultTables.ADMModelSnapshot()
    elif table == "ADMPredictorBinningSnapshot":
        definition = PegaDefaultTables.ADMPredictorBinningSnapshot()

    else:
        raise ValueError(table)

    mapping = getTypeMapping(df, definition, verbose=verbose, **timestamp_opts)

    if len(mapping) > 0:
        return df.with_columns(mapping)
    else:
        return df


def inferTableDefinition(df):
    cols = _capitalize(df.collect_schema().names())
    vf = ["Propensity", "Stage"]
    predictors = ["PredictorName", "ModelID", "BinSymbol"]
    models = ["ModelID", "Performance"]
    if all(value in cols for value in vf):
        return "pyValueFinder"
    elif all(value in cols for value in predictors):
        return "ADMPredictorBinningSnapshot"
    elif all(value in cols for value in models):
        return "ADMModelSnapshot"
    else:
        print("Could not find table definition.")
        return cols


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
    nlabels = len(np.unique(groundtruth))
    if nlabels < 2:
        return 0.5
    if nlabels > 2:
        raise Exception("'Groundtruth' has more than two levels.")

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
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    if probs is None:
        probs = pos / (pos + neg)

    binorder = np.argsort(probs)[::-1]
    FPR = np.cumsum(neg[binorder]) / np.sum(neg)
    TPR = np.cumsum(pos[binorder]) / np.sum(pos)

    Area = (np.diff(FPR, prepend=0)) * (TPR + np.insert(np.roll(TPR, 1)[1:], 0, 0)) / 2
    return safe_range_auc(np.sum(Area))


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
    nlabels = len(np.unique(groundtruth))
    if nlabels < 2:
        return 0.0
    if nlabels > 2:
        raise Exception("'Groundtruth' has more than two levels.")

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
    Area = (recall - prevrecall) * (precision + prevprecision) / 2
    return np.sum(Area[1:])


def auc2GINI(auc: float) -> float:
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


def _capitalize(fields: list) -> list:
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
    capitalizeEndWords = [
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
    ]
    if not isinstance(fields, list):
        fields = [fields]
    fields = [re.sub("^p(x|y|z)", "", field.lower()) for field in fields]
    fields = list(
        map(lambda x: x.replace("configurationname", "configuration"), fields)
    )
    for word in capitalizeEndWords:
        fields = [re.sub(word, word, field, flags=re.I) for field in fields]
        fields = [field[:1].upper() + field[1:] for field in fields]
    return fields


def _polarsCapitalize(df: pl.LazyFrame):
    df_cols = df.collect_schema().names()
    return df.rename(
        dict(
            zip(
                df_cols,
                _capitalize(df_cols),
            )
        )
    )


def fromPRPCDateTime(
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


def toPRPCDateTime(dt: datetime.datetime) -> str:
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
    return weighted_average_polars("Performance", "ResponseCount")


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


def zRatio(
    posCol: pl.Expr = pl.col("BinPositives"), negCol: pl.Expr = pl.col("BinNegatives")
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

    def getFracs(posCol=pl.col("BinPositives"), negCol=pl.col("BinNegatives")):
        return posCol / posCol.sum(), negCol / negCol.sum()

    def zRatioimpl(
        posFractionCol=pl.col("posFraction"),
        negFractionCol=pl.col("negFraction"),
        PositivesCol=pl.sum("BinPositives"),
        NegativesCol=pl.sum("BinNegatives"),
    ):
        return (
            (posFractionCol - negFractionCol)
            / (
                (posFractionCol * (1 - posFractionCol) / PositivesCol)
                + (negFractionCol * (1 - negFractionCol) / NegativesCol)
            ).sqrt()
        ).alias("ZRatio")

    return zRatioimpl(*getFracs(posCol, negCol), posCol.sum(), negCol.sum())


def lift(
    posCol: pl.Expr = pl.col("BinPositives"), negCol: pl.Expr = pl.col("BinNegatives")
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

    def liftImpl(binPos, binNeg, totalPos, totalNeg):
        return (
            # TODO not sure how polars (mis)behaves when there are no positives at all
            # I would hope for a NaN but base python doesn't do that. Polars perhaps.
            # Stijn: It does have proper None value support, may work like you say
            binPos
            * (totalPos + totalNeg)
            / ((binPos + binNeg) * totalPos)
        ).alias("Lift")

    return liftImpl(posCol, negCol, posCol.sum(), negCol.sum())


def LogOdds(
    Positives=pl.col("Positives"),
    Negatives=pl.col("ResponseCount") - pl.col("Positives"),
):
    N = Positives.count()
    return (
        (
            ((Positives + 1 / N).log() - (Positives + 1).sum().log())
            - ((Negatives + 1 / N).log() - (Negatives + 1).sum().log())
        )
        .round(2)
        .alias("LogOdds")
    )


# TODO: reconsider this. Feature importance now stored in datamart
# perhaps we should not bother to calculate it ourselves.
def featureImportance(over=["PredictorName", "ModelID"]):
    varImp = weighted_average_polars(
        LogOdds(
            pl.col("BinPositives"), pl.col("BinResponseCount") - pl.col("BinPositives")
        ),
        "BinResponseCount",
    ).alias("FeatureImportance")
    if over is not None:
        varImp = varImp.over(over)
    return varImp


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

    sortExpr = pl.col(value) if index is None else pl.col(value) / pl.col(index)
    indexExpr = (
        (pl.int_range(1, pl.count() + 1) / pl.count())
        if index is None
        else (pl.cum_sum(index) / pl.sum(index))
    )

    if by is None:
        gains_df = pl.concat(
            [
                pl.DataFrame(data={"cum_x": [0.0], "cum_y": [0.0]}).lazy(),
                df.lazy()
                .sort(sortExpr, descending=True)
                .select(
                    indexExpr.cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value)).cast(pl.Float64).alias("cum_y"),
                ),
            ]
        )
    else:
        by_as_list = by if isinstance(by, list) else [by]
        sortExpr = by_as_list + [sortExpr]
        gains_df = (
            df.lazy()
            .sort(sortExpr, descending=True)
            .select(
                by_as_list
                + [
                    indexExpr.over(by).cast(pl.Float64).alias("cum_x"),
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


def sync_reports(checkOnly: bool = False, autoUpdate: bool = False):
    """Compares the report files in your local directory to the repo

    If any of the files are different from the ones in GitHub,
    will prompt you to update them.

    Parameters
    ----------
    checkOnly : bool, default = False
        If True, only checks, does not prompt to update
    autoUpdate : bool, default = False
        If True, doensn't prompt for update and goes ahead
    """
    import glob
    import urllib

    from pdstools import __reports__

    files = [f for f in glob.glob(str(__reports__ / "*.qmd"))]
    latest = {}
    replacement = {}
    for file in files:
        name = file.rsplit("/")[-1]
        path = f"http://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/python/pdstools/reports/{name}"
        fileFromUrl = urllib.request.urlopen(path).read()
        latest[file] = (
            int.from_bytes(fileFromUrl) ^ int.from_bytes(open(file).read().encode())
            == 0
        )
        if not latest[file]:
            replacement[file] = fileFromUrl
    if False in latest.values():
        if not checkOnly and (
            autoUpdate
            or input("One or more files out of sync. Enter 'y' to update them:")
        ):
            for filename, file in replacement.items():
                with open(filename, "w") as f:
                    f.write(file.decode())
            return True
        else:
            return False
    else:
        return True


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
    file_paths = [Path(fp) for fp in file_paths]
    base_file_name = Path(base_file_name)

    if not file_paths:
        return b"", ""

    if len(file_paths) == 1:
        try:
            with file_paths[0].open("rb") as file:
                return file.read(), base_file_name.name
        except IOError as e:
            print(f"Error reading file {file_paths[0]}: {e}")
            return b"", ""

    # Multiple files
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, "w") as zipf:
        for file_path in file_paths:
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
    try:
        response = requests.get("https://pypi.org/pypi/pdstools/json")
        return response.json()["info"]["version"]
    except:
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
    working_dir: Union[str, Path, None] = None,
):
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
