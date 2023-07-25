# -*- coding: utf-8 -*-
"""
cdhtools: Data Science add-ons for Pega.

Various utilities to access and manipulate data from Pega for purposes
of data analysis, reporting and monitoring.
"""

from typing import List, Union
import polars as pl
import re
import numpy as np
import datetime
from .types import any_frame
from .errors import NotEagerError
from .table_definitions import PegaDefaultTables

import pytz


def defaultPredictorCategorization(
    x: Union[str, pl.Expr] = pl.col("PredictorName")
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
        pl.when(x.str.split(".").list.lengths() > 1)
        .then(x.str.split(".").list.get(0))
        .otherwise(pl.lit("Primary"))
    ).alias("PredictorCategory")


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

    Parameters
    ----------
    df: Union[pl.DataFrame, pl.LazyFrame]
        The dataframe to extract the keys from
    """
    if import_strategy != "eager":
        raise NotEagerError("Extracting keys")

    def safeName():
        return (
            pl.when(~pl.col(col).cast(pl.Utf8).str.starts_with("{"))
            .then(pl.concat_str([pl.lit('{"pyName":"'), pl.col(col), pl.lit('"}')]))
            .otherwise(pl.col(col))
        ).alias("tempName")

    series = (
        df.select(
            safeName().str.json_extract(),
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
    def getMapping(columns, reverse=False):
        if not reverse:
            return dict(zip(columns, _capitalize(columns)))
        else:
            return dict(zip(_capitalize(columns), columns))

    named = getMapping(df.columns)
    typed = getMapping(
        [col for col in dir(definition) if not col.startswith("__")], reverse=True
    )
    types = []
    for col, renamedCol in named.items():
        try:
            new_type = getattr(definition, typed[renamedCol])
            if df.schema[col].base_type() != new_type:
                if new_type == pl.Datetime and df.schema[col] != pl.Date:
                    types.append(parsePegaDateTimeFormats(col, **timestamp_opts))
                else:
                    types.append(pl.col(col).cast(new_type))
        except:
            if verbose:
                print(f"Column {col} not in default table schema, can't set type.")
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
    cols = _capitalize(df.columns)
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

    df = pl.DataFrame({"truth": groundtruth, "probs": probs})
    binned = df.groupby(by="probs").agg(
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
      
    Area = (np.diff(FPR, prepend=0)) * (TPR + np.insert(np.roll(TPR, 1)[1:], 0,0)) / 2  
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
    binned = df.groupby(by="probs").agg(
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
    return df.rename(
        dict(
            zip(
                df.columns,
                _capitalize(df.columns),
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


def weighed_average_polars(
    vals: Union[str, pl.Expr], weights: Union[str, pl.Expr]
) -> pl.Expr:
    if isinstance(vals, str):
        vals = pl.col(vals)
    if isinstance(weights, str):
        weights = pl.col(weights)
    return ((vals * weights).sum()) / weights.sum()


def weighed_performance_polars() -> pl.Expr:
    """Polars function to return a weighted performance"""
    return weighed_average_polars("Performance", "ResponseCount")


def zRatio(
    posCol: pl.Expr = pl.col("BinPositives"), negCol: pl.Expr = pl.col("BinNegatives")
) -> pl.Expr:
    """Calculates the Z-Ratio for predictor bins.

    The Z-ratio is a measure of how the propensity in a bin differs from the average,
    but takes into account the size of the bin and thus is statistically more relevant.
    It represents the number of standard deviations from the avreage,
    so centers around 0. The wider the spread, the better the predictor is.

    To recreate the OOTB ZRatios from the datamart, use in a groupby.
    See `examples`.

    Parameters
    ----------
    posCol: pl.Expr
        The (Polars) column of the bin positives
    negCol: pl.Expr
        The (Polars) column of the bin positives

    Examples
    --------
    >>> df.groupby(['ModelID', 'PredictorName']).agg([zRatio()]).explode()
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


def featureImportance(over=["PredictorName", "ModelID"]):
    varImp = weighed_average_polars(
        LogOdds(
            pl.col("BinPositives"), pl.col("BinResponseCount") - pl.col("BinPositives")
        ),
        "BinResponseCount",
    ).alias("FeatureImportance")
    if over is not None:
        varImp = varImp.over(over)
    return varImp


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
        colors.append(trace.legendgroup)
    colors.sort()
    indexed_colors = {k: v for v, k in enumerate(colors)}
    for trace in fig.data:
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
    from pdstools import __reports__
    import glob
    import urllib

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
