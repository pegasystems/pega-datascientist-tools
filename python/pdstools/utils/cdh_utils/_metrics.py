"""Performance metrics: AUC, lift, log-odds, gains, feature importance."""

import math
from collections.abc import Sequence

import polars as pl

from ._polars import weighted_average_polars


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
    return 0.5 + np.abs(0.5 - auc)


def auc_from_probs(groundtruth: list[int], probs: list[float]) -> float:
    """Calculates AUC from an array of truth values and predictions.
    Calculates the area under the ROC curve from an array of truth values and
    predictions, making sure to always return a value between 0.5 and 1.0 and
    returns 0.5 when there is just one groundtruth label.

    Parameters
    ----------
    groundtruth : list[int]
        The 'true' values, Positive values must be represented as
        True or 1. Negative values must be represented as False or 0.
    probs : list[float]
        The predictions, as a numeric vector of the same length as groundtruth

    Returns : float
        The AUC as a value between 0.5 and 1.

    Examples
    --------
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
        ],
    )

    return auc_from_bincounts(
        binned.get_column("pos"),
        binned.get_column("neg"),
        binned.get_column("probs"),
    )


def auc_from_bincounts(
    pos: Sequence[int] | pl.Series,
    neg: Sequence[int] | pl.Series,
    probs: Sequence[float] | pl.Series | None = None,
) -> float:
    """Calculates AUC from counts of positives and negatives directly
    This is an efficient calculation of the area under the ROC curve directly from an array of positives
    and negatives. It makes sure to always return a value between 0.5 and 1.0
    and will return 0.5 when there is just one groundtruth label.

    Parameters
    ----------
    pos : list[int]
        Vector with counts of the positive responses
    neg: list[int]
        Vector with counts of the negative responses
    probs: list[float]
        Optional list with probabilities which will be used to set the order of the bins. If missing defaults to pos/(pos+neg).

    Returns
    -------
    float
        The AUC as a value between 0.5 and 1.

    Examples
    --------
        >>> auc_from_bincounts([3,1,0], [2,0,1])

    """
    import numpy as np

    pos_arr = np.asarray(pos)
    neg_arr = np.asarray(neg)

    if (np.sum(pos_arr) == 0) or (np.sum(neg_arr) == 0):
        return 0.5

    if probs is None:
        probs = pos_arr / (pos_arr + neg_arr)

    binorder = np.argsort(probs)[::-1]
    FPR = np.cumsum(neg_arr[binorder]) / np.sum(neg_arr)
    TPR = np.cumsum(pos_arr[binorder]) / np.sum(pos_arr)

    area = (np.diff(FPR, prepend=0)) * (TPR + np.insert(np.roll(TPR, 1)[1:], 0, 0)) / 2
    return safe_range_auc(np.sum(area))


def aucpr_from_probs(groundtruth: list[int], probs: list[float]) -> float:
    """Calculates PR AUC (precision-recall) from an array of truth values and predictions.
    Calculates the area under the PR curve from an array of truth values and
    predictions. Returns 0.0 when there is just one groundtruth label.

    Parameters
    ----------
    groundtruth : list[int]
        The 'true' values, Positive values must be represented as
        True or 1. Negative values must be represented as False or 0.
    probs : list[float]
        The predictions, as a numeric vector of the same length as groundtruth

    Returns : float
        The AUC as a value between 0.5 and 1.

    Examples
    --------
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
        ],
    )

    return aucpr_from_bincounts(
        binned.get_column("pos"),
        binned.get_column("neg"),
        binned.get_column("probs"),
    )


def aucpr_from_bincounts(
    pos: Sequence[int] | pl.Series,
    neg: Sequence[int] | pl.Series,
    probs: Sequence[float] | pl.Series | None = None,
) -> float:
    """Calculates PR AUC (precision-recall) from counts of positives and negatives directly.
    This is an efficient calculation of the area under the PR curve directly from an
    array of positives and negatives. Returns 0.0 when there is just one
    groundtruth label.

    Parameters
    ----------
    pos : list[int]
        Vector with counts of the positive responses
    neg: list[int]
        Vector with counts of the negative responses
    probs: list[float]
        Optional list with probabilities which will be used to set the order of the bins. If missing defaults to pos/(pos+neg).

    Returns
    -------
    float
        The PR AUC as a value between 0.0 and 1.

    Examples
    --------
        >>> aucpr_from_bincounts([3,1,0], [2,0,1])

    """
    import numpy as np

    pos_arr = np.asarray(pos)
    neg_arr = np.asarray(neg)
    if probs is None:
        o = np.argsort(-(pos_arr / (pos_arr + neg_arr)))
    else:
        o = np.argsort(-np.asarray(probs))
    recall = np.cumsum(pos_arr[o]) / np.sum(pos_arr)
    precision = np.cumsum(pos_arr[o]) / np.cumsum(pos_arr[o] + neg_arr[o])
    prevrecall = np.insert(recall[0 : (len(recall) - 1)], 0, 0)
    prevprecision = np.insert(precision[0 : (len(precision) - 1)], 0, 0)
    area = (recall - prevrecall) * (precision + prevprecision) / 2
    return np.sum(area[1:])


def auc_to_gini(auc: float) -> float:
    """Convert AUC performance metric to GINI

    Parameters
    ----------
    auc: float
        The AUC (number between 0.5 and 1)

    Returns
    -------
    float
        GINI metric, a number between 0 and 1

    Examples
    --------
        >>> auc2GINI(0.8232)

    """
    return 2 * safe_range_auc(auc) - 1


# NOTE: the helpers below (z_ratio, lift, bin_log_odds, ...) could be
# consistently named with a ``_polars`` suffix for clarity.


def z_ratio(
    pos_col: str | pl.Expr = pl.col("BinPositives"),
    neg_col: str | pl.Expr = pl.col("BinNegatives"),
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
    if isinstance(pos_col, str):
        pos_col = pl.col(pos_col)
    if isinstance(neg_col, str):
        neg_col = pl.col(neg_col)

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

    pos_frac, neg_frac = get_fracs(pos_col, neg_col)
    return z_ratio_impl(pos_frac, neg_frac, pos_col.sum(), neg_col.sum())


def lift(
    pos_col: str | pl.Expr = pl.col("BinPositives"),
    neg_col: str | pl.Expr = pl.col("BinNegatives"),
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
    if isinstance(pos_col, str):
        pos_col = pl.col(pos_col)
    if isinstance(neg_col, str):
        neg_col = pl.col(neg_col)

    def lift_impl(bin_pos, bin_neg, total_pos, total_neg):
        return (
            # NOTE: when there are no positives at all this could produce
            # NaN/None. Polars supports proper None values, so this likely
            # behaves correctly without special-casing.
            bin_pos * (total_pos + total_neg) / ((bin_pos + bin_neg) * total_pos)
        ).alias("Lift")

    return lift_impl(pos_col, neg_col, pos_col.sum(), neg_col.sum())


# log odds contribution of the bins, including Laplace smoothing
def bin_log_odds(bin_pos: list[float], bin_neg: list[float]) -> list[float]:
    sum_pos = sum(bin_pos)
    sum_neg = sum(bin_neg)
    nbins = len(bin_pos)  # must be > 0
    return [
        (math.log(pos + 1 / nbins) - math.log(sum_pos + 1)) - (math.log(neg + 1 / nbins) - math.log(sum_neg + 1))
        for pos, neg in zip(bin_pos, bin_neg, strict=False)
    ]


def log_odds_polars(
    positives: pl.Expr | str = pl.col("Positives"),
    negatives: pl.Expr | str = pl.col("ResponseCount") - pl.col("Positives"),
) -> pl.Expr:
    """Calculate log odds per bin with correct Laplace smoothing.

    Formula (per bin i in predictor p):
        log(pos_i + 1/nBins) - log(sum(pos) + 1)
        - [log(neg_i + 1/nBins) - log(sum(neg) + 1)]

    Laplace smoothing uses 1/nBins where nBins is the number of bins
    for that specific predictor. This matches the platform implementation
    in GroupedPredictor.java.

    Must be used with .over() to calculate nBins per predictor group:
        .with_columns(log_odds_polars().over("PredictorName", "ModelID"))

    Parameters
    ----------
    positives : pl.Expr or str
        Column with positive response counts per bin
    negatives : pl.Expr or str
        Column with negative response counts per bin

    Returns
    -------
    pl.Expr
        Log odds expression (use with .over() for correct grouping)

    See Also
    --------
    feature_importance : Calculate predictor importance from log odds
    bin_log_odds : Pure Python version (reference implementation)

    References
    ----------
    - ADM Explained: Log Odds calculation section
    - Issue #263: https://github.com/pegasystems/pega-datascientist-tools/issues/263
    - Platform: GroupedPredictor.java lines 603-606

    Examples
    --------
    >>> # For propensity calculation in classifier
    >>> df.with_columns(
    ...     log_odds_polars(
    ...         pl.col("BinPositives"),
    ...         pl.col("BinNegatives")
    ...     ).over("PredictorName", "ModelID")
    ... )
    """
    if isinstance(positives, str):
        positives = pl.col(positives)
    if isinstance(negatives, str):
        negatives = pl.col(negatives)

    nBins = positives.count()  # Correct when used with .over()

    return (
        ((positives + 1 / nBins).log() - (positives.sum() + 1).log())
        - ((negatives + 1 / nBins).log() - (negatives.sum() + 1).log())
    ).alias("LogOdds")


def feature_importance(
    over: list[str] | None = None,
    scaled: bool = True,
) -> pl.Expr:
    """Calculate feature importance for Naive Bayes predictors.

    Feature importance represents the weighted average of absolute log odds
    values across all bins, weighted by bin response counts. This measures
    how strongly the predictor differentiates between positive and negative
    outcomes.

    Algorithm (matches platform GroupedPredictor.calculatePredictorImportance()):
    1. Calculate log odds per bin with Laplace smoothing (1/nBins)
    2. Take absolute value of each bin's log odds
    3. Calculate weighted average: Sum(|logOdds(bin)| × binResponses) / totalResponses
    4. Optional: Scale to 0-100 range (scaled=True, default)

    This matches the Pega platform implementation in:
    adaptive-learning-core-lib/.../GroupedPredictor.java lines 371-382

    Formula:
        Feature Importance = Σ |logOdds(bin)| × (binResponses / totalResponses)

    Parameters
    ----------
    over : list[str], optional
        Grouping columns. Defaults to ``["PredictorName", "ModelID"]``.
    scaled : bool, default True
        If True, scale importance to 0-100 where max predictor = 100

    Returns
    -------
    pl.Expr
        Feature importance expression

    Examples
    --------
    >>> df.with_columns(
    ...     feature_importance().over("PredictorName", "ModelID")
    ... )

    Notes
    -----
    This implementation matches the platform calculation exactly. Issue #263
    incorrectly suggested "diff from mean" based on R implementation, but
    the platform actually uses weighted average of absolute log odds.

    See Also
    --------
    log_odds_polars : Calculate per-bin log odds

    References
    ----------
    - Issue #263: Calculation of Feature Importance incorrect
    - Issue #404: Add feature importance explanation to ADM Explained
    - Platform: GroupedPredictor.java calculatePredictorImportance()
    - ADM Explained: Feature Importance section
    """
    # Step 1: Calculate log odds per bin (must use .over() in calling code)
    log_odds_expr = log_odds_polars(
        pl.col("BinPositives"),
        pl.col("BinResponseCount") - pl.col("BinPositives"),
    )

    # Step 2 & 3: Absolute value, then weighted average
    abs_log_odds = log_odds_expr.abs()
    importance = weighted_average_polars(abs_log_odds, pl.col("BinResponseCount"))

    result = importance.alias("FeatureImportance")

    # Apply grouping for per-predictor aggregation
    if over is None:
        over = ["PredictorName", "ModelID"]
    result = result.over(over)

    # Step 4: Optional scaling (must happen AFTER .over() to scale across all predictors)
    if scaled:
        result = result * 100.0 / result.max()

    return result


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
    index_expr = (pl.int_range(1, pl.len() + 1) / pl.len()) if index is None else (pl.cum_sum(index) / pl.sum(index))

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
            ],
        )
    else:
        by_as_list = by if isinstance(by, list) else [by]
        sort_exprs: list[str | pl.Expr] = by_as_list + [sort_expr]
        gains_df = (
            df.lazy()
            .sort(sort_exprs, descending=True)
            .select(
                by_as_list
                + [
                    index_expr.over(by).cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value)).over(by).cast(pl.Float64).alias("cum_y"),
                ],
            )
        )
        # Add entry for the (0,0) point
        gains_df = pl.concat(
            [gains_df.group_by(by).agg(cum_x=pl.lit(0.0), cum_y=pl.lit(0.0)), gains_df],
        ).sort(by_as_list + ["cum_x"])

    return gains_df.collect()
