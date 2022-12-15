import polars as pl
from ..utils import cdh_utils
import plotly.express as px

def SuccessRate(outcome_col="pyOutcome", **kwargs) -> pl.Expr:
    return (pl.sum(outcome_col) / pl.count(outcome_col)).alias("SuccessRate")


def AUC(outcome_col="pyOutcome", propensity_col="pyModelPropensity") -> pl.Expr:
    return cdh_utils.auc_from_probs(
        pl.col(outcome_col).to_numpy(), pl.col(propensity_col).to_numpy()
    )

def _metricPerPeriod(
    df, period, metrics, OutcomeTime_col="pxOutcomeTime", by=None, **kwargs
):
    if kwargs.pop('cumulative'):
        df = _cumulativeMetricPerPeriod(df, OutcomeTime_col, period, by, metrics)

    df = (
        df.sort(OutcomeTime_col)
        .groupby_dynamic(OutcomeTime_col, every=period, by=by)
        .agg(metrics)
    )
    if isinstance(df, pl.LazyFrame):
        with pl.StringCache():
            return df.collect()
    return df

def _cumulativeMetricPerPeriod(df, period, metrics, OutcomeTime_col="pxOutcomeTime", by=None, **kwargs):
    return NotImplemented()

def successRatePerPeriod(df, period="1d", **kwargs):
    return _metricPerPeriod(df, period, SuccessRate(**kwargs), **kwargs)


def volumesPerPeriod(df, period="1d", **kwargs):
    return _metricPerPeriod(df, period, pl.count().alias("ResponseCount"), **kwargs)

def _plotPerPeriod(
    df: pl.LazyFrame,
    to_plot,
    period,
    color=None,
    facet_col=None,
    facet_row=None,
    **kwargs,
):
    by = [col for col in [color, facet_col, facet_row] if col is not None]
    title = kwargs.pop("title", to_plot)
    if color is not None:
        title += f" per {color}"
    if len(by) == 0:
        by = None
    if facet_col is not None:
        title += f", by {facet_col}"
    if facet_row is not None:
        title += f", by {facet_row}"

    if to_plot == "SuccessRate":
        df = successRatePerPeriod(df, period, by=by, **kwargs)
    elif to_plot == "ResponseCount":
        df = volumesPerPeriod(df, period, by=by, **kwargs)

    return px.line(
        df.to_pandas(),
        x=kwargs.pop("OutcomeTime_col", "pxOutcomeTime"),
        y=to_plot,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        title=title,
        template="none",
    )


def plotSuccessRatePerPeriod(
    df: pl.LazyFrame,
    period,
    cumulative=False,
    color=None,
    facet_col=None,
    facet_row=None,
    **kwargs,
):
    return _plotPerPeriod(
        df, "SuccessRate", period, color, facet_col, facet_row, **kwargs
    )


def plotVolumesPerPeriod(
    df: pl.LazyFrame,
    period,
    cumulative=False,
    color=None,
    facet_col=None,
    facet_row=None,
    **kwargs,
):
    return _plotPerPeriod(
        df, "ResponseCount", period, color, facet_col, facet_row, **kwargs
    )

def plotPropensityDistribution(
    df:pl.LazyFrame,
    channel,
    direction,
    issue,
    group,
    name,
    propensityType,
    **kwargs
):
    channelCol = kwargs.get('Channel_col', 'pyChannel')
    directionCol = kwargs.get('Direction_col', 'pyDirection')
    issueCol = kwargs.get('Issue_col', 'pyIssue')
    groupCol = kwargs.get('Group_col', 'pyGroup')
    actionCol = kwargs.get('pyName_col', 'pyName')
    return px.histogram(df.filter([
        pl.col(channelCol) == channel,
        pl.col(directionCol) == direction,
        pl.col(issueCol) == issue,
        pl.col(groupCol) == group,
        pl.col(actionCol) == name
    ]).select(propensityType).to_pandas())