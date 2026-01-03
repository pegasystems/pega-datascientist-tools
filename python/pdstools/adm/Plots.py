__all__ = ["Plots"]
import logging
import math
from datetime import timedelta
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import polars as pl
import polars.selectors as cs
from typing_extensions import Concatenate, ParamSpec

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.plot_utils import get_colorscale
from ..utils.types import QUERY

logger = logging.getLogger(__name__)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")

if TYPE_CHECKING:  # pragma: no cover
    from .ADMDatamart import ADMDatamart
COLORSCALE_TYPES = Union[List[Tuple[float, str]], List[str]]

Figure = Union[Any, "go.Figure"]

T = TypeVar("T", bound="Plots")
P = ParamSpec("P")


def requires(
    model_columns: Optional[Iterable[str]] = None,
    predictor_columns: Optional[Iterable[str]] = None,
    combined_columns: Optional[Iterable[str]] = None,
):
    def decorator(
        func: Callable[Concatenate[T, P], Union[Figure, pl.LazyFrame]],
    ) -> Callable[Concatenate[T, P], Union[Figure, pl.LazyFrame]]:
        @overload
        def wrapper(
            self: T, *args: P.args, return_df: Literal[False] = ..., **kwargs: P.kwargs
        ) -> Figure: ...

        @overload
        def wrapper(
            self: T, *args: P.args, return_df: Literal[True], **kwargs: P.kwargs
        ) -> pl.LazyFrame: ...

        @wraps(func)
        def wrapper(
            self: T, *args: P.args, return_df: bool = False, **kwargs: P.kwargs
        ) -> Union[Figure, pl.LazyFrame]:
            # Validation logic (unchanged)
            if model_columns:
                if self.datamart.model_data is None:
                    raise ValueError("Missing data: model_data")
                missing = {
                    c
                    for c in model_columns
                    if c not in self.datamart.model_data.collect_schema().names()
                }
                if missing:
                    raise ValueError(f"Missing required model columns:{missing}")

            if predictor_columns:
                if self.datamart.predictor_data is None:
                    raise ValueError("Missing data: predictor_data")
                missing = {
                    c
                    for c in predictor_columns
                    if c not in self.datamart.predictor_data.collect_schema().names()
                }
                if missing:
                    raise ValueError(f"Missing required predictor columns:{missing}")

            if combined_columns:
                if self.datamart.combined_data is None:
                    raise ValueError("Missing data: combined_data")
                missing = {
                    c
                    for c in combined_columns
                    if c not in self.datamart.combined_data.collect_schema().names()
                }
                if missing:
                    raise ValueError(f"Missing required combined columns:{missing}")

            return func(self, *args, return_df=return_df, **kwargs)

        return wrapper

    return decorator


def fig_update_facet(
    fig: Figure, n_cols: int = 2, base_height: int = 250, step_height: int = 270
) -> Figure:
    """Update faceted plot layout with proper height and simplified annotation text.

    This utility function adjusts the height of faceted plots based on the number of
    facets and columns, and simplifies facet annotation text by showing only the
    value part after "=" splits.

    Parameters
    ----------
    fig : Figure
        The plotly figure to update
    n_cols : int, optional
        Number of columns in the facet layout, by default 2
    base_height : int, optional
        Base height for the plot, by default 250
    step_height : int, optional
        Additional height per row of facets, by default 270

    Returns
    -------
    Figure
        The updated plotly figure
    """
    n_rows = max(math.ceil(len(fig.layout.annotations) / n_cols), 1)
    height = base_height + (n_rows * step_height)
    return fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[1])
    ).update_layout(autosize=True, height=height)


def add_bottom_left_text_to_bubble_plot(
    fig: Figure, df: pl.LazyFrame, bubble_size: int
):
    def get_nonperforming_models(df: pl.LazyFrame):
        return (
            df.filter(
                (pl.col("Performance") == 50)
                & ((pl.col("SuccessRate").is_null()) | (pl.col("SuccessRate") == 0))
            )
            .select(pl.first().count())
            .collect()
            .item()
        )

    if len(fig.layout.annotations) > 0:
        for i in range(0, len(fig.layout.annotations)):
            oldtext = fig.layout.annotations[i].text.split("=")
            subset = df.filter(pl.col(oldtext[0]) == oldtext[1])
            num_models = subset.select(pl.first().count()).collect().item()
            if num_models > 0:
                bottomleft = get_nonperforming_models(subset)
                newtext = f"{num_models} models: {bottomleft} ({round(bottomleft / num_models * 100, 2)}%) at (50,0)"
                fig.layout.annotations[i].text += f"<br><sup>{newtext}</sup>"
                if len(fig.data) > i:
                    fig.data[i].marker.size *= bubble_size
                else:
                    print(fig.data, i)
        return fig
    num_models = df.select(pl.first().len()).collect().item()
    bottomleft = get_nonperforming_models(df)
    newtext = f"{num_models} models: {bottomleft} ({round(bottomleft / num_models * 100, 2)}%) at (50,0)"
    fig.layout.title.text += f"<br><sup>{newtext}</sup>"
    fig.data[0].marker.size *= bubble_size
    return fig


def distribution_graph(df: pl.LazyFrame, title: str):
    plot_df = df.collect()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=plot_df["BinSymbol"], y=plot_df["BinResponseCount"], name="Responses")
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["BinSymbol"],
            y=plot_df["BinPropensity"],
            yaxis="y2",
            name="Propensity",
            mode="lines+markers",
        )
    )
    fig.update_layout(
        template="pega", title=title, xaxis_title="Range", yaxis_title="Responses"
    )
    fig.update_yaxes(title_text="Propensity", secondary_y=True)
    fig.layout.yaxis2.tickformat = ",.3%"
    fig.layout.yaxis2.zeroline = False
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(type="category")

    return fig


class Plots(LazyNamespace):
    dependencies = ["plotly"]
    dependency_group = "adm"

    def __init__(self, datamart: "ADMDatamart"):
        self.datamart = datamart
        super().__init__()

    @requires({"ModelID", "Performance", "SuccessRate", "ResponseCount", "Name"})
    def bubble_chart(
        self,
        *,
        last: bool = True,
        rounding: int = 5,
        query: Optional[QUERY] = None,
        facet: Optional[Union[str, pl.Expr]] = None,
        color: Optional[str] = "Performance",
        return_df: bool = False,
    ):
        """The Bubble Chart, as seen in Prediction Studio

        Parameters
        ----------
        last : bool, optional
            Whether to only include the latest snapshot, by default True
        rounding: int, optional
            To how many digits to round the performance number
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[Union[str, pl.Expr]], optional
            Column name or Polars expression to facet the plot into subplots, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False
        """
        # why do we need this select? it's not ideal because columns used in the query are not always selected
        columns_to_select = list(
            set(
                [
                    "ModelID",
                    color,
                    "SuccessRate",
                    "ResponseCount",
                    "ModelTechnique",
                    "SnapshotTime",
                    "LastUpdate",
                    "Configuration",
                    *self.datamart.context_keys,
                ]
            )
        ) + (["Performance"] if color != "Performance" else [])
        if facet is not None:
            if isinstance(facet, pl.Expr):
                facet_columns = facet.meta.root_names()
                columns_to_select.extend(
                    col for col in facet_columns if col not in columns_to_select
                )
                facet_name = facet.meta.output_name()
            else:
                if facet not in columns_to_select:
                    columns_to_select.append(facet)
                facet_name = facet
        else:
            facet_name = None
        df = (
            (self.datamart.aggregates.last() if last else self.datamart.model_data)
            .select(*columns_to_select)
            .with_columns((pl.col("Performance") * pl.lit(100)).round(rounding))
        )

        if facet_name is not None and isinstance(facet, pl.Expr):
            df = df.with_columns(facet.alias(facet_name))
        df = cdh_utils._apply_query(df, query)

        if return_df:
            return df

        title = "over all models"
        fig = px.scatter(
            df.collect().with_columns(
                pl.col("LastUpdate").dt.strftime("%v"),
                pl.col(self.datamart.context_keys).fill_null(""),
            ),
            x="Performance",
            y="SuccessRate",
            color=color,
            size="ResponseCount",
            facet_col=facet_name,
            facet_col_wrap=2,
            hover_name="Name",
            hover_data=["ModelID"] + self.datamart.context_keys + ["LastUpdate"],
            title=f"Bubble Chart {title}",
            template="pega",
            facet_row_spacing=0.01,
            labels={"LastUpdate": "Last Updated"},
        )
        fig = add_bottom_left_text_to_bubble_plot(fig, df, 1)
        fig.update_traces(marker=dict(line=dict(color="black")))
        fig.update_yaxes(tickformat=".3%")
        return fig

    # TODO support ["Channel", "Direction"] - mulitiple by's in over_time

    @requires({"SnapshotTime"})
    def over_time(
        self,
        metric: str = "Performance",
        by: Union[pl.Expr, str] = "ModelID",
        *,
        every: Union[str, timedelta] = "1d",
        cumulative: bool = True,
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: bool = False,
    ):
        """Statistics over time

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "Performance"
        by : Union[pl.Expr, str], optional
            The column to group by, by default "ModelID"
        every : Union[str, timedelta], optional
            By what time period to group, by default "1d", see https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html
            for periods.
        cumulative : bool, optional
            Whether to show cumulative values or period-over-period changes, by default True
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[str], optional
            Whether to facet the plot into subplots, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False
        """
        percentage_metrics = ["Performance", "SuccessRate"]
        metric_formatting = {
            "SuccessRate": ":.4%",
            "Performance": ":.2",
            "Positives": ":.d",
            "ResponseCount": ":.d",
        }
        if not isinstance(by, pl.Expr):
            by = pl.col(by)
        by_col = by.meta.output_name()

        if self.datamart.model_data is None:
            raise ValueError("Visualisation requires model_data")

        columns_to_select: Set[str] = {"SnapshotTime", metric, "ResponseCount"}
        columns_to_select.update(by.meta.root_names())
        if facet:
            columns_to_select.add(facet)

        is_percentage = metric in percentage_metrics
        metric_scaling = pl.lit(100.0 if metric == "Performance" else 1.0)

        df = (
            cdh_utils._apply_query(self.datamart.model_data, query)
            .sort("SnapshotTime")
            .select(list(columns_to_select))
        )

        grouping_columns = [by_col]
        if facet:
            grouping_columns.append(facet)
        df = df.with_columns(by).set_sorted("SnapshotTime")

        agg_expr = [
            (
                metric_scaling
                * cdh_utils.weighted_average_polars(metric, "ResponseCount")
                if is_percentage
                else pl.sum(metric)
            ).alias(metric)
        ]
        if is_percentage:
            agg_expr.append(pl.sum("ResponseCount"))

        df = (
            df.with_columns(pl.col("SnapshotTime"))
            .group_by(grouping_columns + ["SnapshotTime"])
            .agg(agg_expr)
            .sort("SnapshotTime", by_col)
        )

        unique_intervals = df.select(pl.col("SnapshotTime").unique()).collect().height

        if not cumulative and unique_intervals <= 1:
            logger.warning(
                f"Only one {every} interval of data found. Cannot calculate interval-to-interval differences. "
                "Automatically switching to cumulative mode to show values instead."
            )
            cumulative = True

        plot_metric = metric
        if not cumulative:
            plot_metric = f"{metric}_change"
            df = (
                df.group_by_dynamic(
                    "SnapshotTime", every=every, group_by=grouping_columns
                )
                .agg(pl.last(metric))
                .with_columns(
                    pl.col(metric).diff().over(grouping_columns).alias(plot_metric)
                )
            )

        if return_df:
            return df
        final_df = df.collect()

        facet_col_wrap = None
        if facet:
            unique_facet_values = final_df.select(facet).unique().height
            facet_col_wrap = max(2, int(unique_facet_values**0.5))

        title = "over all models" if facet is None else f"per {facet}"
        fig = px.line(
            final_df,
            x="SnapshotTime",
            y=plot_metric,
            color=by_col,
            hover_data={
                by_col: ":.d",
                plot_metric: metric_formatting[metric.split("_")[0]],
            },
            markers=True,
            title=f"{metric} over time, per {by_col} {title}",
            facet_col=facet,
            facet_col_wrap=facet_col_wrap,
            template="pega",
        )

        if metric == "SuccessRate":
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(yaxis={"rangemode": "tozero"})

        return fig

    @requires({"ModelID", "Name"})
    def proposition_success_rates(
        self,
        metric: str = "SuccessRate",
        by: str = "Name",
        *,
        top_n: int = 0,
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: bool = False,
    ):
        """Proposition Success Rates

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "SuccessRate"
        by : str, optional
            By which column to group the, by default "Name"
        top_n : int, optional
            Whether to take a top_n on the `by` column, by default 0
        query : Optional[QUERY], optional
            A query to apply to the data, by default None
        facet : Optional[str], optional
            What facetting column to apply to the graph, by default None
        return_df : bool, optional
            Whether to return a DataFrame instead of the graph, by default False
        """
        if self.datamart.model_data is None:
            raise ValueError("Visualisation requires model_data")

        df = cdh_utils._apply_query(self.datamart.model_data, query).select(
            {"ModelID", "Name", metric, by, facet}
        )

        if return_df:
            return df

        title = "over all models" if facet is None else f"per {facet}"
        fig = px.histogram(
            df.collect(),
            x=metric,
            y=by,
            color=by,
            facet_col=facet,
            facet_col_wrap=5,
            histfunc="avg",
            title=f"{metric} of each proposition {title}",
            template="pega",
        )
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(dtick=1, automargin=True)
        errors = df.group_by(by).agg(pl.std("SuccessRate").fill_null(0)).collect()
        for i, bar in enumerate(fig.data):
            fig.data[i]["error_x"] = {
                "array": [errors.filter(Name=bar["name"])["SuccessRate"].item()],
                "valueminus": 0,
            }
        return fig

    @requires(
        combined_columns={
            "PredictorName",
            "Name",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
            "ModelID",
        }
    )
    def score_distribution(
        self,
        model_id: str,
        *,
        active_range: bool = True,
        return_df: bool = False,
    ):
        """Generate a score distribution plot for a specific model.

        Parameters
        ----------
        model_id : str
            The ID of the model to generate the score distribution for
        active_range : bool, optional
            Whether to filter to active score range only, by default True
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly figure showing score distribution or DataFrame if return_df=True

        Raises
        ------
        ValueError
            If no data is available for the provided model ID
        """
        df = (
            self.datamart.aggregates.last(table="combined_data")
            .select(
                {
                    "PredictorName",
                    "Name",
                    "BinIndex",
                    "BinSymbol",
                    "BinResponseCount",
                    "BinPropensity",
                    "ModelID",
                    "Configuration",
                }
                | set(
                    self.datamart.context_keys,
                )
            )
            .filter(PredictorName="Classifier", ModelID=model_id)
        ).sort("BinIndex")

        if active_range:
            active_ranges = self.datamart.active_ranges(model_id).collect()
            if active_ranges.height > 0:
                active_range_info = active_ranges.to_dicts()[0]
                active_range_filter_expr = (
                    pl.col("BinIndex") >= active_range_info["idx_min"]
                ) & (pl.col("BinIndex") <= active_range_info["idx_max"])
                df = df.filter(active_range_filter_expr)

        if df.select(pl.first().len()).collect().item() == 0:
            raise ValueError(f"There is no data for the provided modelid '{model_id}'")

        if return_df:
            return df

        context = "/".join(
            df.select(
                pl.col("Configuration", *self.datamart.context_keys).fill_null(
                    "MISSING"
                )
            )
            .unique()
            .collect()
            .row(0)
        )
        return distribution_graph(
            df,
            f"""Classifier score distribution<br>
            <sup>{context}</sup>
            """,
        )

    def multiple_score_distributions(
        self, query: Optional[QUERY] = None, show_all: bool = True
    ) -> List[Figure]:
        """Generate the score distribution plot for all models in the query

        Parameters
        ----------
        query : Optional[QUERY], optional
            A query to apply to the data, by default None
        show_all : bool, optional
            Whether to 'show' all plots or just get a list of them, by default True

        Returns
        -------
        List[go.Figure]
            A list of Plotly charts, one for each model instance
        """
        plots = []
        for model_id in (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="combined_data"),
                query,
            )
            .select(pl.col("ModelID").unique())
            .collect()["ModelID"]
        ):
            fig = self.score_distribution(model_id=model_id)
            if show_all:
                fig.show()
            plots.append(fig)
        return plots

    @requires(
        combined_columns={
            "PredictorName",
            "Name",
            "ModelID",
            "Configuration",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
        }
    )
    def predictor_binning(
        self, model_id: str, predictor_name: str, return_df: bool = False
    ):
        """Generate a predictor binning plot for a specific model and predictor.

        Parameters
        ----------
        model_id : str
            The ID of the model containing the predictor
        predictor_name : str
            Name of the predictor to analyze
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly figure showing predictor binning or DataFrame if return_df=True

        Raises
        ------
        ValueError
            If no data is available for the provided model ID and predictor name
        """
        df = (
            self.datamart.aggregates.last(table="combined_data")
            .select(
                {
                    "PredictorName",
                    "Name",
                    "ModelID",
                    "Configuration",
                    "BinIndex",
                    "BinSymbol",
                    "BinResponseCount",
                    "BinPropensity",
                }
                | set(
                    self.datamart.context_keys,
                )
            )
            .filter(
                pl.col("PredictorName") == predictor_name, pl.col("ModelID") == model_id
            )
        ).sort("BinIndex")

        if df.select(pl.first().len()).collect().item() == 0:
            raise ValueError(
                f"There is no data for the provided modelid {model_id} and predictor {predictor_name}"
            )

        if return_df:
            return df
        context = "/".join(
            df.select(
                pl.col("Configuration", *self.datamart.context_keys).fill_null(
                    "MISSING"
                )
            )
            .unique()
            .collect()
            .row(0)
        )
        return distribution_graph(
            df, title=f"""Predictor binning for {predictor_name}<br><sup>{context}"""
        )

    def multiple_predictor_binning(
        self, model_id: str, query: Optional[QUERY] = None, show_all=True
    ) -> List[Figure]:
        """Generate predictor binning plots for all predictors in a model.

        Parameters
        ----------
        model_id : str
            The ID of the model to generate predictor binning plots for
        query : Optional[QUERY], optional
            A query to apply to the predictor data, by default None
        show_all : bool, optional
            Whether to display all plots or just return the list, by default True

        Returns
        -------
        List[Figure]
            A list of Plotly figures, one for each predictor in the model
        """
        plots = []
        for predictor in (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="predictor_data"),
                query,
            )
            .filter(pl.col("ModelID") == model_id)
            .select(pl.col("PredictorName").unique())
            .sort("PredictorName")
            .collect()["PredictorName"]
        ):
            fig = self.predictor_binning(model_id=model_id, predictor_name=predictor)
            if show_all:
                fig.show()
            plots.append(fig)
        return plots

    def _boxplot_pre_aggregated(
        self,
        df: pl.LazyFrame,
        *,
        y_col: str,
        metric_col: str,
        metric_weight_col: Optional[str] = None,
        legend_col: Optional[str] = None,
        return_df: bool = False,
    ):
        if legend_col is None:
            legend_col = y_col

        if metric_weight_col is None:
            mean_expr = pl.mean(metric_col)
        else:
            mean_expr = cdh_utils.weighted_average_polars(metric_col, metric_weight_col)
        pre_aggs = (
            df.group_by(list({y_col, legend_col}))
            .agg(
                median=pl.median(metric_col),
                mean=mean_expr,
                q1=pl.quantile(metric_col, 0.25),
                q3=pl.quantile(metric_col, 0.75),
                min=pl.min(metric_col),
                max=pl.max(metric_col),
                whisker_low=pl.quantile(metric_col, 0.1),
                whisker_high=pl.quantile(metric_col, 0.9),
            )
            .with_columns(iqr=pl.col("q3") - pl.col("q1"))
            # limiting the decimals should reduce the size of
            # the plotly data in HTML significantly. TODO not sure thats true!!
            .with_columns(cs.numeric().round(3))
            .sort(["median", y_col], descending=True)
            .collect()
        )

        if return_df:
            return pre_aggs

        if pre_aggs.select(pl.first().len()).item() == 0:
            return None

        fig = go.Figure()

        # Fixed colors for specific predictor categories
        # TODO move elsewhere
        fixed_colors = {
            "IH": "#1f77b4",  # Blue
            "Param": "#ff7f0e",  # Orange
            "Primary": "#2ca02c",  # Green
            "Other": "#d62728",  # Red
        }

        # Fallback colors from pega template for other categories
        template_colors = [
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # TODO bad code below
        color_map = {}
        template_color_index = 0
        for cat in pre_aggs.select(pl.col(legend_col).unique())[legend_col].to_list():
            if cat in fixed_colors:
                color_map[cat] = fixed_colors[cat]
            else:
                color_map[cat] = template_colors[
                    template_color_index % len(template_colors)
                ]
                template_color_index += 1

        # Track which categories have been added to legend
        legend_added = set()

        # Create a box plot for each predictor
        for i, row in enumerate(pre_aggs.iter_rows(named=True)):
            show_in_legend = row[legend_col] not in legend_added
            if show_in_legend:
                legend_added.add(row[legend_col])

            fig.add_trace(
                go.Box(
                    q1=[row["q1"]],
                    median=[row["median"]],
                    q3=[row["q3"]],
                    lowerfence=[row["whisker_low"]],
                    upperfence=[row["whisker_high"]],
                    mean=[row["mean"]],
                    y=[row[y_col]],
                    boxpoints=False,
                    marker=dict(color=color_map[row[legend_col]]),
                    name=row[legend_col],
                    legendgroup=row[legend_col],
                    orientation="h",
                    showlegend=show_in_legend,
                )
            )

        fig.update_layout(
            title=f"{metric_col} by {legend_col}",
            xaxis_title=metric_col,
            yaxis_title="",
            template="pega",
        )

        # Set y-axis category order to show highest median values at the top
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=pre_aggs[y_col].to_list(),
            automargin=True,
            autorange="reversed",
        )
        return fig

    @requires(
        combined_columns={
            "ModelID",
            "PredictorName",
            "PredictorCategory",
            "ResponseCountBin",
            "Type",
            "EntryType",
        }
    )
    def predictor_performance(
        self,
        *,
        metric: str = "Performance",
        top_n: Optional[int] = None,
        active_only: bool = False,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Plots a box plot of the performance of the predictors

        Use the query argument to drill down to a more specific subset
        If top n is given, chooses the top predictors based on the
        weighted average performance across models, ordered by their median performance.

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "Performance"
            This is more for future-proofing, once FeatureImportance gets more used.
        top_n : Optional[int], optional
            The top n predictors to plot, by default None
        active_only : bool, optional
            Whether to only consider active predictor performance, by default False
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        See also
        --------
        pdstools.adm.ADMDatamart.apply_predictor_categorization : how to override the out of the box predictor categorization
        """

        # in combined_data, the performance metric is renamed to PredictorPerformance
        metric = "PredictorPerformance" if metric == "Performance" else metric
        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data")
            .with_columns(
                pl.col("PredictorPerformance") * 100.0,
            )
            .filter(pl.col("EntryType") != "Classifier")
            .filter(pl.col("ResponseCountBin") > 0)
            .unique(subset=["ModelID", "PredictorName"], keep="first"),
            query=query,
            allow_empty=True,
        )
        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")
        if top_n:
            df = self.datamart.aggregates._top_n(
                df,
                top_n,
                metric,
            )

        fig = self._boxplot_pre_aggregated(
            df,
            y_col="PredictorName",
            metric_col=metric,
            # It is confusing, but ResponseCountBin is the correct total response
            # count for a predictor. It is the sum of BinResponseCount. Thanks SK :)
            metric_weight_col="ResponseCountBin",
            legend_col="PredictorCategory",
            return_df=return_df,
        )

        if fig is not None and not return_df:
            fig.update_xaxes(title="Performance")
            fig.update_layout(title=f"{metric} by Predictor Category")

        return fig

    @requires(
        combined_columns={
            "ModelID",
            "PredictorName",
            "PredictorCategory",
            "ResponseCountBin",
            "Type",
            "EntryType",
        }
    )
    def predictor_category_performance(
        self,
        *,
        metric: str = "Performance",
        active_only: bool = False,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Plot the predictor category performance

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "Performance"
        active_only : bool, optional
            Whether to only analyze active predictors, by default False
        query : Optional[QUERY], optional
            An optional query to apply, by default None
        return_df : bool, optional
            An optional flag to get the dataframe instead, by default False

        Returns
        -------
        px.Figure
            A Plotly figure


        See also
        --------
        pdstools.adm.ADMDatamart.apply_predictor_categorization : how to override the out of the box predictor categorization
        """
        metric = "PredictorPerformance" if metric == "Performance" else metric

        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data")
            .with_columns(
                pl.col("PredictorPerformance") * 100.0,
            )
            .filter(pl.col("EntryType") != "Classifier")
            .filter(pl.col("ResponseCountBin") > 0),
            query=query,
        )
        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")

        fig = self._boxplot_pre_aggregated(
            df,
            y_col="PredictorCategory",
            metric_col="PredictorPerformance",
            legend_col="PredictorCategory",
            return_df=return_df,
        )
        if fig is not None and not return_df:
            fig.update_xaxes(title="Performance")

        return fig

    @requires(
        combined_columns={
            "PredictorName",
            "PredictorPerformance",
            "BinResponseCount",
            "PredictorCategory",
        }
    )
    def predictor_contribution(
        self,
        *,
        by: str = "Configuration",
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Plots the predictor contribution for each configuration

        Parameters
        ----------
        by : str, optional
            By which column to plot the contribution, by default "Configuration"
        query : Optional[QUERY], optional
            An optional query to apply to the data, by default None
        return_df : bool, optional
            An optional flag to get a Dataframe instead, by default False

        Returns
        -------
        px.Figure
            A plotly figure

        See also
        --------
        pdstools.adm.ADMDatamart.apply_predictor_categorization : how to override the out of the box predictor categorization
        """
        df = (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="combined_data"),
                query,
            )
            .filter(pl.col("PredictorName") != "Classifier")
            # Huh?
            .with_columns((pl.col("PredictorPerformance") - 0.5) * 2)
            .group_by(by, "PredictorCategory")
            .agg(
                Performance=cdh_utils.weighted_average_polars(
                    "PredictorPerformance", "BinResponseCount"
                )
            )
            .with_columns(
                Contribution=(pl.col("Performance") / pl.sum("Performance").over(by))
                * 100
            )
            .sort("PredictorCategory")
        )
        if return_df:
            return df

        fig = px.bar(
            df.collect(),
            x="Contribution",
            y=by,
            color="PredictorCategory",
            orientation="h",
            template="pega",
            title="Contribution of different sources",
        )
        return fig

    @requires(
        combined_columns={
            "PredictorName",
            "Name",
            "Performance",
            "PredictorPerformance",
            "ResponseCountBin",
        }
    )
    def predictor_performance_heatmap(
        self,
        *,
        top_predictors: int = 20,
        top_groups: Optional[int] = None,
        by: str = "Name",
        active_only: bool = False,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Generate a heatmap showing predictor performance across different groups.

        Parameters
        ----------
        top_predictors : int, optional
            Number of top-performing predictors to include, by default 20
        top_groups : int, optional
            Number of top groups to include, by default None (all groups)
        by : str, optional
            Column to group by for the heatmap, by default "Name"
        active_only : bool, optional
            Whether to only include active predictors, by default False
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly heatmap figure or DataFrame if return_df=True
        """
        if isinstance(by, str):
            by_name = by
        else:
            by = by.alias("Predictor")
            by_name = "Predictor"

        df = self.datamart.aggregates.predictor_performance_pivot(
            query=query,
            by=by,
            top_predictors=top_predictors,
            top_groups=top_groups,
            active_only=active_only,
        )

        df = df.collect().transpose(
            include_header=True, header_name=by_name, column_names=by_name
        )

        if return_df:
            return df.lazy()

        title = "over all models"
        fig = px.imshow(
            df.select(pl.all().exclude(by_name)),
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale=get_colorscale("Performance"),
            title=f"Top predictors {title}",
            range_color=[0.5, 1],
            y=df[by_name],
        )

        fig.update_yaxes(dtick=1, automargin=True)
        fig.update_xaxes(
            dtick=1,
        )
        return fig

    def response_gain(): ...  # TODO: more generic plot_gains function?

    def tree_map(
        self,
        metric: Literal[
            "ResponseCount",
            "Positives",
            "Performance",
            "SuccessRate",
            "percentage_without_responses",
        ] = "Performance",
        *,
        by: str = "Name",
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Generate a tree map visualization showing hierarchical model metrics.

        Parameters
        ----------
        metric : Literal["ResponseCount", "Positives", "Performance", "SuccessRate", "percentage_without_responses"], optional
            The metric to visualize in the tree map, by default "Performance"
        by : str, optional
            Column to group by for the tree map hierarchy, by default "Name"
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly treemap figure or DataFrame if return_df=True
        """
        # TODO: clean up implementation a bit

        group_by = (
            self.datamart.context_keys[: self.datamart.context_keys.index(by)]
            if by != "ModelID"
            else [by]
        )
        df = self.datamart.aggregates.model_summary(by=by, query=query).select(
            pl.col(group_by).cast(pl.Utf8).fill_null("Missing"),
            pl.col("count").alias("Model Count"),
            pl.col("Percentage_without_responses").alias(
                "Percentage without responses"
            ),
            pl.col("ResponseCount_sum").alias("Total number of responses"),
            pl.col("Weighted_success_rate").alias("Weighted average Success Rate"),
            pl.col("Weighted_performance").alias("Weighted average Performance"),
            pl.col("Positives_sum").alias("Total number of positives"),
        )
        if return_df:
            return df

        context_keys = [px.Constant("All contexts")] + group_by

        colorscale = get_colorscale(metric)

        label_map = {
            "ResponseCount": "Total number of responses",
            "Positives": "Total number of positives",
            "Performance": "Weighted average Performance",
            "SuccessRate": "Weighted average Success Rate",
            "percentage_without_responses": "Percentage without responses",
        }

        hover_data = {
            "Model Count": ":.d",
            "Percentage without responses": ":.0%",
            "Total number of positives": ":.d",
            "Total number of responses": ":.d",
            "Weighted average Success Rate": ":.3%",
            "Weighted average Performance": ":.2%",
        }

        fig = px.treemap(
            df.collect(),
            path=context_keys,
            color=label_map.get(metric),
            values="Model Count",
            title=f"{label_map.get(metric)} by {by}",
            hover_data=hover_data,
            color_continuous_scale=colorscale,
            range_color=[0.5, 1] if metric == "Performance" else None,
        )

        return fig

    def predictor_count(
        self,
        *,
        by: Union[str, List[str]] = ["EntryType", "Type"],
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Generate a box plot showing the distribution of predictor counts by type.

        Parameters
        ----------
        by : Union[str, List[str]], optional
            Column(s) to group predictors by, by default ["EntryType", "Type"]
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly box plot figure or DataFrame if return_df=True
        """
        if isinstance(by, str):
            by = [by]

        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data"), query
        ).filter(pl.col("EntryType") != "Classifier")

        df = (
            df.group_by(["ModelID"] + by)
            .agg(Count=pl.n_unique("PredictorName"))
            .collect()
        )

        if len(by) > 1:
            df = pl.concat(
                [
                    df,
                    df.group_by(["ModelID"] + by[1:])
                    .agg(pl.col("Count").sum())
                    .with_columns(pl.lit("Overall").alias(by[0])),
                ],
                how="diagonal_relaxed",
            )
        df = df.sort(by)

        if return_df:
            return df

        fig = px.box(
            df.with_columns(_Type=pl.concat_str(reversed(by), separator=" / ")),
            x="Count",
            y="_Type",
            color=by[0],
            template="pega",
        )

        # Update title and x-axis label if we have a figure (not None and not a DataFrame)
        if fig is not None and not return_df:
            fig.update_layout(title="Predictors by Type")
            fig.update_xaxes(title="Number of Predictors")
            # Ensure y-axis labels fit properly by increasing left margin
            fig.update_yaxes(automargin=True)
            fig.update_layout(margin=dict(l=150))
            fig.update_yaxes(title="")
            # Reverse the legend order
            fig.update_layout(legend=dict(traceorder="reversed"))

        return fig

    def binning_lift(
        self,
        model_id: str,
        predictor_name: str,
        *,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Generate a binning lift plot for a specific predictor showing propensity lift per bin.

        Parameters
        ----------
        model_id : str
            The ID of the model containing the predictor
        predictor_name : str
            Name of the predictor to analyze for lift
        query : Optional[QUERY], optional
            Optional query to filter the predictor data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly bar chart showing binning lift or DataFrame if return_df=True
        """
        df = cdh_utils._apply_query(
            (
                self.datamart.aggregates.last(table="predictor_data")
                .filter(
                    pl.col("PredictorName") == predictor_name,
                    pl.col("ModelID") == model_id,
                )
                .sort("BinIndex")
            ),
            query,
        ).select(
            "PredictorName", "BinIndex", "BinPositives", "BinNegatives", "BinSymbol"
        )
        cols = df.collect_schema().names()

        if "Lift" not in cols:
            df = df.with_columns(
                (
                    cdh_utils.lift(pl.col("BinPositives"), pl.col("BinNegatives")) - 1.0
                ).alias("Lift")
            )

        shading_expr = (
            pl.col("BinPositives") <= 5 if "BinPositives" in cols else pl.lit(False)
        )

        plot_df = df.with_columns(
            pl.when((pl.col("Lift") >= 0.0) & shading_expr.not_())
            .then(pl.lit("pos"))
            .when((pl.col("Lift") >= 0.0) & shading_expr)
            .then(pl.lit("pos_shaded"))
            .when((pl.col("Lift") < 0.0) & shading_expr.not_())
            .then(pl.lit("neg"))
            .otherwise(pl.lit("neg_shaded"))
            .alias("Direction"),
            # TODO generalize this, use it in the standard bin plot as well
            # and make sure the resulting labels are unique - with just the
            # truncate they are not necessarily unique
            BinSymbolAbbreviated=pl.when(pl.col("BinSymbol").str.len_chars() < 25)
            .then(pl.col("BinSymbol"))
            .otherwise(
                pl.concat_str([pl.col("BinSymbol").str.slice(0, 25), pl.lit("...")])
            ),
        ).sort(["PredictorName", "BinIndex"])

        if return_df:
            return plot_df

        fig = px.bar(
            plot_df.collect(),
            x="Lift",
            y="BinSymbolAbbreviated",
            color="Direction",
            color_discrete_map={
                "neg": "#A01503",
                "pos": "#5F9F37",
                "neg_shaded": "#DAA9AB",
                "pos_shaded": "#C5D9B7",
            },
            orientation="h",
            title=f"Propensity Lift for {predictor_name}",
            template="pega",
            custom_data=["PredictorName", "BinSymbol"],
            facet_col_wrap=3,
            category_orders=plot_df.collect().to_dict(),
        )
        fig.update_traces(
            hovertemplate="<br>".join(
                ["<b>%{customdata[0]}</b>", "%{customdata[1]}", "<b>Lift: %{x:.2%}</b>"]
            )
        )
        fig.add_vline(x=0, line_color="black")

        fig.update_layout(
            showlegend=False,
            hovermode="y",
        )
        fig.update_xaxes(title="", tickformat=",.2%")
        fig.update_yaxes(
            type="category",
            categoryorder="array",
            automargin=True,
            title="",
            dtick=1,  # show all bins
            matches=None,  # allow independent y-labels if there are row facets
        )
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )  # split plotly facet label, show only right side
        return fig

    def action_overlap(
        self,
        group_col: Union[str, list[str], pl.Expr] = "Channel",
        overlap_col="Name",
        *,
        show_fraction=True,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        """Generate an overlap matrix heatmap showing shared actions across different groups.

        Parameters
        ----------
        group_col : Union[str, list[str], pl.Expr], optional
            Column(s) to group by for overlap analysis, by default "Channel"
        overlap_col : str, optional
            Column containing values to analyze for overlap, by default "Name"
        show_fraction : bool, optional
            Whether to show overlap as fraction or absolute count, by default True
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly heatmap showing action overlap or DataFrame if return_df=True
        """
        df = cdh_utils._apply_query(
            (self.datamart.model_data),
            query,
        )

        if isinstance(group_col, list):
            group_col_name = "/".join(group_col)
            df = df.with_columns(
                pl.concat_str(*group_col, separator="/").alias(group_col_name)
            )
        elif isinstance(group_col, pl.Expr):
            group_col_name = group_col.meta.output_name()
            df = df.with_columns(group_col.alias(group_col_name))
        else:
            group_col_name = group_col

        overlap_data = cdh_utils.overlap_matrix(
            df.group_by(group_col_name)
            .agg(pl.col(overlap_col).unique())
            .sort(group_col_name)
            .collect(),
            overlap_col,
            by=group_col_name,
            show_fraction=show_fraction,
        )

        if return_df:
            return overlap_data

        plt = px.imshow(
            overlap_data.drop(group_col_name),
            text_auto=".1%" if show_fraction else ".d",
            aspect="equal",
            title=f"Overlap of {overlap_col}s",
            x=overlap_data[group_col_name],
            y=overlap_data[group_col_name],
            template="pega",
            labels=dict(
                x=f"{group_col_name} on x", y=f"{group_col_name} on y", color="Overlap"
            ),
        )
        plt.update_coloraxes(showscale=False)
        return plt

    def partitioned_plot(
        self,
        func: Callable,
        facets: List[Dict[str, Optional[str]]],
        show_plots: bool = True,
        *args,
        **kwargs,
    ):
        """Execute a plotting function across multiple faceted subsets of data.

        This method applies a given plotting function to multiple filtered subsets of data,
        where each subset is defined by the facet conditions. It's useful for generating
        multiple plots with different filter conditions applied.

        Parameters
        ----------
        func : Callable
            The plotting function to execute for each facet
        facets : List[Dict[str, Optional[str]]]
            List of dictionaries defining filter conditions for each facet
        show_plots : bool, optional
            Whether to display the plots as they are generated, by default True
        *args : tuple
            Additional positional arguments to pass to the plotting function
        **kwargs : dict
            Additional keyword arguments to pass to the plotting function

        Returns
        -------
        List[Figure]
            List of Plotly figures, one for each facet condition
        """
        figs = []
        existing_query = kwargs.get("query")
        for facet in facets:
            combined_query = existing_query
            for k, v in facet.items():
                if v is None:
                    new_query = pl.col(k).is_null()
                else:
                    new_query = pl.col(k).eq(v)
                if combined_query is not None:
                    combined_query = cdh_utils._combine_queries(
                        combined_query, new_query
                    )
                else:
                    combined_query = new_query

            kwargs["query"] = combined_query
            fig = func(*args, **kwargs)
            figs.append(fig)
        if show_plots and fig is not None:
            # fig.update_layout(title=title)
            fig.show()

        return figs
