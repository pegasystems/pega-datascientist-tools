__all__ = ["Plots"]
import logging
from datetime import timedelta
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
from typing_extensions import Concatenate, ParamSpec

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

logger = logging.getLogger(__name__)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from ..utils import pega_template as pega_template
except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

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
                newtext = f"{num_models} models: {bottomleft} ({round(bottomleft/num_models*100, 2)}%) at (50,0)"
                fig.layout.annotations[i].text += f"<br><sup>{newtext}</sup>"
                if len(fig.data) > i:
                    fig.data[i].marker.size *= bubble_size
                else:
                    print(fig.data, i)
        return fig
    num_models = df.select(pl.first().count()).collect().item()
    bottomleft = get_nonperforming_models(df)
    newtext = f"{num_models} models: {bottomleft} ({round(bottomleft/num_models*100, 2)}%) at (50,0)"
    fig.layout.title.text += f"<br><sup>{newtext}</sup>"
    fig.data[0].marker.size *= bubble_size
    return fig


def distribution_graph(df: pl.LazyFrame, title: str):
    plot_df = df.collect().to_pandas(use_pyarrow_extension_array=True)
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
        columns_to_select = [
            "ModelID",
            "Performance",
            "SuccessRate",
            "ResponseCount",
            *self.datamart.context_keys,
        ]
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
            df.collect().to_pandas(use_pyarrow_extension_array=False),
            x="Performance",
            y="SuccessRate",
            color="Performance",
            size="ResponseCount",
            facet_col=facet_name,
            facet_col_wrap=2,
            hover_name="Name",
            hover_data=["ModelID"] + self.datamart.context_keys,
            title=f"Bubble Chart {title}",
            template="pega",
            facet_row_spacing=0.01,
        )
        fig = add_bottom_left_text_to_bubble_plot(fig, df, 1)
        fig.update_traces(marker=dict(line=dict(color="black")))
        fig.update_yaxes(tickformat=".3%")
        return fig

    @requires({"SnapshotTime"})
    def over_time(
        self,
        metric: str = "Performance",
        by: Union[pl.Expr, str] = "ModelID",
        *,
        every: Union[str, timedelta] = "1d",
        cumulative: bool = False,
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
            By what time period to group, by default "1d"
        cumulative : bool, optional
            Whether to take the cumulative value or the absolute one, by default False
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[str], optional
            Whether to facet the plot into subplots, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False
        """
        if not isinstance(by, pl.Expr):
            by = pl.col(by)

        by_col = by.meta.output_name()

        metric_formatting = {
            "SuccessRate_weighted_average": ":.4%",
            "Performance_weighted_average": ":.2",  # is not a percentage!
            "Positives": ":.d",
            "ResponseCount": ":.d",
        }

        if metric == "Performance":
            metric_scaling: pl.Expr = pl.lit(100.0)
        else:
            metric_scaling: pl.Expr = pl.lit(1.0)

        if self.datamart.model_data is None:
            raise ValueError("Visualisation requires model_data")

        columns_to_select: Set[str] = {"SnapshotTime", metric, "ResponseCount"}
        columns_to_select.update(by.meta.root_names())
        if facet:
            columns_to_select.add(facet)

        df = (
            cdh_utils._apply_query(
                self.datamart.model_data.sort(by="SnapshotTime"), query
            )
            .sort("SnapshotTime")
            .select(list(columns_to_select))
        )

        grouping_columns = list(set(by.meta.root_names()))
        df = df.with_columns(by)
        if facet:
            grouping_columns = [by_col, facet]
        else:
            grouping_columns = [by_col]
        if metric in ["Performance", "SuccessRate"]:  # we need to weigh these
            df = (
                df.group_by_dynamic(
                    "SnapshotTime", every=every, group_by=grouping_columns
                )
                .agg(
                    (
                        metric_scaling
                        * cdh_utils.weighted_average_polars(metric, "ResponseCount")
                    ).name.suffix("_weighted_average")
                )
                .sort("SnapshotTime", by_col)
            )
            metric += "_weighted_average"
        elif cumulative:
            df = (
                df.group_by(grouping_columns + ["SnapshotTime"])
                .agg(pl.sum(metric))
                .sort(grouping_columns + ["SnapshotTime"])
            )
        else:
            df = (
                df.with_columns(
                    Delta=pl.col(metric).cast(pl.Int64).diff().over(grouping_columns)
                )
                .group_by_dynamic(
                    "SnapshotTime", every=every, group_by=grouping_columns
                )
                .agg(Increase=pl.sum("Delta"))
            )

        if return_df:
            return df
        final_df = df.collect()
        unique_facet_values = final_df.select(facet).unique().shape[0]
        facet_col_wrap = max(2, int(unique_facet_values**0.5))

        title = "over all models" if facet is None else f"per {facet}"
        fig = px.line(
            final_df.to_pandas(use_pyarrow_extension_array=False),
            x="SnapshotTime",
            y=metric,
            color=by_col,
            hover_data={
                by_col: ":.d",
                metric: metric_formatting[metric],
            },
            markers=True,
            title=f"{metric} over time, per {by_col} {title}",
            facet_col=facet,
            facet_col_wrap=facet_col_wrap,
            template="pega",
        )
        if metric in ["SuccessRate"]:
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
            df.collect().to_pandas(use_pyarrow_extension_array=True),
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
        return_df: bool = False,
    ):
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
            .filter(
                pl.col("PredictorName") == "Classifier", pl.col("ModelID") == model_id
            )
        ).sort("BinIndex")

        if df.select(pl.first().count()).collect().item() == 0:
            raise ValueError(f"There is no data for the provided modelid {model_id}")

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

        if df.select(pl.first().count()).collect().item() == 0:
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

    @requires(
        combined_columns={
            "Channel",
            "PredictorName",
            "ModelID",
            "Name",
            "ResponseCountBin",
            "Type",
            "PredictorCategory",
        }
    )
    def predictor_performance(
        self,
        *,
        metric: str = "Performance",
        top_n: Optional[int] = None,
        active_only: bool = False,
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: bool = False,
    ):
        """Plots a bar chart of the performance of the predictors

        By default, shows the performance over all models.
        Use the query argument to drill down to a more specific subset
        If top n is given, chooses the top predictors based on the
        weighted average performance across models, sorted by their median performance.

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "Performance"
            This is more for future-proofing, once FeatureImportant gets more used.
        top_n : Optional[int], optional
            The top n predictors to plot, by default None
        active_only : bool, optional
            Whether to only consider active predictor performance, by default False
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[str], optional
            Whether to facet the plot into subplots, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        See also
        --------
        pdstools.adm.ADMDatamart.apply_predictor_categorization : how to override the out of the box predictor categorization
        """

        metric = "PredictorPerformance" if metric == "Performance" else metric
        try:
            df = cdh_utils._apply_query(
                self.datamart.aggregates.last(table="combined_data")
                .select(
                    {
                        "Channel",
                        "PredictorName",
                        "ModelID",
                        "Name",
                        "ResponseCountBin",
                        "Type",
                        "PredictorCategory",
                        "Configuration",
                        metric,
                        facet,
                    }
                    | set(
                        self.datamart.context_keys,
                    )
                )
                .filter(pl.col("PredictorName") != "Classifier")
                .unique(subset=["ModelID", "PredictorName"], keep="first")
                .rename({"PredictorCategory": "Legend"}),
                query=query,
            )
        except ValueError:
            return None
        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")
        if top_n:
            df = self.datamart.aggregates._top_n(
                df, top_n, metric, facets=[facet] if facet else None
            )

        order = (
            df.group_by("PredictorName")
            .agg(pl.median(metric).name.prefix("median_"))
            .fill_nan(0)
            .sort(f"median_{metric}", descending=False)
            .select("PredictorName")
            .collect()["PredictorName"]
        )

        if return_df:
            return df.rename({"Legend": "PredictorCategory"})

        title_suffix = "over all models" if facet is None else f"per {facet}"
        title_prefix = metric

        df = df.collect().to_pandas(use_pyarrow_extension_array=True)
        y = "PredictorName"
        if len(order) > 0:
            df[y] = df[y].astype("category")
            df[y] = df[y].cat.set_categories(order.to_list())

        fig = px.box(
            df.sort_values([y]),
            x=metric,
            y=y,
            color="Legend",
            template="pega",
            title=f"{title_prefix} {title_suffix}",
            facet_col=facet,
            facet_col_wrap=5,
            labels={
                "PredictorName": "Predictor Name",
                "PredictorPerformance": "Performance",
            },
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=order, automargin=True, dtick=1
        )

        fig.update_layout(
            boxgap=0, boxgroupgap=0, legend_title_text="Predictor category"
        )
        return fig

    @requires(
        combined_columns={
            "ModelID",
            "Configuration",
            "Channel",
            "Direction",
            "PredictorName",
            "ResponseCountBin",
            "Type",
            "PredictorCategory",
        }
    )
    def predictor_category_performance(
        self,
        *,
        metric: str = "Performance",
        active_only: bool = False,
        query: Optional[QUERY] = None,
        facet: Optional[Union[pl.Expr, str]] = None,
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
        facet : Optional[Union[pl.Expr, str]], optional
            By which columns to facet the result, by default None
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

        # Determine columns to select and grouping
        select_columns = {
            "ModelID",
            "Configuration",
            "Channel",
            "Direction",
            "PredictorName",
            "ResponseCountBin",
            "Type",
            "PredictorCategory",
            metric,
        } | set(self.datamart.context_keys)

        groups = [pl.col("ModelID"), pl.col("PredictorCategory")]

        facet_col = None
        if isinstance(facet, str):
            select_columns.add(facet)
            groups.insert(0, pl.col(facet))
        elif facet is not None:
            select_columns |= set(facet.meta.root_names())
            groups.insert(0, facet)
            facet_col = facet.meta.output_name()

        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data")
            .select(select_columns)
            .filter(pl.col("PredictorName") != "Classifier"),
            query=query,
        )

        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")

        df = df.group_by(groups).agg(
            PredictorPerformance=cdh_utils.weighted_average_polars(
                "PredictorPerformance", "ResponseCountBin"
            )
        )

        if return_df:
            return df
        y = "PredictorCategory"
        order = df.collect().get_column("PredictorCategory").unique().to_list()
        order.sort()
        fig = px.box(
            df.sort(y).collect(),
            x=metric,
            y=y,
            color="PredictorCategory",
            template="pega",
            facet_col=facet_col,
            # title=f"{title_prefix} {title_suffix}",
            facet_col_wrap=3,
            labels={
                "PredictorName": "Predictor Name",
                "PredictorPerformance": "Performance",
            },
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=order, automargin=True, dtick=1
        )

        fig.update_layout(
            boxgap=0, boxgroupgap=0, legend_title_text="Predictor category"
        )
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
        by: Union[QUERY, str] = "Name",
        active_only: bool = False,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        df = self.datamart.aggregates.predictor_performance_pivot(
            query=query,
            by=by,
            top_predictors=top_predictors,
            top_groups=top_groups,
            active_only=active_only,
        )

        if return_df:
            return df

        title = "over all models"
        plot_df = df.collect().to_pandas(use_pyarrow_extension_array=True)
        plot_df.set_index(plot_df.columns[0], inplace=True)
        fig = px.imshow(
            plot_df.T,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale=self.datamart.cdh_guidelines.colorscales.get(
                "Performance"
            ),
            title=f"Top predictors {title}",
            range_color=[0.5, 1],
        )

        fig.update_yaxes(dtick=1, automargin=True)
        fig.update_xaxes(
            dtick=1,
        )
        return fig

    def response_gain(): ...  # TODO: more generic plot_gains function?

    # TODO anyone using this still?
    # consider removing
    def models_by_positives(
        self,
        by: str = "Channel",
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):  # TODO: more generic plot gains function?
        df = (
            cdh_utils._apply_query(self.datamart.aggregates.last(), query=query)
            .select([by, "Positives", "ModelID"])
            .collect()
        )
        models_by_positives = (
            df.with_columns(
                PositivesBin=df["Positives"].cut(
                    breaks=list(range(0, 210, 10)),
                )
            )
            .lazy()
            .group_by([by, "PositivesBin"])
            .agg([pl.min("Positives"), pl.n_unique("ModelID").alias("ModelCount")])
            .with_columns(
                (pl.col("ModelCount") / (pl.sum("ModelCount").over(by))).alias(
                    "cumModels"
                )
            )
            .sort("Positives")
        ).collect()
        if return_df:
            return models_by_positives

        title = "Positives vs Number of Models"
        fig = px.line(
            models_by_positives.filter(pl.col("ModelCount") > 0)
            .with_columns(pl.col(by).fill_null("NA"))
            .to_pandas(use_pyarrow_extension_array=True),
            x="PositivesBin",
            y="cumModels",
            color=by,
            markers=True,
            title=title,
            labels={"cumModels": "Percentage of Models", "PositivesBin": "Positives"},
            template="pega",
            category_orders={
                "PositivesBin": models_by_positives.select("Positives", "PositivesBin")
                .unique()
                .sort("Positives")["PositivesBin"]
                .to_list()
            },
        )
        fig.update_layout(
            xaxis_title="Number of Positives",
            yaxis_title="Percentage of Models",
        )

        return fig

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

        colorscale = self.datamart.cdh_guidelines.colorscales.get(metric, None) or [
            "#d91c29",
            "#F76923",
            "#20aa50",
        ]

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
            df.collect().to_pandas(use_pyarrow_extension_array=True),
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
        by: str = "Type",
        facet: str = "Configuration",
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        df = self.datamart.aggregates.predictor_counts(by=by, facet=facet, query=query)
        if return_df:
            return df

        return px.box(
            df.collect().to_pandas(use_pyarrow_extension_array=True),
            x="PredictorCount",
            y="Type",
            facet_col=facet,
            facet_col_wrap=2,
            color="EntryType",
            template="pega",
            title="Predictor Count across all models",
        )

    def binning_lift(
        self,
        model_id: str,
        predictor_name: str,
        *,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
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
            plot_df.collect(), #.to_pandas(use_pyarrow_extension_array=False),
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

    def partitioned_plot(
        self,
        func: Callable,
        facets: Set,
        partition_col: str = "Configuration",
        show_plots: bool = True,
        *args,
        **kwargs,
    ):
        existing_query = kwargs.get("query")
        figs = []
        for facet in facets:
            new_query = pl.col(partition_col).eq(facet)
            if existing_query is not None:
                combined_query = cdh_utils._combine_queries(existing_query, new_query)
            else:
                combined_query = new_query
            kwargs["query"] = combined_query
            fig = func(*args, **kwargs)
            figs.append(fig)
            if show_plots and fig is not None:
                fig.show()
        return figs


    # TODO I took the propensity distrib plot out of the HC as 
    # it wasn't very clear, also didn't look great visually.

    @requires(
        predictor_columns={
            "BinPropensity",
        },
        combined_columns={"Channel", "Direction"},
    )
    def propensity_distribution(
        self,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ):
        to_plot = "BinPropensity"
        df = cdh_utils._apply_query(
            (
                self.datamart.combined_data.filter(
                    pl.col("PredictorName") != "Classifier"
                )
                .group_by([to_plot, "Channel", "Direction"])
                .agg(pl.sum("BinResponseCount"))
                .with_columns(pl.col(to_plot).round(4).cast(pl.Float64))
                .collect()
            ),
            query,
        )

        custom_bins = [
            0,
            0.0005,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            0.8,
            1.0,
        ]
        df_pl = df.with_columns(
            pl.col(to_plot)
            .fill_null(0)
            .fill_nan(0)
            .cut(breaks=custom_bins)
            .alias(f"{to_plot}_range")
        )

        out = (
            df_pl.group_by(["Channel", f"{to_plot}_range"])
            .agg([pl.sum("BinResponseCount"), pl.min(to_plot).alias("break_label")])
            .sort(["Channel", "break_label"])
            .with_columns(
                [
                    (
                        pl.col("BinResponseCount")
                        / pl.col("BinResponseCount").sum().over("Channel")
                    ).alias("Responses")
                ]
            )
        )
        if return_df:
            return df.lazy()
        fig = px.bar(
            out.to_pandas(use_pyarrow_extension_array=True),
            x=f"{to_plot}_range",
            y="Responses",
            color="Channel",
            template="pega",
            barmode="overlay",
        )
        fig.update_xaxes(type="category")
        fig.update_yaxes(tickformat=",.0%")

        return fig
