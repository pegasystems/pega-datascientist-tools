import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import polars as pl
from typing_extensions import ParamSpec

from ..utils.cdh_utils import _apply_query, lazy_sample
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

logger = logging.getLogger(__name__)
try:
    import plotly.express as px  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    from ..utils import pega_template as pega_template
except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

    from .ValueFinder import ValueFinder

COLORSCALE_TYPES = Union[List[Tuple[float, str]], List[str]]

Figure = Union[Any, "go.Figure"]

T = TypeVar("T", bound="Plots")
P = ParamSpec("P")


class Plots(LazyNamespace):
    dependencies = ["plotly"]

    def __init__(self, vf: "ValueFinder"):
        self.vf = vf
        super().__init__()

    @overload
    def funnel_chart(
        self,
        by: str,
        query: Optional[QUERY] = None,
        return_df: Literal[False] = False,
    ) -> Figure: ...

    @overload
    def funnel_chart(
        self,
        by: str,
        query: Optional[QUERY] = None,
        return_df: Literal[True] = True,
    ) -> pl.LazyFrame: ...

    def funnel_chart(
        self, by: str = "Group", query: Optional[QUERY] = None, return_df: bool = False
    ):
        df = _apply_query(self.vf.df, query)
        df = (
            df.group_by("Stage")
            .agg(pl.col(by).value_counts(sort=True, name="Count"))
            .explode(by)
            .unnest(by)
            .sort("Stage")
        )
        if return_df:
            return df

        fig = px.funnel(
            df.with_columns(pl.col(pl.Categorical).cast(pl.Utf8)).collect().to_pandas(),
            y="Count",
            x="Stage",
            color=by,
            text=by,
            title=f"Distribution of {by if by != 'Name' else 'Action'}s over the stages",
            template="none",
        )

        fig.update_xaxes(categoryorder="array", categoryarray=self.vf.nbad_stages)
        fig.update_layout(legend_title_text=by)
        return fig

    def propensity_distribution(self, sample_size: int = 10_000) -> Figure:
        import plotly.figure_factory as ff  # type: ignore[import-untyped]

        i = 0
        figs = make_subplots(
            rows=len(self.vf.nbad_stages),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.005,
        )
        for stage in self.vf.nbad_stages:
            data = self.vf.df.filter(pl.col("Stage") == stage)
            cols = ["ModelPropensity"]
            sample = lazy_sample(data, sample_size).select(cols).drop_nulls().collect()
            temp = ff.create_distplot(
                [sample["ModelPropensity"].to_list()],
                ["ModelPropensity"],
                show_rug=False,
                show_hist=False,
            )
            tempdf = pl.DataFrame(
                {"x": temp["data"][0]["x"], "y": temp["data"][0]["y"]}
            )
            fig = go.Scatter(
                x=tempdf["x"],
                y=tempdf["y"],
                name=stage,
                showlegend=False,
                line_color="rgba(0,0,0)",
            )
            boxy = go.Box(
                x=sample["ModelPropensity"], name=stage, y0=0, boxpoints="outliers"
            )
            figs.add_trace(fig, row=i + 1, col=1)
            figs.add_trace(boxy, row=i + 1, col=1)
            figs.update_yaxes(title_text="Density", row=i + 1, col=1)
            i += 1

        figs.update_xaxes(title_text="ModelPropensity", row=4, col=1)
        figs.update_layout(
            title_text="Propensity Distribution<br><sup>per NBAD stage</sup>",
            legend_title_text="Stage",
            height=800,
        )
        return figs

    def propensity_threshold(
        self, sample_size: int = 10_000, stage="Eligibility"
    ) -> Figure:
        import plotly.figure_factory as ff

        colors = ["#001F5F", "#10A5AC", "#F76923"]
        propensities = ["FinalPropensity", "Propensity", "ModelPropensity"]
        i = 0
        figs = make_subplots(rows=len(propensities), cols=1, shared_xaxes=True)
        yrange = [0, 15]
        data = lazy_sample(
            self.vf.df.filter(pl.col("Stage") == stage).select(propensities),
            sample_size,
        ).collect()

        for ptype in propensities:
            plotdf = data[ptype].to_list()
            temp = ff.create_distplot(
                [plotdf], ["value"], show_rug=False, show_hist=False
            )
            tempdf = pl.DataFrame(
                {"x": temp["data"][0]["x"], "y": temp["data"][0]["y"]}
            )
            figs.add_trace(
                go.Scatter(
                    x=tempdf["x"],
                    y=tempdf["y"],
                    name=ptype,
                    showlegend=False,
                    line_color=colors[i],
                    hoverinfo="skip",
                ),
                row=i + 1,
                col=1,
            )
            figs.add_trace(
                go.Histogram(
                    x=plotdf,
                    name=ptype,
                    histnorm="probability density",
                    marker_color=colors[i],
                ),
                row=i + 1,
                col=1,
            )
            figs.update_yaxes(range=yrange, col=1, row=i + 1)
            figs.add_shape(
                type="line",
                x0=self.vf.threshold,
                x1=self.vf.threshold,
                y0=yrange[0],
                y1=yrange[1],
                line=dict(dash="dot"),
                col=1,
                row=i + 1,
            )
            i += 1
        figs.update_xaxes(
            title_text="Propensity", ticks="outside", showticklabels=True, row=3, col=1
        )
        figs.update_yaxes(title_text="density", row=2, col=1)
        figs.update_layout(
            title_text=f"Propensity Distribution <br><sup>{stage} stage</sup>",
            title_x=0.1,
            legend_title="Type",
            height=800,
        )
        return figs

    def _get_thresholds(
        self,
        thresholds: Optional[Iterable[float]] = None,
        quantiles: Optional[Iterable[float]] = None,
        default: Optional[Iterable[float]] = None,
    ) -> Iterable[float]:
        if thresholds is not None and quantiles is not None:
            raise ValueError("Please only supply thresholds OR quantiles, not both.")
        elif thresholds is None and quantiles is None:
            return default or [self.vf.threshold]
        elif thresholds is not None:
            return thresholds
        elif quantiles is not None:
            return map(self.vf.aggregates.get_threshold_from_quantile, quantiles)
        raise ValueError()  # pragma: no cover

    def pie_charts(
        self,
        *,
        thresholds: Optional[Iterable[float]] = None,
        quantiles: Optional[Iterable[float]] = None,
        rounding: int = 3,
    ):
        thresholds = self._get_thresholds(thresholds, quantiles)
        df = pl.concat(
            [
                self.vf.aggregates.get_counts_for_threshold(th).with_columns(
                    Threshold=pl.lit(th).round(rounding)
                )
                for th in thresholds
            ]
        )
        colors = ["#219e3f", "#fca52e", "#cd001f"]
        fig = make_subplots(
            rows=1,
            cols=len(self.vf.nbad_stages),
            specs=[[{"type": "domain"}] * len(self.vf.nbad_stages)],
            subplot_titles=self.vf.nbad_stages,
        )
        steps = []
        last_threshold = 0.0
        default_n: Optional[int] = None
        for i, ((stage, threshold), _df) in enumerate(
            df.group_by("Stage", "Threshold", maintain_order=True)
        ):
            stage, threshold = cast(str, stage), cast(float, threshold)
            n = i // len(self.vf.nbad_stages)
            default_n = n if threshold == self.vf.threshold else default_n
            visible = [False] * len(df)  # Initialize visibility for all traces

            # Set visibility for the current set of traces
            for j in range(len(self.vf.nbad_stages)):
                visible[n * len(self.vf.nbad_stages) + j] = True

            fig.add_trace(
                go.Pie(
                    values=_df.drop("Threshold", "Stage").row(0),
                    labels=list(
                        _df.rename(
                            {
                                "RelevantActions": "At least one relevant action",
                                "IrrelevantActions": "Only irrelevant actions",
                                "NoActions": "Without actions",
                            }
                        )
                        .drop("Threshold", "Stage")
                        .columns
                    ),
                    name=stage,
                    visible=default_n is not None and default_n == n,
                    sort=False,
                    marker=dict(colors=colors),
                ),
                1,
                self.vf.nbad_stages.index(stage) + 1,
            )
            # Create a slider step for each threshold
            if threshold != last_threshold:
                step = dict(
                    method="update",
                    label=str(round(threshold, rounding)),
                    args=[
                        {"visible": visible},
                        {
                            "title": f"Distribution of customers per stage at propensity threshold {round(threshold, rounding):.{rounding-2}%}"
                        },
                    ],
                )
                steps.append(step)
                last_threshold = threshold

        # Add sliders to the figure
        sliders = [
            dict(
                active=default_n or 0,
                currentvalue={"prefix": "Propensity threshold: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        fig.update_layout(
            sliders=sliders,
            title_text=f"Distribution of customers per stage at propensity threshold {round(self.vf.threshold, rounding):.1%}",
        )
        if not default_n:
            for i in range(0, 4):
                fig.data[i].visible = True

        for i in range(len(fig.layout.sliders[0].steps)):
            fig.layout.sliders[0].steps[
                i
            ].label = f"{float(fig.layout.sliders[0].steps[i].label):.{rounding-2}%}"

        return fig

    def distribution_per_threshold(
        self,
        *,
        thresholds: Optional[Iterable[float]] = None,
        quantiles: Optional[Iterable[float]] = None,
        rounding: int = 3,
    ):
        thresholds = self._get_thresholds(
            thresholds, quantiles, ((x + 1) * 0.05 for x in range(20))
        )

        df = [
            self.vf.aggregates.get_counts_for_threshold(th).with_columns(
                Threshold=pl.lit(th).round(rounding)
            )
            for th in thresholds
        ]

        plot_df = (
            pl.concat(df)
            .to_pandas()
            .set_index("Threshold")
            .rename(
                columns={
                    "RelevantActions": "At least one relevant action",
                    "IrrelevantActions": "Only irrelevant actions",
                    "NoActions": "Without actions",
                }
            )
        )
        fig = (
            px.area(
                plot_df,
                facet_col="Stage",
                category_orders={"Stage": self.vf.nbad_stages},
                labels={"value": "Number of people"},
                title="Distribution of offers per stage",
                template="pega",
            )
            .update_layout(legend_title_text="Status")
            .update_xaxes(tickformat=",.1%")
        )
        return fig
