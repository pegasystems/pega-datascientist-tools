"""Plotting utilities for Impact Analyzer visualization."""

from __future__ import annotations

import logging
from typing import ClassVar, Literal, SupportsFloat, TYPE_CHECKING, cast, overload

import polars as pl

from ..utils.cdh_utils import _apply_query
from ..utils.namespaces import LazyNamespace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..utils.types import QUERY
    from ..utils.plot_utils import Figure
    from .ImpactAnalyzer import ImpactAnalyzer as ImpactAnalyzer_Class


class Plots(LazyNamespace):
    """Visualization methods for Impact Analyzer experiment data.

    This class provides plotting capabilities for analyzing Impact Analyzer
    experiment results. It is accessed through the `plot` attribute of an
    :class:`~pdstools.impactanalyzer.ImpactAnalyzer.ImpactAnalyzer` instance.

    All plot methods support:

    - Custom titles via `title` parameter
    - Data filtering via `query` parameter
    - Faceting by dimension via `facet` parameter
    - Returning underlying data via `return_df=True`

    See Also
    --------
    pdstools.impactanalyzer.ImpactAnalyzer : Main analysis class.

    Examples
    --------
    >>> ia = ImpactAnalyzer.from_pdc("export.json")
    >>> ia.plot.overview()
    >>> ia.plot.trend(metric="Value_Lift", facet="Channel")

    """

    dependencies: ClassVar[list[str]] = ["plotly"]
    dependency_group = "adm"
    ia: "ImpactAnalyzer_Class"
    """Reference to the parent ImpactAnalyzer instance."""

    def __init__(self, ia: "ImpactAnalyzer_Class"):
        """Initialize a Plots instance.

        Parameters
        ----------
        ia : ImpactAnalyzer
            The parent ImpactAnalyzer instance providing the data.

        """
        super().__init__()
        self.ia = ia

    @staticmethod
    def _get_experiment_color_map() -> list[str]:
        """Get the canonical product-order list of experiment names.

        Used as the ``category_orders`` for plotly so that experiments
        always render in the same order shown by the Pega Infinity
        Impact Analyzer UI, and so colour assignment stays consistent
        across plots.

        Returns
        -------
        list[str]
            Experiment names in canonical product order.

        """
        # Local import avoids a circular import at module load time.
        from pdstools.impactanalyzer.ImpactAnalyzer import ImpactAnalyzer

        return list(ImpactAnalyzer.default_ia_experiments.keys())

    @staticmethod
    def _get_facet_config(data: pl.DataFrame, facet: str | None) -> dict:
        """Determine optimal faceting configuration.

        Automatically selects column wrapping based on the number of
        distinct facet values for better layout.

        Parameters
        ----------
        data : pl.DataFrame
            The collected plot data.
        facet : str or None
            Column name for faceting, or None for no faceting.

        Returns
        -------
        dict
            Configuration with keys 'facet_col' and 'facet_col_wrap'.

        """
        if facet is None:
            return {"facet_col": None, "facet_col_wrap": None}

        if facet not in data.columns:
            return {"facet_col": facet, "facet_col_wrap": None}

        distinct_values = data[facet].n_unique()

        if distinct_values <= 3:
            return {"facet_col": facet, "facet_col_wrap": None}
        if distinct_values == 4:
            return {"facet_col": facet, "facet_col_wrap": 2}
        return {"facet_col": facet, "facet_col_wrap": 3}

    @overload
    def overview(
        self,
        *,
        title: str | None = ...,
        query: QUERY | None = ...,
        metric: str = ...,
        facet: str | None = ...,
        return_df: Literal[False] = ...,
    ) -> Figure: ...

    @overload
    def overview(
        self,
        *,
        title: str | None = ...,
        query: QUERY | None = ...,
        metric: str = ...,
        facet: str | None = ...,
        return_df: Literal[True],
    ) -> pl.LazyFrame: ...

    def overview(
        self,
        *,
        title: str | None = None,
        query: QUERY | None = None,
        metric: str = "CTR_Lift",
        facet: str | None = None,
        return_df: bool = False,
    ) -> Figure | pl.LazyFrame:
        """Create a bar chart comparing experiment performance.

        Displays a horizontal bar chart comparing Impact Analyzer experiments
        for the specified lift metric.

        Parameters
        ----------
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        metric : str, default "CTR_Lift"
            Metric to display: "CTR_Lift" or "Value_Lift".
        facet : str, optional
            Column name for faceting (e.g., "Channel").
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly figure, or LazyFrame if `return_df=True`.

        See Also
        --------
        trend : Time series of experiment metrics.
        control_groups_trend : Time series of control group metrics.

        Examples
        --------
        >>> ia.plot.overview()
        >>> ia.plot.overview(metric="Value_Lift", facet="Channel")
        >>> df = ia.plot.overview(return_df=True)

        """
        grouping_columns = [facet] if facet is not None else []
        plot_data = self.ia.summarize_experiments(by=grouping_columns)

        # Filter out rows with invalid metric values (NaN, Infinity)
        plot_data = plot_data.filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite(),
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        import plotly.express as px

        collected_data = plot_data.collect()
        facet_config = self._get_facet_config(collected_data, facet)

        fig = px.bar(
            collected_data,
            y="Experiment",
            x=metric,
            color="Experiment",
            facet_col=facet_config["facet_col"],
            facet_col_wrap=facet_config["facet_col_wrap"],
            facet_col_spacing=0.1,
            facet_row_spacing=0.25,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            category_orders={"Experiment": self._get_experiment_color_map()},
            template="pega",
        )

        fig.update_layout(
            showlegend=False,
            title=title or f"Overview Impact Analyzer Experiments for {metric}",
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(automargin=True, title="")
        fig.update_xaxes(
            title="",
            tickformat=".1%",
            matches=None if facet is not None else "x",
            showticklabels=True,
        )

        return fig

    @overload
    def control_groups_trend(
        self,
        *,
        title: str | None = ...,
        query: QUERY | None = ...,
        metric: str = ...,
        facet: str | None = ...,
        every: str | None = ...,
        return_df: Literal[False] = ...,
    ) -> Figure: ...

    @overload
    def control_groups_trend(
        self,
        *,
        title: str | None = ...,
        query: QUERY | None = ...,
        metric: str = ...,
        facet: str | None = ...,
        every: str | None = ...,
        return_df: Literal[True],
    ) -> pl.LazyFrame: ...

    def control_groups_trend(
        self,
        *,
        title: str | None = None,
        query: QUERY | None = None,
        metric: str = "CTR",
        facet: str | None = None,
        every: str | None = None,
        return_df: bool = False,
    ) -> Figure | pl.LazyFrame:
        """Create a line chart of control group metrics over time.

        Displays how different control groups perform over time for
        the specified metric.

        Parameters
        ----------
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        metric : str, default "CTR"
            Metric to display: "CTR" or "ValuePerImpression".
        facet : str, optional
            Column name for faceting (e.g., "Channel").
        every : str, optional
            Time aggregation period using Polars syntax:
            "1d" (daily), "1w" (weekly), "1mo" (monthly), etc.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly figure, or LazyFrame if `return_df=True`.

        See Also
        --------
        trend : Time series of experiment lift metrics.
        overview : Bar chart of experiment metrics.

        Examples
        --------
        >>> ia.plot.control_groups_trend()
        >>> ia.plot.control_groups_trend(metric="ValuePerImpression", every="1w")
        >>> ia.plot.control_groups_trend(facet="Channel", every="1mo")

        """
        grouping_columns: list[str | pl.Expr]
        if every is not None:
            grouping_columns = [pl.col("SnapshotTime").dt.truncate(every)]
            if facet is not None:
                grouping_columns.append(facet)
        else:
            grouping_columns = ["SnapshotTime", *([facet] if facet is not None else [])]

        plot_data = self.ia.summarize_control_groups(by=grouping_columns).filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite(),
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        import plotly.express as px

        collected_data = plot_data.collect()
        facet_config = self._get_facet_config(collected_data, facet)

        # Stable colour map keyed on ControlGroup so the same group
        # always renders in the same colour across plots / refreshes.
        # Categories listed in default_ia_controlgroups["MktValue"]
        # cover the standard six PDC groups; everything else falls
        # through to the default qualitative palette.
        from .ImpactAnalyzer import ImpactAnalyzer as _IA

        _palette = px.colors.qualitative.Plotly
        _control_groups = [
            cg.removeprefix("NBAHealth_") for cg in _IA.default_ia_controlgroups["MktValue"] if cg is not None
        ]
        color_map = {cg: _palette[i % len(_palette)] for i, cg in enumerate(_control_groups)}
        # NBA is the highlighted "production" arm — give it a clear,
        # neutral-but-strong colour that doesn't clash with the
        # qualitative palette.
        color_map["NBA"] = "#1B2A4A"

        fig = px.line(
            collected_data,
            y=metric,
            x="SnapshotTime",
            color="ControlGroup",
            facet_col=facet_config["facet_col"],
            facet_col_wrap=facet_config["facet_col_wrap"],
            facet_col_spacing=0.1,
            facet_row_spacing=0.15,
            color_discrete_map=color_map,
            category_orders={"ControlGroup": [*[cg for cg in _control_groups if cg != "NBA"], "NBA"]},
            template="pega",
        )

        # Make the NBA arm visually dominant: thicker line and bold
        # legend entry. Plotly doesn't expose per-trace legend styling
        # directly, so we wrap the legend label in <b>…</b> markup
        # (Plotly renders legend text as HTML).
        for trace in fig.data:
            if getattr(trace, "name", None) == "NBA":
                trace.update(line=dict(width=4), name="<b>NBA</b>", legendgroup="<b>NBA</b>")
            else:
                trace.update(line=dict(width=1.75))

        fig.update_layout(
            hovermode="closest",
            title=title or f"Trend of {metric} for Impact Analyzer Control Groups",
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(
            title="",
            tickformat=".1%",
            matches=None if facet is not None else "y",
            showticklabels=True,
            rangemode="tozero",
        )
        fig.update_xaxes(title="")

        return fig

    @overload
    def trend(
        self,
        *,
        title: str | None = ...,
        query: QUERY | None = ...,
        metric: str = ...,
        facet: str | None = ...,
        every: str | None = ...,
        return_df: Literal[False] = ...,
    ) -> Figure: ...

    @overload
    def trend(
        self,
        *,
        title: str | None = ...,
        query: QUERY | None = ...,
        metric: str = ...,
        facet: str | None = ...,
        every: str | None = ...,
        return_df: Literal[True],
    ) -> pl.LazyFrame: ...

    def trend(
        self,
        *,
        title: str | None = None,
        query: QUERY | None = None,
        metric: str = "CTR_Lift",
        facet: str | None = None,
        every: str | None = None,
        return_df: bool = False,
    ) -> Figure | pl.LazyFrame:
        """Create a line chart of experiment lift metrics over time.

        Displays how different experiments' lift metrics evolve over time.

        Parameters
        ----------
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        metric : str, default "CTR_Lift"
            Metric to display: "CTR_Lift" or "Value_Lift".
        facet : str, optional
            Column name for faceting (e.g., "Channel").
        every : str, optional
            Time aggregation period using Polars syntax:
            "1d" (daily), "1w" (weekly), "1mo" (monthly), etc.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly figure, or LazyFrame if `return_df=True`.

        See Also
        --------
        overview : Bar chart of experiment metrics.
        control_groups_trend : Time series of control group metrics.

        Examples
        --------
        >>> ia.plot.trend()
        >>> ia.plot.trend(metric="Value_Lift", every="1w")
        >>> ia.plot.trend(facet="Channel", every="1mo")

        """
        grouping_columns: list[str | pl.Expr]
        if every is not None:
            grouping_columns = [pl.col("SnapshotTime").dt.truncate(every)]
            if facet is not None:
                grouping_columns.append(facet)
        else:
            grouping_columns = ["SnapshotTime", *([facet] if facet is not None else [])]

        plot_data = self.ia.summarize_experiments(by=grouping_columns).filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite(),
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        import plotly.express as px

        collected_data = plot_data.collect()
        facet_config = self._get_facet_config(collected_data, facet)

        fig = px.line(
            collected_data,
            y=metric,
            x="SnapshotTime",
            color="Experiment",
            facet_col=facet_config["facet_col"],
            facet_col_wrap=facet_config["facet_col_wrap"],
            facet_col_spacing=0.1,
            facet_row_spacing=0.15,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            category_orders={"Experiment": self._get_experiment_color_map()},
            template="pega",
        )

        fig.update_layout(
            hovermode="x unified",
            title=title or f"Trend of {metric} for Impact Analyzer Experiments",
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(
            title="",
            tickformat=".1%",
            matches=None if facet is not None else "y",
            showticklabels=True,
        )
        fig.update_xaxes(title="")

        return fig

    @overload
    def control_fraction_heatmap(
        self,
        *,
        title: str | None = ...,
        return_df: Literal[False] = ...,
    ) -> Figure: ...

    @overload
    def control_fraction_heatmap(
        self,
        *,
        title: str | None = ...,
        return_df: Literal[True],
    ) -> pl.LazyFrame: ...

    def control_fraction_heatmap(
        self,
        *,
        title: str | None = None,
        return_df: bool = False,
    ) -> Figure | pl.LazyFrame:
        """Heatmap of control fraction by Channel x Experiment.

        Surfaces mismatches in how control percentages are configured in
        the Pega product. Each cell shows the fraction of impressions
        assigned to the control group for that (Channel, Experiment)
        pair; rows that vary noticeably across channels usually point
        at a misconfigured experiment in one of the channels.

        Parameters
        ----------
        title : str, optional
            Custom title. If None, a default is used.
        return_df : bool, default False
            If True, return the underlying long-form data (one row per
            non-null Channel x Experiment cell) as a LazyFrame instead
            of the figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly heatmap, or the underlying LazyFrame when
            ``return_df=True``.

        """
        plot_data = (
            self.ia.summarize_experiments(by="Channel")
            .filter(pl.col("Control_Fraction").is_not_null())
            .select("Channel", "Experiment", "Control_Fraction")
        )

        if return_df:
            return plot_data

        import plotly.express as px

        df = plot_data.collect()
        if df.is_empty():
            fig = px.imshow([[0]], template="pega")
            fig.update_layout(title=title or "Control Fraction by Channel x Experiment")
            return fig

        wide = df.pivot(
            on="Experiment",
            index="Channel",
            values="Control_Fraction",
        ).sort("Channel")

        experiment_order = [e for e in self._get_experiment_color_map() if e in wide.columns]
        wide = wide.select(["Channel", *experiment_order])

        z = wide.drop("Channel").to_numpy()
        x_labels = experiment_order
        y_labels = wide["Channel"].to_list()

        max_cf_value = df["Control_Fraction"].max()
        max_cf = 0.0 if max_cf_value is None else float(cast(SupportsFloat, max_cf_value))
        fig = px.imshow(
            z,
            x=x_labels,
            y=y_labels,
            color_continuous_scale="Blues",
            zmin=0,
            zmax=max(0.5, max_cf),
            text_auto=".1%",
            aspect="auto",
            template="pega",
        )
        fig.update_layout(
            title=title or "Control Fraction by Channel x Experiment",
            xaxis_title="",
            yaxis_title="",
            coloraxis_colorbar=dict(title="Control %", tickformat=".0%"),
            margin=dict(l=60, r=60, t=80, b=60),
        )
        fig.update_xaxes(side="top", tickangle=-30)
        fig.update_traces(
            hovertemplate="Channel: %{y}<br>Experiment: %{x}<br>Control: %{z:.2%}<extra></extra>",
        )

        return fig
