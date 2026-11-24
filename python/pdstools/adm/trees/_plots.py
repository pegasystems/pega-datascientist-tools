"""Plot namespace for :class:`~pdstools.adm.trees.ADMTreesModel`.

All methods are accessible via ``trees_model.plot.<method>`` where the
method name is the former ``plot_<method>`` name with the ``plot_`` prefix
removed.

Examples
--------
>>> Trees.plot.tree(18, highlighted=x)
>>> Trees.plot.gain_per_tree()
>>> Trees.plot.feature_importance_by_gain(top_n=10)
"""

from __future__ import annotations

import logging
import unicodedata
from typing import TYPE_CHECKING, ClassVar, Literal, SupportsFloat, cast, overload

import polars as pl

from ...utils import cdh_utils
from ...utils.namespaces import LazyNamespace, MissingDependenciesException
from ...utils.plot_utils import abbreviate_label
from ..ADMDatamart import _STANDARD_PREDICTOR_CATEGORY_COLORS
from ._nodes import _iter_nodes

# Fixed color assignments for well-known predictor categories, shared with
# ADMDatamart predictor-category plots.
_PREDICTOR_CATEGORY_COLORMAP: dict[str, str] = dict(_STANDARD_PREDICTOR_CATEGORY_COLORS)

# Un-aliased value expression — import and extend this in notebooks to add
# custom predictor categories before calling plot methods.
# Example (in a notebook cell):
#   from pdstools.adm.trees._plots import _PREDICTOR_CATEGORY_VALUE
#   AGBModel.plot.predictor_category_expr = (
#       pl.when(pl.col("predictor").str.starts_with("MyPrefix"))
#       .then(pl.lit("External Model"))
#       .otherwise(_PREDICTOR_CATEGORY_VALUE)
#       .alias("PredictorCategory")
#   )
_PREDICTOR_CATEGORY_VALUE: pl.Expr = cdh_utils.default_predictor_categorization("predictor").meta.undo_aliases()

_PREDICTOR_CATEGORY_EXPR: pl.Expr = _PREDICTOR_CATEGORY_VALUE.alias("PredictorCategory")

# Backwards-compatible aliases for notebooks written before the AGB plots were
# aligned with ADM's PredictorCategory terminology.
_PREDICTOR_GROUP_COLORMAP = _PREDICTOR_CATEGORY_COLORMAP
_PREDICTOR_GROUP_VALUE = _PREDICTOR_CATEGORY_VALUE
_PREDICTOR_GROUP_EXPR = _PREDICTOR_CATEGORY_EXPR

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go
    import pydot

    from ._model import ADMTreesModel


class Plots(LazyNamespace):
    """Namespace exposing all AGB model plots.

    Consumed via ``trees_model.plot.<method>`` on an
    :class:`~pdstools.adm.trees.ADMTreesModel` instance.
    """

    dependencies: ClassVar[list[str]] = ["plotly"]
    dependency_group = "adm"

    def __init__(self, trees_model: "ADMTreesModel") -> None:
        self.trees_model = trees_model
        self.predictor_category_expr: pl.Expr = _PREDICTOR_CATEGORY_EXPR
        """Polars expression producing the ``"PredictorCategory"`` column.

        Set this attribute to a custom expression to override the default
        categorization (e.g., to map external-score predictors to their own
        category). See :data:`_PREDICTOR_CATEGORY_VALUE` for a composable base
        expression.
        """
        self.predictor_category_colormap: dict[str, str] = _PREDICTOR_CATEGORY_COLORMAP
        """Color map from predictor category name to hex color string.

        Extend this dict when you introduce a custom predictor category so the
        new category always gets a consistent color across all plots.
        """
        self.predictor_category_order: list[str] = list(_PREDICTOR_CATEGORY_COLORMAP)
        """Fixed display order of predictor categories in plot legends and axes.

        Defaults to the key order of :data:`_PREDICTOR_CATEGORY_COLORMAP`.
        Category-share bars instead sort independently by decreasing
        contribution, while keeping category colors consistent.
        """
        self.predictor_group_expr = self.predictor_category_expr
        self.predictor_group_colormap = self.predictor_category_colormap
        self.predictor_group_order = self.predictor_category_order
        super().__init__()

    # ------------------------------------------------------------------
    # Original plots
    # ------------------------------------------------------------------

    @overload
    def splits_per_variable(self, subset: set | None = ..., *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def splits_per_variable(self, subset: set | None = ..., *, return_df: Literal[False] = ...) -> list: ...

    def splits_per_variable(
        self,
        subset: set | None = None,
        *,
        return_df: bool = False,
    ) -> "list[go.Figure] | pl.DataFrame":
        """Box-plot of gains per split, grouped by predictor variable.

        Parameters
        ----------
        subset : set or None, default None
            If provided, only plot predictors whose names are in this set.
        return_df : bool, default False
            If True, return the underlying gains DataFrame instead of figures.

        Returns
        -------
        list[plotly.graph_objects.Figure] or pl.DataFrame
            List of figures when ``return_df=False``; DataFrame with columns
            ``split``, ``gains``, ``predictor`` when ``return_df=True``.
        """
        if return_df:
            base = self.trees_model.gains_per_split
            if subset is not None:
                base = base.filter(pl.col("predictor").is_in(subset))
            return base
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None
        figlist = []
        for (name,), data in self.trees_model.gains_per_split.group_by("predictor"):
            if subset is None or name in subset:
                split_values = data.get_column("split").unique(maintain_order=True).to_list()
                split_tick_text = [abbreviate_label(value, max_length=48) for value in split_values]
                fig = make_subplots()
                fig.add_trace(
                    go.Box(
                        x=data.get_column("split"),
                        y=data.get_column("gains"),
                        name="Gain",
                    ),
                )
                grouped = self.trees_model.grouped_gains_per_split.filter(pl.col("predictor") == name)
                fig.add_trace(
                    go.Scatter(
                        x=grouped.select("split").to_series().to_list(),
                        y=grouped.select("n").to_series().to_list(),
                        name="Number of splits",
                        mode="lines+markers",
                    ),
                )
                fig.update_layout(
                    template="none",
                    title=f"Splits on {name}",
                    xaxis_title="Split",
                    yaxis_title="Number",
                    xaxis_automargin=True,
                    margin=dict(b=120),
                )
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=split_values,
                    ticktext=split_tick_text,
                    tickangle=45,
                    automargin=True,
                )
                figlist.append(fig)
        return figlist

    @overload
    def tree(
        self,
        tree_number: int,
        highlighted: dict | list | None = ...,
        *,
        return_df: Literal[True],
    ) -> pl.DataFrame: ...

    @overload
    def tree(
        self,
        tree_number: int,
        highlighted: dict | list | None = ...,
        *,
        return_df: Literal[False] = ...,
    ) -> "pydot.Graph": ...

    def tree(
        self,
        tree_number: int,
        highlighted: dict | list | None = None,
        *,
        return_df: bool = False,
    ) -> "pydot.Graph | pl.DataFrame":
        """Visualise one decision tree or return its node structure as a DataFrame.

        Parameters
        ----------
        tree_number : int
            Index of the tree to visualise.
        highlighted : dict or list or None, default None
            If a dict, score it through the tree and highlight the visited
            nodes in green.  If a list, treat it directly as the set of
            node keys to highlight.
        return_df : bool, default False
            If True, return the node structure as a DataFrame instead of the
            ``pydot.Graph``.

        Returns
        -------
        pydot.Graph or pl.DataFrame
            The pydot graph when ``return_df=False``; DataFrame with columns
            ``node_id``, ``score``, ``split``, ``gain``, ``parent_node``,
            ``left_child``, ``right_child`` when ``return_df=True``.
        """
        if isinstance(highlighted, dict):
            highlighted = self.trees_model.get_visited_nodes(tree_number, highlighted)[0]
        else:  # pragma: no cover
            highlighted = highlighted or []
        nodes = self.trees_model.get_tree_representation(tree_number)
        if return_df:
            rows = [
                {
                    "node_id": k,
                    "score": node["score"],
                    "split": node.get("split"),
                    "gain": node.get("gain"),
                    "parent_node": node.get("parent_node"),
                    "left_child": node.get("left_child"),
                    "right_child": node.get("right_child"),
                }
                for k, node in nodes.items()
            ]
            return pl.DataFrame(
                rows,
                schema={
                    "node_id": pl.Int64,
                    "score": pl.Float64,
                    "split": pl.String,
                    "gain": pl.Float64,
                    "parent_node": pl.Int64,
                    "left_child": pl.Int64,
                    "right_child": pl.Int64,
                },
            )
        try:
            import pydot
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["pydot"], "AGB", deps_group="adm") from None
        from ._nodes import parse_split

        graph = pydot.Dot("my_graph", graph_type="graph", rankdir="BT")
        for key, node in nodes.items():
            color = "green" if key in highlighted else "white"
            label = f"Score: {node['score']}"
            if "split" in node:
                split_obj = parse_split(node["split"])
                if split_obj.operator == "in" and isinstance(split_obj.value, tuple):
                    members = split_obj.value
                    if len(members) <= 3:
                        members_label: str = str(set(members))
                    else:
                        totallen = len(self.trees_model.all_values_per_split[split_obj.variable])
                        members_label = f"{[*list(members[:2]), '...']} ({len(members)}/{totallen})"
                    label += f"\nSplit: {split_obj.variable} in {members_label}\nGain: {node['gain']}"
                else:
                    label += f"\nSplit: {node['split']}\nGain: {node['gain']}"
                # Normalize to ASCII so pydot can write the DOT file on
                # any platform regardless of system encoding (e.g. cp1252
                # on Windows can't encode characters like Ś or °).
                safe_label = unicodedata.normalize("NFD", label).encode("ascii", "ignore").decode("ascii")
                graph.add_node(
                    pydot.Node(
                        name=str(key),
                        label=safe_label,
                        shape="box",
                        style="filled",
                        fillcolor=color,
                    )
                )
            else:
                graph.add_node(
                    pydot.Node(
                        name=str(key),
                        label=label,
                        shape="ellipse",
                        style="filled",
                        fillcolor=color,
                    )
                )
            if "parent_node" in node:
                graph.add_edge(pydot.Edge(str(key), str(node["parent_node"])))

        try:
            from IPython.display import Image, display

            create_png = getattr(graph, "create_png")
            display(Image(create_png()))
        except ImportError:
            pass
        except FileNotFoundError as exc:
            logger.exception(
                "Dot/Graphviz not installed; please install it on your machine: %s",
                exc,
            )
        return graph

    @overload
    def contribution_per_tree(self, x: dict, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def contribution_per_tree(self, x: dict, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def contribution_per_tree(
        self,
        x: dict,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot the per-tree contribution toward the final propensity.

        Parameters
        ----------
        x : dict
            Feature values for a single subject, mapping predictor name to
            its observed value.
        return_df : bool, default False
            If True, return the underlying scores DataFrame instead of the
            figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``treeID``, ``score``, ``scoresum``, ``mean``, ``propensity``
            when ``return_df=True``.
        """
        scores = (
            self.trees_model.get_all_visited_nodes(x)
            .sort("treeID")
            .with_row_index("row_idx")
            .with_columns(
                [
                    pl.col("score").cum_sum().alias("scoresum"),
                    (pl.col("score").cum_sum() / (pl.col("row_idx") + 1)).alias("mean"),
                    (1 / (1 + (-pl.col("score").cum_sum()).exp())).alias("propensity"),
                ],
            )
        )
        if return_df:
            return scores.select(["treeID", "score", "scoresum", "mean", "propensity"])
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None
        fig = px.scatter(
            scores,
            x="row_idx",
            y="score",
            template="none",
            title="Score contribution per tree, for single prediction",
            labels={"row_idx": "Tree", "score": "Score"},
        )
        fig["data"][0]["showlegend"] = True
        fig["data"][0]["name"] = "Individual scores"
        fig.add_trace(go.Scatter(x=scores["row_idx"], y=scores["mean"], name="Cumulative mean"))
        fig.add_trace(go.Scatter(x=scores["row_idx"], y=scores["propensity"], name="Propensity"))
        fig.add_trace(
            go.Scatter(
                x=[scores["row_idx"][-1]],
                y=[scores["propensity"][-1]],
                text=[scores["propensity"][-1]],
                mode="markers+text",
                textposition="top right",
                name="Final propensity",
            ),
        )
        fig.update_xaxes(zeroline=False)
        return fig

    # ------------------------------------------------------------------
    # Ensemble / feature-importance plots
    # ------------------------------------------------------------------

    @overload
    def gain_per_tree(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def gain_per_tree(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def gain_per_tree(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot total gain per tree as bars with a secondary score line.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``treeID``, ``total_gain``, ``score`` when ``return_df=True``.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        df = (
            self.trees_model.tree_stats.select(["treeID", "gains", "score"])
            .with_columns(
                pl.col("gains").list.sum().alias("total_gain"),
            )
            .select(["treeID", "total_gain", "score"])
        )
        if return_df:
            return df

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=df["treeID"].to_list(),
                y=df["total_gain"].to_list(),
                name="Total gain",
                marker_color="steelblue",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["treeID"].to_list(),
                y=df["score"].to_list(),
                name="Root score",
                mode="lines",
                line={"color": "firebrick"},
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Total gain and root score per tree",
            xaxis_title="Tree index",
            template="none",
            margin=dict(r=100),
        )
        fig.update_yaxes(title_text="Total gain", secondary_y=False)
        fig.update_yaxes(title_text="Root score", secondary_y=True)
        return fig

    @overload
    def cumulative_gain_share(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def cumulative_gain_share(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def cumulative_gain_share(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot the share of total gain accumulated as trees are added.

        Shares are unavailable when the total gain is not positive or any
        tree's total gain is negative.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``treeID``, ``cumulative_gain_share`` when ``return_df=True``.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        base = (
            self.trees_model.tree_stats.select(["treeID", "gains"])
            .with_columns(pl.col("gains").list.sum().alias("total_gain"))
            .select(["treeID", "total_gain"])
        )
        grand_total = base["total_gain"].sum()
        valid_share = grand_total > 0 and (base["total_gain"] >= 0).all()
        share_expr = pl.col("total_gain").cum_sum() / grand_total if valid_share else pl.lit(None, dtype=pl.Float64)
        df = base.with_columns(share_expr.alias("cumulative_gain_share")).select(["treeID", "cumulative_gain_share"])
        if return_df:
            return df

        half_idx = (
            df.filter(pl.col("cumulative_gain_share") >= 0.5)["treeID"].first()
            if valid_share and (df["cumulative_gain_share"] >= 0.5).any()
            else None
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["treeID"].to_list(),
                y=df["cumulative_gain_share"].to_list(),
                mode="lines",
                name="Cumulative gain share",
                line={"color": "steelblue"},
            )
        )
        if half_idx is not None:
            tree_marker = float(cast(SupportsFloat, half_idx))
            fig.add_vline(
                x=half_idx,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"50 % @ tree {tree_marker:g}",
                annotation_position="top right",
            )
        if not valid_share:
            fig.add_annotation(
                text="Gain share undefined: total gain is not positive<br>or some trees have negative total gain",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        fig.update_layout(
            title="Cumulative gain share across trees",
            xaxis_title="Tree index",
            yaxis_title="Cumulative gain share",
            yaxis_tickformat=".0%",
            template="none",
            margin=dict(l=90),
        )
        return fig

    @overload
    def feature_importance_by_gain(
        self,
        top_n: int = ...,
        *,
        return_df: Literal[True],
    ) -> pl.DataFrame: ...

    @overload
    def feature_importance_by_gain(
        self,
        top_n: int = ...,
        *,
        return_df: Literal[False] = ...,
    ) -> "go.Figure": ...

    def feature_importance_by_gain(
        self,
        top_n: int = 15,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot top predictors ranked by total information gain.

        Parameters
        ----------
        top_n : int, default 15
            Number of top predictors to display.
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``predictor``, ``total_gain``, ``PredictorCategory`` when
            ``return_df=True``.
        """
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        df = (
            self.trees_model.gains_per_split.group_by("predictor")
            .agg(pl.col("gains").sum().alias("total_gain"))
            .sort("total_gain", descending=True)
            .head(top_n)
            .with_columns(self.predictor_category_expr)
        )
        if return_df:
            return df

        sorted_df = df.sort("total_gain")
        return px.bar(
            sorted_df,
            x="total_gain",
            y="predictor",
            color="PredictorCategory",
            orientation="h",
            title=f"Top {top_n} predictors by total gain",
            labels={"total_gain": "Total gain", "predictor": "", "PredictorCategory": "Predictor category"},
            template="none",
            category_orders={
                "predictor": sorted_df["predictor"].to_list(),
                "PredictorCategory": self.predictor_category_order,
            },
            color_discrete_map=self.predictor_category_colormap,
        ).update_layout(yaxis_automargin=True, yaxis_autorange="reversed")

    @overload
    def early_vs_late_gain(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def early_vs_late_gain(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def early_vs_late_gain(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Compare each predictor's gain in early vs late trees.

        Predictors above the diagonal are "late refiners"; predictors below
        are "early specialists". Gains are signed. Axes become linear when
        either bucket contains non-positive values; bubbles use a uniform
        size when total gains are non-positive.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``predictor``, ``early_gain``, ``late_gain``, ``total_gain``,
            ``PredictorCategory`` when ``return_df=True``.
        """
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        n_trees = len(self.trees_model.model)
        quarter = max(1, n_trees // 4)
        early_ids = set(range(quarter))
        late_ids = set(range(n_trees - quarter, n_trees))

        rows: list[dict] = []
        for tree_id, tree in enumerate(self.trees_model.model):
            bucket: str | None = None
            if tree_id in early_ids:
                bucket = "early"
            elif tree_id in late_ids:
                bucket = "late"
            else:
                continue
            for node in _iter_nodes(tree):
                if node.split is not None:
                    rows.append({"predictor": node.split.variable, "gain": node.gain, "bucket": bucket})

        raw = pl.DataFrame(
            rows,
            schema={"predictor": pl.String, "gain": pl.Float64, "bucket": pl.String},
        )
        early = (
            raw.filter(pl.col("bucket") == "early").group_by("predictor").agg(pl.col("gain").sum().alias("early_gain"))
        )
        late = raw.filter(pl.col("bucket") == "late").group_by("predictor").agg(pl.col("gain").sum().alias("late_gain"))
        df = (
            early.join(late, on="predictor", how="full", coalesce=True)
            .with_columns(
                pl.col("early_gain").fill_null(0.0),
                pl.col("late_gain").fill_null(0.0),
            )
            .with_columns(
                (pl.col("early_gain") + pl.col("late_gain")).alias("total_gain"),
                self.predictor_category_expr,
            )
        )
        if return_df:
            return df

        if df.is_empty():
            import plotly.graph_objects as go

            return go.Figure().update_layout(title="Early vs late tree gain per predictor")

        min_val = min(cast(float, df["early_gain"].min()), cast(float, df["late_gain"].min()))
        max_val = max(cast(float, df["early_gain"].max()), cast(float, df["late_gain"].max()))
        log_axes = min_val > 0
        size = "total_gain" if (df["total_gain"] > 0).all() else None
        fig = px.scatter(
            df,
            x="early_gain",
            y="late_gain",
            color="PredictorCategory",
            size=size,
            hover_data={"total_gain": True},
            hover_name="predictor",
            log_x=log_axes,
            log_y=log_axes,
            title="Early vs late tree gain per predictor",
            labels={
                "early_gain": f"Gain in first {quarter} trees",
                "late_gain": f"Gain in last {quarter} trees",
                "PredictorCategory": "Predictor category",
            },
            template="none",
            category_orders={"PredictorCategory": self.predictor_category_order},
            color_discrete_map=self.predictor_category_colormap,
        )
        if not log_axes:
            fig.update_xaxes(type="linear")
            fig.update_yaxes(type="linear")
        fig.add_shape(
            type="line",
            x0=0.01 if log_axes else min(min_val, 0.0),
            y0=0.01 if log_axes else min(min_val, 0.0),
            x1=max(max_val, 0.0) or 1.0,
            y1=max(max_val, 0.0) or 1.0,
            line={"dash": "dash", "color": "grey"},
        )
        return fig

    @overload
    def gain_by_namespace(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def gain_by_namespace(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def gain_by_namespace(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot a 100% stacked horizontal bar of gain by predictor category.

        The predictor category follows ADM's default categorisation, with AGB
        model-context fields such as ``pyIssue``, ``pyGroup``, and
        ``pyTreatment`` called out separately. Dotted names use their first
        segment (for example, ``IH``, ``Customer``, ``Param``), while other
        undotted names are ``Primary``. Override
        ``AGBModel.plot.predictor_category_expr`` to define custom categories.
        Gain shares are unavailable when the total gain is not positive or
        any category's total gain is negative; the figure explains why
        instead of plotting percentages. Individual negative split gains
        are retained in the category totals. Hover shows the raw gain.
        Null categories are labelled ``Uncategorized`` so all gains are counted.
        Segments and legend entries are ordered by decreasing gain share,
        with category names breaking ties.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``PredictorCategory``, ``total_gain``, ``gain_share`` when
            ``return_df=True``.
        """
        df = (
            self.trees_model.gains_per_split.with_columns(self.predictor_category_expr)
            .with_columns(pl.col("PredictorCategory").fill_null("Uncategorized"))
            .group_by("PredictorCategory")
            .agg(pl.col("gains").sum().alias("total_gain"))
            .sort(["total_gain", "PredictorCategory"], descending=[True, False])
        )
        total = df["total_gain"].sum()
        valid_share = total > 0 and (df["total_gain"] >= 0).all()
        share_expr = pl.col("total_gain") / total if valid_share else pl.lit(None, dtype=pl.Float64)
        df = df.with_columns(share_expr.alias("gain_share"))
        if return_df:
            return df

        return self._category_share_plot(
            df,
            share_column="gain_share",
            total_column="total_gain",
            title="Share of total gain by predictor category",
            axis_title="Share of total gain",
            unavailable_text=(
                "Gain shares unavailable: total gain is not positive<br>"
                "or a predictor category has negative total gain.<br>"
                "Inspect signed totals with return_df=True."
            ),
        )

    @overload
    def splits_by_namespace(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def splits_by_namespace(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def splits_by_namespace(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot a 100% stacked horizontal bar of splits by predictor category.

        Count every internal split node across the exported ensemble,
        including repeated conditions and splits with zero or negative gain.
        These are split-node counts, not unique predictors, tree coverage,
        or training-response counts. Category definitions and colors match
        :meth:`gain_by_namespace`; segments and legend entries are ordered
        by decreasing split share, with category names breaking ties.
        Null categories are labelled ``Uncategorized`` in both plots so no
        splits are omitted.
        If the model has no splits, the figure shows an explanatory note.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``PredictorCategory``, ``split_count``, ``split_share`` when
            ``return_df=True``.
        """
        df = (
            self.trees_model.gains_per_split.with_columns(self.predictor_category_expr)
            .with_columns(pl.col("PredictorCategory").fill_null("Uncategorized"))
            .group_by("PredictorCategory")
            .agg(pl.len().alias("split_count"))
            .sort(["split_count", "PredictorCategory"], descending=[True, False])
        )
        total = df["split_count"].sum()
        share = pl.col("split_count") / total if total else pl.lit(None, dtype=pl.Float64)
        df = df.with_columns(share.alias("split_share"))
        if return_df:
            return df
        return self._category_share_plot(
            df,
            share_column="split_share",
            total_column="split_count",
            title="Share of splits by predictor category",
            axis_title="Share of split nodes",
            unavailable_text="Split shares unavailable: this model has no split nodes.",
        )

    def _category_share_plot(
        self,
        df: pl.DataFrame,
        *,
        share_column: str,
        total_column: str,
        title: str,
        axis_title: str,
        unavailable_text: str,
    ) -> "go.Figure":
        """Build a single normalized category bar, or explain unavailable shares."""
        import plotly.express as px
        import plotly.graph_objects as go

        if df.is_empty() or df[share_column].null_count():
            fig = go.Figure()
            fig.add_annotation(
                text=unavailable_text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        else:
            categories = df["PredictorCategory"].to_list()
            colors = dict(self.predictor_category_colormap)
            missing_colors = sorted(set(categories) - set(colors))
            palette = px.colors.qualitative.Plotly
            colors.update({category: palette[index % len(palette)] for index, category in enumerate(missing_colors)})
            fig = px.bar(
                df.with_columns(pl.lit("Model").alias("_model")),
                x=share_column,
                y="_model",
                color="PredictorCategory",
                text=share_column,
                custom_data=[total_column],
                orientation="h",
                category_orders={"PredictorCategory": categories},
                color_discrete_map=colors,
            )
            total_label = "Split nodes" if total_column == "split_count" else "Total gain"
            total_format = ",.0f" if total_column == "split_count" else ",.4g"
            fig.update_traces(
                texttemplate="%{x:.0%}",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    "%{fullData.name}<br>Share: %{x:.1%}<br>"
                    f"{total_label}: %{{customdata[0]:{total_format}}}<extra></extra>"
                ),
            )
        fig.update_layout(
            title=title,
            barmode="stack",
            xaxis_title=axis_title,
            xaxis_tickformat=".0%",
            xaxis_range=[0, 1],
            yaxis_title=None,
            yaxis_showticklabels=False,
            legend_title_text=None,
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
            uniformtext_minsize=11,
            uniformtext_mode="hide",
            height=270,
            margin=dict(l=30, r=30, t=90, b=55),
            template="none",
        )
        return fig

    @overload
    def feature_role_map(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def feature_role_map(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def feature_role_map(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Bubble chart mapping each predictor's role: router vs refiner.

        X-axis = mean edge-count depth of splits; Y-axis = tree coverage
        (distinct trees used in); bubble size = total signed gain.
        Bubbles use a uniform size when any total gain is non-positive.
        Shallow + wide-coverage predictors are "routers"; deep + narrow
        ones are "refiners".

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``predictor``, ``mean_depth``, ``tree_coverage``, ``total_gain``,
            ``PredictorCategory`` when ``return_df=True``.
        """
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        rows: list[dict] = []
        for tree_id, tree in enumerate(self.trees_model.model):
            for node in _iter_nodes(tree):
                if node.split is not None:
                    rows.append(
                        {
                            "predictor": node.split.variable,
                            "depth": node.depth,
                            "gain": node.gain,
                            "treeID": tree_id,
                        }
                    )

        raw = pl.DataFrame(
            rows,
            schema={"predictor": pl.String, "depth": pl.Int64, "gain": pl.Float64, "treeID": pl.Int64},
        )
        df = (
            raw.group_by("predictor")
            .agg(
                pl.col("depth").mean().alias("mean_depth"),
                pl.col("treeID").n_unique().alias("tree_coverage"),
                pl.col("gain").sum().alias("total_gain"),
            )
            .with_columns(self.predictor_category_expr)
        )
        if return_df:
            return df

        if df.is_empty():
            import plotly.graph_objects as go

            return go.Figure().update_layout(title="Feature role map (router vs refiner)")

        return px.scatter(
            df,
            x="mean_depth",
            y="tree_coverage",
            size="total_gain" if (df["total_gain"] > 0).all() else None,
            hover_data={"total_gain": True},
            color="PredictorCategory",
            hover_name="predictor",
            title="Feature role map (router vs refiner)",
            labels={
                "mean_depth": "Mean split depth",
                "tree_coverage": "Tree coverage (# trees)",
                "PredictorCategory": "Predictor category",
            },
            template="none",
            category_orders={"PredictorCategory": self.predictor_category_order},
            color_discrete_map=self.predictor_category_colormap,
        )

    @overload
    def training_stream_timeline(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def training_stream_timeline(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def training_stream_timeline(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot the root-node sample count across trees as a training timeline.

        Requires the model export to carry ``sampleCount`` on tree nodes
        (available from Prediction Studio ≥ Infinity '24).

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``treeID``, ``root_sample_count`` when ``return_df=True``.

        Raises
        ------
        ValueError
            If no tree carries ``sampleCount`` data.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        rows = [
            {"treeID": i, "root_sample_count": tree.get("sampleCount")} for i, tree in enumerate(self.trees_model.model)
        ]
        df = pl.DataFrame(
            rows,
            schema={"treeID": pl.Int32, "root_sample_count": pl.Int64},
        )
        if df["root_sample_count"].is_null().all():
            raise ValueError(
                "No sampleCount data found in this model export. "
                "Use a model exported from Prediction Studio ≥ Infinity '24."
            )
        if return_df:
            return df

        fig = go.Figure(
            go.Scatter(
                x=df["treeID"].to_list(),
                y=df["root_sample_count"].to_list(),
                mode="lines+markers",
                name="Root sample count",
                line={"color": "steelblue"},
            )
        )
        fig.update_layout(
            title="Training stream timeline — root sample count per tree",
            xaxis_title="Tree index",
            yaxis_title="Root sample count",
            template="none",
        )
        return fig

    @overload
    def inter_tree_gaps(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def inter_tree_gaps(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def inter_tree_gaps(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot the change in root sample count between consecutive trees.

        A large positive gap indicates a burst of new training data
        arrived before that tree was built.

        Requires the model export to carry ``sampleCount`` on tree nodes.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``treeID``, ``sample_gap`` when ``return_df=True``.

        Raises
        ------
        ValueError
            If no tree carries ``sampleCount`` data.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        counts = [tree.get("sampleCount") for tree in self.trees_model.model]
        if all(c is None for c in counts):
            raise ValueError(
                "No sampleCount data found in this model export. "
                "Use a model exported from Prediction Studio ≥ Infinity '24."
            )
        gaps = []
        for i in range(1, len(counts)):
            current = counts[i]
            previous = counts[i - 1]
            gaps.append(None if (current is None or previous is None) else current - previous)
        df = pl.DataFrame(
            {"treeID": list(range(1, len(counts))), "sample_gap": gaps},
            schema={"treeID": pl.Int32, "sample_gap": pl.Int64},
        )
        if return_df:
            return df

        fig = go.Figure(
            go.Bar(
                x=df["treeID"].to_list(),
                y=df["sample_gap"].to_list(),
                name="Sample gap",
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            title="Inter-tree sample gaps",
            xaxis_title="Tree index",
            yaxis_title="Δ root sample count",
            template="none",
            margin=dict(l=90),
        )
        return fig

    @overload
    def gain_decay_dual_lens(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    @overload
    def gain_decay_dual_lens(self, *, return_df: Literal[False] = ...) -> "go.Figure": ...

    def gain_decay_dual_lens(
        self,
        *,
        return_df: bool = False,
    ) -> "go.Figure | pl.DataFrame":
        """Plot gain decay viewed through both tree index and training age.

        "Training age" of tree *k* is defined as
        ``totalCount - root_sampleCount_k``, where ``totalCount`` is the
        total training count from model metadata (``trainingStats``) or the
        root sample count of the first tree.  A higher training age means
        the tree was built on an older, smaller slice of the training
        stream.

        Requires ``sampleCount`` on tree nodes.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``treeID``, ``node_age``, ``total_gain`` when ``return_df=True``.

        Raises
        ------
        ValueError
            If no tree carries ``sampleCount`` data.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        root_counts = [tree.get("sampleCount") for tree in self.trees_model.model]
        if all(c is None for c in root_counts):
            raise ValueError(
                "No sampleCount data found in this model export. "
                "Use a model exported from Prediction Studio ≥ Infinity '24."
            )

        props = getattr(self.trees_model, "_properties", {}) or {}
        training = props.get("trainingStats", {})
        total_count = (
            training.get("totalCount")
            or (training.get("positiveCount", 0) + training.get("negativeCount", 0))
            or root_counts[0]
        )

        gain_rows = (
            self.trees_model.tree_stats.select(["treeID", "gains"])
            .with_columns(pl.col("gains").list.sum().alias("total_gain"))
            .select(["treeID", "total_gain"])
        )
        ages = [(total_count - c) if c is not None else None for c in root_counts]
        age_df = pl.DataFrame(
            {"treeID": list(range(len(ages))), "node_age": ages},
            schema={"treeID": pl.Int32, "node_age": pl.Int64},
        )
        df = gain_rows.join(age_df, on="treeID", how="left")
        if return_df:
            return df

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df["treeID"].to_list(),
                y=df["total_gain"].to_list(),
                name="Gain (by tree index)",
                mode="lines",
                line={"color": "steelblue"},
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["node_age"].to_list(),
                y=df["total_gain"].to_list(),
                name="Gain (by training age)",
                mode="lines",
                line={"color": "darkorange", "dash": "dot"},
                visible="legendonly",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Gain decay: tree index vs training age",
            xaxis_title="Tree index",
            template="none",
            margin=dict(r=120),
        )
        fig.update_yaxes(title_text="Total gain", secondary_y=False)
        fig.update_yaxes(title_text="Total gain (training-age axis)", secondary_y=True)
        return fig

    @overload
    def splits_per_variable_type(
        self,
        predictor_categorization: "Callable | None" = ...,
        *,
        return_df: Literal[True],
    ) -> pl.DataFrame: ...

    @overload
    def splits_per_variable_type(
        self,
        predictor_categorization: "Callable | None" = ...,
        *,
        return_df: Literal[False] = ...,
    ) -> "go.Figure": ...

    def splits_per_variable_type(
        self,
        predictor_categorization: "Callable | None" = None,
        *,
        return_df: bool = False,
        **kwargs,
    ) -> "go.Figure | pl.DataFrame":
        """Stacked-area chart of categorised split counts per tree.

        Parameters
        ----------
        predictor_categorization : callable or None, default None
            Optional function mapping predictor name to a type label.
            When ``None``, uses the model's built-in categorisation.
        return_df : bool, default False
            If True, return the underlying wide-format DataFrame instead of
            the figure.
        **kwargs
            Additional keyword arguments forwarded to
            ``plotly.express.area``.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; wide-format DataFrame with
            one column per predictor type and one row per tree when
            ``return_df=True``.
        """
        if predictor_categorization is not None:  # pragma: no cover
            to_plot = self.trees_model.compute_categorization_over_time(predictor_categorization)[0]
        else:
            to_plot = self.trees_model.splits_per_variable_type[0]
        df = pl.DataFrame(to_plot)
        df = df.select(sorted(df.columns))
        if return_df:
            return df
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None
        fig = px.area(
            df,
            title="Variable types per tree",
            labels={"index": "Tree number", "value": "Number of splits", "variable": "Predictor category"},
            template="none",
            color_discrete_map=self.predictor_category_colormap,
            **kwargs,
        )
        fig.layout["updatemenus"] += (
            dict(
                type="buttons",
                direction="left",
                active=0,
                buttons=[
                    dict(
                        args=[
                            {"groupnorm": None},
                            {"yaxis": {"title": "Number of splits"}},
                        ],
                        label="Absolute",
                        method="update",
                    ),
                    dict(
                        args=[
                            {"groupnorm": "percent"},
                            {"yaxis": {"title": "Percentage of splits"}},
                        ],
                        label="Relative",
                        method="update",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
            ),
        )
        return fig
