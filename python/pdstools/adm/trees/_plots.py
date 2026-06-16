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
from typing import TYPE_CHECKING, ClassVar, Literal, overload

import polars as pl

from ...utils.namespaces import LazyNamespace, MissingDependenciesException
from ._nodes import _iter_nodes

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

            display(Image(graph.create_png()))
        except ImportError:
            pass
        except FileNotFoundError as exc:
            logger.error(
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
        df = base.with_columns((pl.col("total_gain").cum_sum() / grand_total).alias("cumulative_gain_share")).select(
            ["treeID", "cumulative_gain_share"]
        )
        if return_df:
            return df

        half_idx = (
            df.filter(pl.col("cumulative_gain_share") >= 0.5)["treeID"].first()
            if (df["cumulative_gain_share"] >= 0.5).any()
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
            fig.add_vline(
                x=half_idx,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"50 % @ tree {half_idx}",
                annotation_position="top right",
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
            ``predictor``, ``total_gain``, ``namespace`` when
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
            .with_columns(pl.col("predictor").str.splitn(".", 2).struct.field("field_0").alias("namespace"))
        )
        if return_df:
            return df

        return px.bar(
            df.sort("total_gain"),
            x="total_gain",
            y="predictor",
            color="namespace",
            orientation="h",
            title=f"Top {top_n} predictors by total gain",
            labels={"total_gain": "Total gain", "predictor": ""},
            template="none",
        ).update_layout(yaxis_automargin=True)

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
        are "early specialists".

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``predictor``, ``early_gain``, ``late_gain``, ``total_gain``,
            ``namespace`` when ``return_df=True``.
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
                if node.split is not None and node.gain > 0:
                    rows.append({"predictor": node.split.variable, "gain": node.gain, "bucket": bucket})

        if not rows:
            raise ValueError("No gain data found in tree splits.")

        raw = pl.DataFrame(rows)
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
                pl.col("predictor").str.splitn(".", 2).struct.field("field_0").alias("namespace"),
            )
        )
        if return_df:
            return df

        max_val = max(df["early_gain"].max() or 0.0, df["late_gain"].max() or 0.0)
        fig = px.scatter(
            df,
            x="early_gain",
            y="late_gain",
            color="namespace",
            size="total_gain",
            hover_name="predictor",
            log_x=True,
            log_y=True,
            title="Early vs late tree gain per predictor",
            labels={
                "early_gain": f"Gain in first {quarter} trees",
                "late_gain": f"Gain in last {quarter} trees",
            },
            template="none",
        )
        fig.add_shape(
            type="line",
            x0=0.01,
            y0=0.01,
            x1=max_val,
            y1=max_val,
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
        """Plot total information gain broken down by predictor namespace.

        The namespace is the first dot-separated segment of the predictor
        name, e.g. ``IH``, ``Customer``, ``py*``.

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``namespace``, ``total_gain``, ``gain_share`` when
            ``return_df=True``.
        """
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        df = (
            self.trees_model.gains_per_split.with_columns(
                pl.col("predictor").str.splitn(".", 2).struct.field("field_0").alias("namespace")
            )
            .group_by("namespace")
            .agg(pl.col("gains").sum().alias("total_gain"))
            .sort("total_gain", descending=True)
        )
        total = df["total_gain"].sum()
        df = df.with_columns((pl.col("total_gain") / total).alias("gain_share"))
        if return_df:
            return df

        fig = px.bar(
            df.sort("total_gain"),
            x="total_gain",
            y="namespace",
            color="namespace",
            orientation="h",
            title="Total gain by predictor namespace",
            labels={"total_gain": "Total gain", "namespace": ""},
            template="none",
        )
        fig.update_layout(showlegend=False)
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

        X-axis = mean depth of splits; Y-axis = tree coverage (distinct
        trees used in); bubble size = total gain.  Shallow + wide-coverage
        predictors are "routers"; deep + narrow ones are "refiners".

        Parameters
        ----------
        return_df : bool, default False
            If True, return the underlying DataFrame instead of the figure.

        Returns
        -------
        plotly.graph_objects.Figure or pl.DataFrame
            Figure when ``return_df=False``; DataFrame with columns
            ``predictor``, ``mean_depth``, ``tree_coverage``, ``total_gain``,
            ``namespace`` when ``return_df=True``.
        """
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm") from None

        rows: list[dict] = []
        for tree_id, tree in enumerate(self.trees_model.model):
            for node in _iter_nodes(tree):
                if node.split is not None and node.gain > 0:
                    rows.append(
                        {
                            "predictor": node.split.variable,
                            "depth": node.depth,
                            "gain": node.gain,
                            "treeID": tree_id,
                        }
                    )

        if not rows:
            raise ValueError("No split data found in trees.")

        raw = pl.DataFrame(rows)
        df = (
            raw.group_by("predictor")
            .agg(
                pl.col("depth").mean().alias("mean_depth"),
                pl.col("treeID").n_unique().alias("tree_coverage"),
                pl.col("gain").sum().alias("total_gain"),
            )
            .with_columns(pl.col("predictor").str.splitn(".", 2).struct.field("field_0").alias("namespace"))
        )
        if return_df:
            return df

        return px.scatter(
            df,
            x="mean_depth",
            y="tree_coverage",
            size="total_gain",
            color="namespace",
            hover_name="predictor",
            title="Feature role map (router vs refiner)",
            labels={"mean_depth": "Mean split depth", "tree_coverage": "Tree coverage (# trees)"},
            template="none",
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
        gaps = [
            None if (counts[i] is None or counts[i - 1] is None) else counts[i] - counts[i - 1]
            for i in range(1, len(counts))
        ]
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
            labels={"index": "Tree number", "value": "Number of splits"},
            template="none",
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
