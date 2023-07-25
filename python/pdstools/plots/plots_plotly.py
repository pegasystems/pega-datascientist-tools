import warnings
import logging

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import math
from plotly.subplots import make_subplots
from typing import Optional
from pdstools.utils import pega_template


class ADMVisualisations:
    @staticmethod
    def distribution_graph(df, title):
        df = df.to_pandas(use_pyarrow_extension_array=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=df["BinSymbol"], y=df["BinResponseCount"], name="Responses")
        )
        fig.add_trace(
            go.Scatter(
                x=df["BinSymbol"],
                y=df["BinPropensity"],
                yaxis="y2",
                name="Propensity",
                mode="lines+markers",
            )
        )
        fig.update_layout(
            template="none", title=title, xaxis_title="Range", yaxis_title="Responses"
        )
        fig.update_yaxes(title_text="Propensity", secondary_y=True)
        fig.layout.yaxis2.tickformat = ",.2%"
        fig.layout.yaxis2.zeroline = False
        fig.update_yaxes(showgrid=False)

        return fig

    @staticmethod
    def post_plot(
        fig,
        name,
        query=None,
        title=None,
        file_title=None,
        file_path=".",
        show_each=False,
        image_format=None,
        to_html=False,
        **kwargs,
    ):
        if query is not None:
            fig.layout.title.text += f"<br><sup>Query: {query}</sup>"

        if to_html:  # pragma: no cover
            filename = name
            if file_title is not None:
                filename += file_title
            if title is not None:
                filename += title
            fig.write_html(f"{file_path}/{filename}.html")
        if show_each:  # pragma: no cover
            fig.show(image_format)
        return fig

    def PerformanceSuccessRateBubbleChart(
        self, df, add_bottom_left_text=True, facet=None, context_keys=None, **kwargs
    ):
        """Creates bubble chart similar to ADM OOTB reports

        Parameters
        ----------
        last : bool
            Whether to only include the last snapshot for each model
        add_bottom_left_text : bool
            Whether to subtitle text to indicate how many models are at 0,50
        to_html : bool
            Whether to write image to html, with title file_title at path file_path
        file_title : Optional[str]
            The title of the image when written to html
        file_path : Optional[str]
            The location the file will be written when written to html
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        show_each : bool
            Whether to show each file when multiple facets are used
        facets : Optional[Union[list, str]]
            Whether to create a chart for multiple facets or subsets.
            For example, if facets == 'Channel', a bubble plot is made for each channel
            Depending on show_each, every chart is either shown or not
            If more than one facet is visualised, they are returned in a list

        Returns
        -------
        px.Figure
        """

        bubble_size = kwargs.pop("bubble_size", 1)
        title = "over all models" if facet is None else f"per {facet}"
        fig = px.scatter(
            df.to_pandas(use_pyarrow_extension_array=True),
            x="Performance",
            y="SuccessRate",
            color="Performance",
            size="ResponseCount",
            facet_col=facet,
            facet_col_wrap=kwargs.pop("facet_col_wrap", 5),
            hover_name="Name",
            hover_data=["ModelID"] + context_keys,
            title=f'Bubble Chart {title} {kwargs.get("title_text","")}',
            template="pega",
            facet_row_spacing=0.01,
        )
        fig.update_traces(marker=dict(line=dict(color="black")))

        if add_bottom_left_text:
            if len(fig.layout.annotations) > 0:
                for i in range(0, len(fig.layout.annotations)):
                    oldtext = fig.layout.annotations[i].text.split("=")
                    subset = df.filter(pl.col(oldtext[0]) == oldtext[1])
                    bottomleft = len(
                        subset.filter(
                            (pl.col("Performance") == 50)
                            & (
                                (pl.col("SuccessRate").is_null())
                                | (pl.col("SuccessRate") == 0)
                            )
                        )
                    )
                    if len(subset) > 0:
                        newtext = f"{len(subset)} models: {bottomleft} ({round(bottomleft/len(subset)*100, 2)}%) at (50,0)"
                        fig.layout.annotations[i].text += f"<br><sup>{newtext}</sup>"
                        if len(fig.data) > i:
                            fig.data[i].marker.size *= bubble_size
                        else:
                            print(fig.data, i)

            else:
                bottomleft = len(
                    df.filter(
                        (pl.col("Performance") == 50)
                        & (
                            (pl.col("SuccessRate").is_null())
                            | (pl.col("SuccessRate") == 0)
                        )
                    )
                )
                newtext = f"{len(df)} models: {bottomleft} ({round(bottomleft/len(df)*100, 2)}%) at (50,0)"
                fig.layout.title.text += f"<br><sup>{newtext}</sup>"
                fig.data[0].marker.size *= bubble_size

        return self.post_plot(fig, name="Bubble", title=title, **kwargs)

    def OverTime(self, df, metric="Performance", by="ModelID", facet=None, **kwargs):
        """Shows metric of models over time

        Parameters
        ----------
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Returns
        -------
        plt.figure
        """

        hide_legend = kwargs.pop("hide_legend", False)
        metric_hovers = {
            "SuccessRate": ":.4%",
            "Performance": ":.4f",
            "weighted_performance": ":.4f",
            "Positives": ":.d",
            "ResponseCount": ":.d",
        }
        if metric in ["Performance", "weighted_performance", "SuccessRate"]:
            df = df.to_pandas(use_pyarrow_extension_array=False)
            x = "SnapshotTime"
            y = metric
            color = by
            hover_data = {by: ":.d", metric: metric_hovers[metric]}
        else:
            df = df.to_pandas(use_pyarrow_extension_array=False).set_index(
                "SnapshotTime"
            )
            x = None
            y = "Increase"
            color = by
            hover_data = None
        title = "over all models" if facet is None else f"per {facet}"
        if len(df) > 500:
            print(
                f"Warning: plotting this much data ({len(df)} rows) will probably be slow while not providing many insights. Consider filtering the data by either limiting the number of models, filtering on SnapshotTime or facetting."
            )
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            hover_data=hover_data,
            markers=True,
            title=f'{metric} over time, per {by} {title} {kwargs.get("title_text","")}',
            facet_col=facet,
            facet_col_wrap=kwargs.pop("facet_col_wrap", 5),
            template="pega",
        )
        if hide_legend:
            fig.update_layout(showlegend=False)
        if metric == "SuccessRate":
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(yaxis={"rangemode": "tozero"})

        return self.post_plot(fig, name="Lines_over_time", title=title, **kwargs)

    def PropositionSuccessRates(
        self, df, metric="SuccessRate", by="Name", show_error=True, facet=None, **kwargs
    ):
        """Shows all latest proposition success rates

        A bar plot to show the success rate of all latest model instances (propositions)
        For reading simplicity, latest success rate is also annotated next to each bar

        Parameters
        ----------
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe


        Returns
        -------
        plt.figure
        """

        title = "over all models" if facet is None else f"per {facet}"
        facet = facet if facet in df.columns else None
        fig = px.histogram(
            df.to_pandas(use_pyarrow_extension_array=True),
            x=metric,
            y=by,
            color=by,
            facet_col=facet,
            facet_col_wrap=kwargs.get("facet_col_wrap", 5),
            histfunc="avg",
            title=f'{metric} of each proposition {title} {kwargs.get("title_text","")}',
            template="pega",
        )
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(dtick=1, automargin=True)
        if show_error:
            errors = {
                i[0]: i[1]
                for i in df.groupby(by, maintain_order=True)
                .agg(pl.std("SuccessRate").fill_nan(0))
                .iter_rows()
            }
            for i, bar in enumerate(fig.data):
                fig.data[i]["error_x"] = {
                    "array": [errors[bar["name"]]],
                    "valueminus": 0,
                }

        return self.post_plot(fig, name="Success_rates", title=title, **kwargs)

    def ScoreDistribution(self, df, facet_val, **kwargs):
        """Show score distribution similar to ADM out-of-the-box report

        Shows a score distribution graph per model. If certain models selected,
        only those models will be shown.
        the only difference between this graph and the one shown on ADM
        report is that, here the raw number of responses are shown on left y-axis whereas in
        ADM reports, the percentage of responses are shown

        Parameters
        ----------
        show_zero_responses:bool
            Whether to include bins with no responses at all
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Returns
        -------
        plt.figure
        """

        df = df.sort("BinIndex")
        fig = self.distribution_graph(
            df,
            f"""Classifier score distribution<br>
            <sup>Model name: {df['Name'].unique().item()}
            <br>Model ID {facet_val[0]}</sup>""",
        )
        return self.post_plot(fig, name="Score_distribution", **kwargs)

    def PredictorBinning(self, df, facet_val, **kwargs):
        """Show predictor graphs for a given model

        For a given model (query) shows all its predictors' graphs. If certain predictors
        selected, only those predictor graphs will be shown

        Parameters
        ----------
        predictors : list
            List of predictors to show their graphs, optional
        modelid : str
            Model IDs to subset on, optional
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        figsize : tuple
            Size of graph

        Returns
        -------
        plt.figure
        """

        name = df.select(["Name"]).row(0)
        modelid, predictorname = facet_val
        title = f"Model name: {name}<br><sup>Model ID: {modelid}<br>Predictor name: {predictorname}</sup>"
        fig = self.distribution_graph(df, title)
        return self.post_plot(fig, name="Predictor_binning", **kwargs)

    def PredictorPerformance(
        self,
        df,
        facet=None,
        order=None,
        to_plot="Performance",
        y="PredictorName",
        **kwargs,
    ):
        """Shows a box plot of predictor performance

        Parameters
        ----------
        top_n : int
            The number of top performing predictors to show
            If 0 (default), all predictors are shown
        to_html : bool
            Whether to write image to html, with title file_title at path file_path
        file_title : Optional[str]
            The title of the image when written to html
        file_path : Optional[str]
            The location the file will be written when written to html
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets : Optional[Union[list, str]]
            Whether to create a chart for multiple facets or subsets.

        Returns
        -------
        px.Figure
        """

        # TODO: perhaps get top n & order per facet.

        title = (
            f"{kwargs.get('facet_val','over all models')}"
            if facet is None
            else f"per {facet}"
        )
        df = df.to_pandas(use_pyarrow_extension_array=True)
        if order:
            df[y] = df[y].astype("category")
            df[y] = df[y].cat.set_categories(order)
        fig = px.box(
            df.sort_values([y]),
            x=to_plot,
            y=y,
            color="Legend",
            template="pega",
            title=f"Predictor Performance {title} {kwargs.get('title_text','')}",
            facet_col=facet,
            facet_col_wrap=kwargs.get("facet_col_wrap", 5),
            labels={"PredictorName": "Predictor Name", "PerformanceBin": "Performance"},
        )

        fig.update_yaxes(
            categoryorder="array", categoryarray=order, automargin=True, dtick=1
        )

        fig.update_layout(
            boxgap=0, boxgroupgap=0, legend_title_text="Predictor category"
        )

        return self.post_plot(fig, name=f"Predictor_{to_plot}", **kwargs)

    def _divide_context(func):
        """
        Divides the context keys if they are combined
        """
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)
            try:
                context_cols = kwargs.get("by").split("/")
                x_axis_values = fig.data[0].x
                y_axis_values = fig.data[0].y
                custom_values = []
                hover_template = "PredictorName: %{y}<br>Performance: %{z:.2f}"
                if context_cols is not None:
                    for i, col in enumerate(context_cols):
                        my_column_values = [x.split("/")[i] for x in x_axis_values]
                        index_column = np.repeat(
                            my_column_values, len(y_axis_values)
                        ).reshape(len(y_axis_values), len(x_axis_values))
                        custom_values.append(index_column)
                        template_text = f"<br>{col}: %{{customdata[{i}]}}"
                        hover_template += template_text

                custom_data = tuple(custom_values)
                fig.update(
                    data=[
                        {
                            "customdata": np.dstack(custom_data),
                            "hovertemplate": hover_template,
                        }
                    ]
                )
                return fig
            except Exception as e:
                logging.info(
                    f"Couldn't seperate the context keys because of the error: {e}"
                )
                return fig

        return wrapper

    @_divide_context
    def PredictorPerformanceHeatmap(self, df, facet=None, **kwargs):
        """Shows a heatmap plot of predictor performance across models

        Parameters
        ----------
        top_n : int
            Whether to subset to a top number of predictors
            If 0 (default), all predictors are shown
        to_html : bool
            Whether to write image to html, with title file_title at path file_path
        file_title : Optional[str]
            The title of the image when written to html
        file_path : Optional[str]
            The location the file will be written when written to html
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        show_each : bool
            Whether to show each file when multiple facets are used
        facets : Optional[Union[list, str]]
            Whether to create a chart for multiple facets or subsets.
            For example, if facets == 'Channel', a bubble plot is made for each channel
            Depending on show_each, every chart is either shown or not
            If more than one facet is visualised, they are returned in a list

        Returns
        -------
        px.Figure
        """

        import plotly
        from packaging import version

        assert version.parse(plotly.__version__) >= version.parse(
            "5.5.0"
        ), f"Visualisation requires plotly version 5.5.0 or later (you have version {plotly.__version__}): please upgrade to a newer version."

        title = "over all models" if facet is None else f"per {facet}"
        if kwargs.get("reindex", None) is not None:
            df = df[kwargs["reindex"]]
        df = df.to_pandas(use_pyarrow_extension_array=True)
        df.set_index(df.columns[0], inplace=True)
        fig = px.imshow(
            df.T,
            text_auto=kwargs.get("text_format", ".1f"),
            aspect="auto",
            color_continuous_scale=kwargs.get(
                "colorscale",
                [
                    (0, "#d91c29"),
                    (kwargs.get("midpoint", 0.01), "#F76923"),
                    (kwargs.get("acceptable", 0.6) / 2, "#20aa50"),
                    (0.8, "#20aa50"),
                    (1, "#0000FF"),
                ],
            ),
            facet_col=facet,
            facet_col_wrap=5,
            title=f'Top predictors {title} {kwargs.get("title_text","")}',
            range_color=kwargs.get("range_color", [50, 100]),
        )
        fig.update_yaxes(dtick=1, automargin=True)
        fig.update_xaxes(dtick=1, tickangle=kwargs.get("tickangle", None))

        return self.post_plot(fig, name="Predictor_performance_heatmap", **kwargs)

    def ResponseGain(self, df, by="Channel", facet=None, **kwargs):
        """Plots the cumulative response per model, subsetted by 'by'

        Parameters
        ----------
        by : str
            The column to calculate response gain by
        to_html : bool
            Whether to write image to html, with title file_title at path file_path
        file_title : Optional[str]
            The title of the image when written to html
        file_path : Optional[str]
            The location the file will be written when written to html
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        show_each : bool
            Whether to show each file when multiple facets are used
        facets : Optional[Union[list, str]]
            Whether to create a chart for multiple facets or subsets.
            For example, if facets == 'Channel', a bubble plot is made for each channel
            Depending on show_each, every chart is either shown or not
            If more than one facet is visualised, they are returned in a list

        Returns
        -------
        px.Figure
        """
        if isinstance(by, list):
            by = by[0]
        title = "over all models" if facet is None else f"per {facet}"

        title = f"Cumulative Responses {title}"
        fig = px.line(
            df.to_pandas(use_pyarrow_extension_array=False),
            x="TotalModelsFraction",
            y="TotalResponseFraction",
            color=by,
            labels={
                "TotalResponseFraction": "Percentage of Responses",
                "TotalModelsFraction": "Percentage of Models",
            },
            title=f'{title} {kwargs.get("title_text","")}<br><sup>by {by}</sup>',
            template="pega",
        )
        fig.layout.yaxis.tickformat = ",.1%"
        fig.layout.xaxis.tickformat = ",.1%"
        fig = self.post_plot(fig, name="Response_gain", **kwargs)
        return fig

    def ModelsByPositives(self, df, by="Channel", **kwargs):
        """Plots the percentage of models vs the number of positive responses

        Parameters
        ----------
        by : str
            The column to calculate the model percentage by
        to_html : bool
            Whether to write image to html, with title file_title at path file_path
        file_title : Optional[str]
            The title of the image when written to html
        file_path : Optional[str]
            The location the file will be written when written to html
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        show_each : bool
            Whether to show each file when multiple facets are used
        facets : Optional[Union[list, str]]
            Whether to create a chart for multiple facets or subsets.
            For example, if facets == 'Channel', a bubble plot is made for each channel
            Depending on show_each, every chart is either shown or not
            If more than one facet is visualised, they are returned in a list

        Returns
        -------
        px.Figure
        """

        title = f"Percentage of models vs number of positive responses {kwargs.get('title_text','')}<br><sup>By {by}</sup>"
        fig = px.line(
            df.filter(pl.col("ModelCount") > 0)
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
                "PositivesBin": df.select("PositivesBin", "break_point")
                .unique()
                .sort("break_point")["PositivesBin"]
                .to_list()
            },
        )
        fig.layout.yaxis.tickformat = ",.0%"
        fig = self.post_plot(fig, name="Models_by_positives", **kwargs)
        return fig

    def TreeMap(
        self,
        df,
        color,
        values,
        title,
        reverse_scale,
        log,
        midpoint,
        format,
        context_keys,
        value_in_text=True,
        **kwargs,
    ):
        """Plots a treemap to view performance over multiple context keys

        Parameters
        ----------
        color : str
            The column to set as the color of the squares
            One out of:
            {responsecount, responsecount_log, positives,
            positives_log, percentage_without_responses,
            performance_weighted, successrate}
        by : str
            The column to set as the size of the squares
        value_in_text : str
            Whether to print the values of the squares in the squares
        midpoint : Optional[float]
            A parameter to assert more control over the color distribution
            Set near 0 to give lower values a 'higher' color
            Set near 1 to give higher values a 'lower' color
            Necessary for, for example, Success Rate, where rates lie very far apart
            If not supplied in such cases, there is no difference in the color
            between low values such as 0.001 and 0.1, so midpoint should be set low
        to_html : bool
            Whether to write image to html, with title file_title at path file_path
        file_title : Optional[str]
            The title of the image when written to html
        file_path : Optional[str]
            The location the file will be written when written to html
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        show_each : bool
            Whether to show each file when multiple facets are used
        facets : Optional[Union[list, str]]
            Whether to create a chart for multiple facets or subsets.
            For example, if facets == 'Channel', a bubble plot is made for each channel
            Depending on show_each, every chart is either shown or not
            If more than one facet is visualised, they are returned in a list

        Returns
        -------
        px.Figure
        """

        context_keys = [px.Constant("All contexts")] + context_keys
        colorscale = kwargs.pop("colorscale", ["#d91c29", "#F76923", "#20aa50"])

        range_color = None

        if color == "Performance weighted mean":
            colorscale = [
                (0, "#d91c29"),
                (kwargs.get("midpoint", 0.01), "#F76923"),
                (kwargs.get("acceptable", 0.6) / 2, "#20aa50"),
                (0.8, "#20aa50"),
                (1, "#0000FF"),
            ]
            range_color = kwargs.get("range_color", [50, 100])

        elif log:
            df.select(
                pl.when(pl.log(color) == float("-inf")).then(0).otherwise(pl.log(color))
            ).to_series()
        if "range_color" in kwargs:
            range_color = kwargs.get("range_color")
        if midpoint is not None:
            midpoint = color.quantile(midpoint)
            colorscale = [
                (0, colorscale[0]),
                (midpoint, colorscale[1]),
                (1, colorscale[2]),
            ]

        hover_data = {
            "Model count": ":.d",
            "Percentage without responses": ":.0%",
            "Response Count sum": ":.d",
            "(%) Success Rate mean": ":.3f",
            "Performance weighted mean": ":.2f",
            "Positives sum": ":.d",
        }

        fig = px.treemap(
            df.to_pandas(use_pyarrow_extension_array=True),
            path=context_keys,
            color=color,
            values=values,
            title=f"{title}",
            hover_data=hover_data,
            color_continuous_scale=colorscale,
            range_color=range_color,
        )
        fig.update_coloraxes(reversescale=reverse_scale)

        if value_in_text:
            fig.update_traces(text=fig.data[0].marker.colors.round(3))
            fig.data[0].textinfo = "label+text"
            if format == "%":
                fig.data[0].texttemplate = "%{label}<br>%{text:.2f}%"

        if kwargs.get("min_text_size", None) is not None:
            fig.update_layout(
                uniformtext_minsize=kwargs.pop("min_text_size"), uniformtext_mode="hide"
            )
        return self.post_plot(fig, name="TreeMap", **kwargs)

    def PredictorCount(self, df, facet, **kwargs):
        title = "over all models" if facet == None else f"per {facet}"

        fig = px.box(
            df.to_pandas(use_pyarrow_extension_array=True),
            x="Predictor Count",
            y="Type",
            color="EntryType",
            template="pega",
            title=f"Predictor Count {title}",
            facet_col=facet,
            facet_col_wrap=2,
        )

        return self.post_plot(fig, name="PredictorCount", **kwargs)

    def PredictorContribution(self, df, by, **kwargs):
        color = "PredictorCategory"
        fig = px.bar(
            df.sort(color).to_pandas(),
            x="Contribution",
            y=by,
            color=color,
            orientation="h",
            template="pega",
            title="Contribution of different sources",
        )
        return self.post_plot(fig, name="PredictorContribution", **kwargs)
