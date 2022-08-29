from typing import Dict

# Don't want to, but Plotly needs to update in order to remove FutureWarnings.
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ADMVisualisations:
    @staticmethod
    def distribution_graph(df, title):
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
        fig.layout.yaxis2.tickformat = ",.0%"
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

        if query != None:
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
        self,
        df,
        add_bottom_left_text=True,
        facets=None,
        context_keys=None,
        **kwargs,
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

        if isinstance(facets, str) or facets is None:
            facets = [facets]

        figlist = []
        bubble_size = kwargs.pop("bubble_size", 1)
        for facet in facets:
            title = "over all models" if facet == None else f"per {facet}"
            fig = px.scatter(
                df,
                x="Performance",
                y="SuccessRate",
                color="Performance",
                size="ResponseCount",
                facet_col=facet,
                facet_col_wrap=5,
                hover_name="ModelName",
                hover_data=["ModelID"] + context_keys,
                title=f'Bubble Chart {title} {kwargs.get("title_text","")}',
                color_continuous_scale="Bluered",
                template="none",
            )
            fig.update_traces(marker=dict(line=dict(color="black")))

            if add_bottom_left_text:
                if len(fig.layout.annotations) > 0:
                    for i in range(0, len(fig.layout.annotations)):
                        oldtext = fig.layout.annotations[i].text.split("=")
                        subset = df[df[oldtext[0]] == oldtext[1]]
                        bottomleft = len(
                            subset.query(
                                "Performance == 50 & (SuccessRate.isnull() | SuccessRate == 0)",
                                engine="python",
                            )
                        )
                        newtext = f"{len(subset)} models: {bottomleft} ({round(bottomleft/len(subset)*100, 2)}%) at (50,0)"
                        fig.layout.annotations[i].text += f"<br><sup>{newtext}</sup>"
                        fig.data[i].marker.size *= bubble_size

                else:
                    bottomleft = len(
                        df.query(
                            "Performance == 50 & (SuccessRate.isnull() | SuccessRate == 0)",
                            engine="python",
                        )
                    )
                    newtext = f"{len(df)} models: {bottomleft} ({round(bottomleft/len(df)*100, 2)}%) at (50,0)"
                    fig.layout.title.text += f"<br><sup>{newtext}</sup>"
                    fig.data[0].marker.size *= bubble_size

            fig = self.post_plot(fig, name="Bubble", title=title, **kwargs)

            figlist.append(fig)

        return figlist if len(figlist) > 1 else figlist[0]

    # def plotResponseCountMatrix(self, lookback=15, fill_null_days=False, query:Union[str, dict]=None, figsize=(14, 10)):
    #     """Creates a calendar heatmap
    #     x axis shows model names and y axis the dates. Data in each cell is the total number
    #     of responses. The color indicates where responses increased/decreased or
    #     did not change compared to the previous day

    #     Parameters
    #     ----------
    #     lookback : int
    #         Defines how many days to look back at data from the last snapshot
    #     fill_null_days : bool
    #         If True, null values will be generated in the dataframe for
    #         days where there is no model snapshot
    #     query : Union[str, dict]
    #         The query to supply to _apply_query
    #         If a string, uses the default Pandas query function
    #         Else, a dict of lists where the key is column name in the dataframe
    #         and the corresponding value is a list of values to keep in the dataframe
    #     figsize : tuple
    #         Size of graph

    #     Returns
    #     -------
    #     plt.figure
    #     """
    #     table = 'modelData'
    #     multi_snapshot = True
    #     required_columns = {'ModelID', 'ModelName', 'SnapshotTime', 'ResponseCount'}
    #     df = self._subset_data(table, required_columns, query=query, multi_snapshot=multi_snapshot)
    #     assert lookback < df['SnapshotTime'].nunique(), f"Lookback ({lookback}) cannot be larger than the number of snapshots {df['SnapshotTime'].nunique()}"

    #     raise NotImplementedError("This visualisation is not yet implemented.")

    def OverTime(
        self,
        df,
        metric="Performance",
        by="ModelID",
        facets=None,
        **kwargs,
    ):
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

        if isinstance(facets, str) or facets is None:
            facets = [facets]
        hide_legend = kwargs.pop("hide_legend", False)

        figlist = []
        for facet in facets:
            title = "over all models" if facet == None else f"per {facet}"
            if len(df) > 500:
                print(
                    f"Warning: plotting this much data ({len(df)} rows) will probably be slow while not providing many insights. Consider filtering the data by either limiting the number of models, filtering on SnapshotTime or facetting."
                )
            df = df.sort_values(by="SnapshotTime")
            fig = px.line(
                df,
                x="SnapshotTime",
                y=metric,
                color=by,
                hover_data=["ModelName", "Performance", "SuccessRate"],
                markers=True,
                title=f'{metric} over time, per {by} {title} {kwargs.get("title_text","")}',
                facet_col=facet,
                facet_col_wrap=5,
                template="none",
            )
            if hide_legend:
                fig.update_layout(showlegend=False)

            fig = self.post_plot(fig, name="Lines_over_time", title=title, **kwargs)

            figlist.append(fig)

        return figlist if len(figlist) > 1 else figlist[0]

    def PropositionSuccessRates(
        self,
        df,
        metric="SuccessRate",
        by="ModelName",
        show_error=True,
        facets=None,
        **kwargs,
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

        if isinstance(facets, str) or facets is None:
            facets = [facets]
        figlist = []
        for facet in facets:
            title = "over all models" if facet == None else f"per {facet}"
            fig = px.histogram(
                df,
                x=metric,
                y=by,
                color=by,
                histfunc="avg",
                title=f'{metric} of each proposition {title} {kwargs.get("title_text","")}',
                template="none",
            )
            fig.update_yaxes(categoryorder="total ascending")
            fig.update_layout(showlegend=False)
            fig.update_yaxes(dtick=1, automargin=True)
            if by == "ModelName" and show_error:
                stds = np.nan_to_num(df.groupby("ModelName")[metric].std(), 0)
                for index, x in np.ndenumerate(stds):
                    fig.data[index[0]]["error_x"] = {"array": [x], "valueminus": 0}

            fig = self.post_plot(fig, name="Success_rates", title=title, **kwargs)

            figlist.append(fig)

        return figlist if len(figlist) > 1 else figlist[0]

    def ScoreDistribution(self, df, show_zero_responses: bool = False, **kwargs):
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

        figlist = []
        for name, group in df:
            if not show_zero_responses:
                if not group["BinResponseCount"].any():  # pragma: no cover
                    pass
            group = group.sort_values("BinIndex")
            fig = self.distribution_graph(
                group,
                f"Classifier score distribution<br><sup>Model name: {group.ModelName.unique()[0]}<Br>Model ID {name}</sup>",
            )
            fig = self.post_plot(
                fig,
                name="Score_distribution",
                **kwargs,
            )
            figlist.append(fig)
        return figlist if len(figlist) > 1 else figlist[0]

    def PredictorBinning(
        self,
        df,
        **kwargs,
    ):
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

        if (
            kwargs.get("show_each", False) and df.PredictorName.nunique() > 10
        ):  # pragma: no cover
            print(
                f"Warning: will create {df.PredictorName.nunique()} plots. Set 'show_each' argument to False to return plots as list, so you can view them one by one."
            )
        figlist = []
        for modelid, modelidgroup in df.groupby("ModelID"):
            modelname = modelidgroup.ModelName.unique()[0]
            for predictor, predictorgroup in modelidgroup.groupby("PredictorName"):
                title = f"Model name: {modelname}<br><sup>Model ID: {modelid}<br>Predictor name: {predictor}</sup>"
                fig = self.distribution_graph(predictorgroup, title)
                fig = self.post_plot(fig, name="Predictor_binning", **kwargs)
                figlist.append(fig)

        return figlist if len(figlist) > 1 else figlist[0]

    def PredictorPerformance(
        self,
        df,
        order,
        facets=None,
        to_plot="Performance",
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

        # TODO: perhaps get top n & order per facet.
        if isinstance(facets, str) or facets is None:
            facets = [facets]

        colormap = df.Legend.unique()

        if to_plot == "Performance":
            var_to_plot = "PerformanceBin"
        else:
            var_to_plot = to_plot

        figlist = []
        for facet in facets:
            title = "over all models" if facet == None else f"per {facet}"

            fig = px.box(
                df,
                x=var_to_plot,
                y="PredictorName",
                color="Legend",
                template="none",
                title=f"Predictor {to_plot} {title} {kwargs.get('title_text','')}",
                facet_col=facet,
                facet_col_wrap=5,
                labels={
                    var_to_plot: to_plot,
                    "PredictorName": "Predictor Name",
                },
            )

            fig.update_yaxes(
                categoryorder="array", categoryarray=order, automargin=True, dtick=1
            )
            fig.update_traces(marker=dict(color="rgb(0,0,0)"), width=0.6)

            colors = [
                "rgb(14,94,165)",
                "rgb(28,168,154)",
                "rgb(254,183,85)",
                "rgb(45,130,66)",
                "rgb(252,136,72)",
                "rgb(125,94,187)",
                "rgb(252,139,130)",
                "rgb(140,81,43)",
                "rgb(175,161,156)",
            ]

            if len(colormap) > 9:  # pragma: no cover
                colors = px.colors.qualitative.Alphabet

            for i in range(len(fig.data)):
                color = fig.data[i].legendgroup
                fig.data[i].fillcolor = colors[
                    np.where(colormap == color)[0].tolist()[0]
                ]

            fig.update_layout(
                boxgap=0, boxgroupgap=0, legend_title_text="Predictor type"
            )
            fig = self.post_plot(fig, name=f"Predictor_{to_plot}", **kwargs)

            figlist.append(fig)

        return figlist if len(figlist) > 1 else figlist[0]

    def PredictorPerformanceHeatmap(
        self,
        df,
        facets=None,
        **kwargs,
    ):
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

        # NOTE: Unable to add text to image, not sure why.
        if isinstance(facets, str) or facets is None:
            facets = [facets]
        figlist = []
        for facet in facets:
            title = "over all models" if facet == None else f"per {facet}"

            from packaging import version
            import plotly

            assert version.parse(plotly.__version__) >= version.parse(
                "5.5.0"
            ), f"Visualisation requires plotly version 5.5.0 or later (you have version {plotly.__version__}): please upgrade to a newer version."
            if kwargs.get("reindex", None) is not None:
                df = df.reindex(kwargs["reindex"])
            fig = px.imshow(
                df.T,
                text_auto=kwargs.get("text_format", ".0%"),
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
                range_color=kwargs.get("range_color", [0.5, 1]),
            )
            fig.update_yaxes(dtick=1, automargin=True)
            fig.update_xaxes(dtick=1, tickangle=kwargs.get("tickangle", None))

            fig = self.post_plot(fig, name="Predictor_performance_heatmap", **kwargs)
            figlist.append(fig)

        return figlist if len(figlist) > 1 else figlist[0]

    def ResponseGain(
        self,
        df,
        by="Channel",
        **kwargs,
    ):
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

        title = "Cumulative Responses by Models"
        fig = px.line(
            df,
            x="TotalModelsFraction",
            y="TotalResponseFraction",
            color=by,
            labels={
                "TotalResponseFraction": "Percentage of Responses",
                "TotalModelsFraction": "Percentage of Models",
            },
            title=f'{title} {kwargs.get("title_text","")}<br><sup>by {by}</sup>',
            template="none",
        )
        fig.layout.yaxis.tickformat = ",.0%"
        fig.layout.xaxis.tickformat = ",.0%"
        fig = self.post_plot(fig, name="Response_gain", **kwargs)
        return fig

    def ModelsByPositives(
        self,
        df,
        by="Channel",
        **kwargs,
    ):
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

        title = "Percentage of models vs number of positive responses"
        fig = px.line(
            df.query("ModelCount>0"),
            x="PositivesBin",
            y="cumModels",
            color=by,
            markers=True,
            title=f'Percentage of models vs number of positive responses {kwargs.get("title_text","")}<br><sup>By {by}</sup>',
            labels={"cumModels": "Percentage of Models", "PositivesBin": "Positives"},
            template="none",
            category_orders={"PositivesBin": df["PositivesBin"].unique().tolist()},
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
            range_color = kwargs.get("range_color", [0.5, 1])

        elif log:
            color = np.where(
                np.log(df[color]) == -np.inf,
                0,
                np.log(df[color]),
            )
        else:
            color = df[color]

        if midpoint is not None:
            midpoint = np.quantile(color, midpoint)
            colorscale = [
                (0, colorscale[0]),
                (midpoint, colorscale[1]),
                (1, colorscale[2]),
            ]

        hover_data = {
            "Model count": ":.d",
            "Percentage without responses": ":.0%",
            "Response Count sum": ":.d",
            "Success Rate mean": ":.3%",
            "Performance weighted mean": ":.0%",
            "Positives sum": ":.d",
        }

        fig = px.treemap(
            df,
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
                fig.data[0].texttemplate = "%{label}<br>%{text:.2%}"

        if kwargs.get("min_text_size", None) is not None:
            fig.update_layout(
                uniformtext_minsize=kwargs.pop("min_text_size"), uniformtext_mode="hide"
            )
        fig = self.post_plot(fig, name="TreeMap", **kwargs)
        return fig
