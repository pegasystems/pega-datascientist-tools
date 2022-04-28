from typing import NoReturn, Tuple, Union

from matplotlib.lines import Line2D
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as mtick


class ADMVisualisations:
    @staticmethod
    def distribution_graph(df: pd.DataFrame, title: str, figsize: tuple) -> plt.figure:
        """Generic method to generate distribution graphs given data and graph size

        Parameters
        ----------
        df : pd.DataFrame
            The input data
        title : str
            Title of graph
        figsize : tuple
            Size of graph

        Returns
        -------
        plt.figure
            The generated figure
        """
        pd.options.mode.chained_assignment = None
        order = df.sort_values("BinIndex")["BinSymbol"]
        fig, ax = plt.subplots(figsize=figsize)
        df.loc[:, "BinPropensity"] *= 100
        sns.barplot(
            x="BinSymbol",
            y="BinResponseCount",
            data=df,
            ax=ax,
            color="blue",
            order=order,
        )
        ax1 = ax.twinx()
        ax1.plot(
            df.sort_values("BinIndex")["BinSymbol"],
            df.sort_values("BinIndex")["BinPropensity"],
            color="orange",
            marker="o",
        )
        for i in ax.get_xmajorticklabels():
            i.set_rotation(90)
        labels = [
            i.get_text()[0:24] + "..." if len(i.get_text()) > 25 else i.get_text()
            for i in ax.get_xticklabels()
        ]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels)
        ax.set_ylabel("Responses")
        ax.set_xlabel("Range")
        ax1.set_ylabel("Propensity (%)")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        patches = [
            mpatches.Patch(color="blue", label="Responses"),
            mpatches.Patch(color="orange", label="Propensity"),
        ]
        ax.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.5,
            frameon=True,
        )
        ax.set_title(title)
        return ax

    def PerformanceSuccessRateBubbleChart(
        self,
        df,
        annotate: bool = False,
        sizes: tuple = (10, 2000),
        aspect: int = 3,
        b_to_anchor: tuple = (1.1, 0.7),
        figsize: tuple = (20, 5),
        **kwargs,
    ) -> plt.figure:
        """Creates bubble chart similar to ADM OOTB reports

        Parameters
        ----------
        annotate : bool
            If set to True, the total responses per model will be annotated
            to the right of the bubble. All bubbles will be the same size
            if this is set to True
        sizes : tuple
            To determine how sizes are chosen when 'size' is used. 'size'
            will not be used if annotate is set to True
        aspect : int
            Aspect ratio of the graph
        b_to_anchor : tuple
            Position of the legend
        last : bool
            Whether to only include the last snapshot for each model
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

        if annotate:
            gg = sns.relplot(
                x="Performance",
                y="SuccessRate",
                aspect=aspect,
                data=df,
                hue="ModelName",
            )
            ax = gg.axes[0, 0]
            for idx, row in (
                df[["Performance", "SuccessRate", "ResponseCount"]]
                .sort_values("ResponseCount")
                .reset_index(drop=True)
                .reset_index()
                .fillna(-1)
                .iterrows()
            ):
                if row[1] != -1 and row[2] != -1 and row[3] != -1:
                    #                     space = (gg.ax.get_xticks()[2]-gg.ax.get_xticks()[1])/((row[0]+15)/(row[0]+1))
                    ax.text(
                        row[1] + 0.003,
                        row[2],
                        str(row[3]).split(".")[0],
                        horizontalalignment="left",
                    )
            c = gg._legend.get_children()[0].get_children()[1].get_children()[0]
            c._children = c._children[0 : df["ModelName"].count() + 1]
        else:
            gg = sns.relplot(
                x="Performance",
                y="SuccessRate",
                size="ResponseCount",
                data=df,
                hue="ModelName",
                sizes=sizes,
                aspect=aspect,
            )

        gg.fig.set_size_inches(figsize[0], figsize[1])
        plt.setp(gg._legend.get_texts(), fontsize="10")
        gg.ax.set_xlabel("Performance")
        gg.ax.set_xlim(48, 100)
        gg.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        gg._legend.set_bbox_to_anchor(b_to_anchor)
        return gg

    def PerformanceAndSuccessRateOverTime(
        self,
        df,
        day_interval: int = 7,
        query: Union[str, dict] = None,
        figsize: tuple = (16, 10),
        **kwargs,
    ) -> plt.figure:
        """Shows responses and performance of models over time

        Reponses are on the y axis and the performance of the model is indicated by heatmap.
        x axis is date

        Parameters
        ----------
        day_interval : int
            Interval of tick labels along x axis
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

        fig, ax = plt.subplots(figsize=figsize)
        norm = colors.Normalize(vmin=0.5, vmax=1)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.gnuplot_r)
        for ids in df["ModelID"].unique():
            _df = df[df["ModelID"] == ids].sort_values("SnapshotTime")
            name = _df["ModelName"].unique()[0]
            ax.plot(
                _df["SnapshotTime"].values, _df["ResponseCount"].values, color="gray"
            )
            ax.scatter(
                _df["SnapshotTime"].values,
                _df["ResponseCount"].values,
                color=[mapper.to_rgba(v) for v in _df["Performance"].values],
            )
            if _df["ResponseCount"].max() > 1:
                ax.text(
                    _df["SnapshotTime"].max(),
                    _df["ResponseCount"].max(),
                    "   " + name,
                    {"fontsize": 9},
                )
        for i in ax.get_xmajorticklabels():
            i.set_rotation(90)
        ax.set_ylabel("ResponseCount")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        ax.set_yscale("log")
        mapper._A = []
        cbar = fig.colorbar(mapper)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel("Model Performance (AUC)")
        print("Maximum AUC across all models: %.2f" % df["Performance"].max())
        return ax

    def ResponseCountMatrix(
        self,
        annot_df,
        heatmap_df,
        lookback=15,
        fill_null_days=False,
        query: Union[str, dict] = None,
        figsize=(14, 10),
        **kwargs,
    ) -> plt.figure:
        """Creates a calendar heatmap

        x axis shows model names and y axis the dates. Data in each cell is the total number
        of responses. The color indicates where responses increased/decreased or
        did not change compared to the previous day

        Parameters
        ----------
        lookback : int
            Defines how many days to look back at data from the last snapshot
        fill_null_days : bool
            If True, null values will be generated in the dataframe for
            days where there is no model snapshot
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

        f, ax = plt.subplots(figsize=figsize)
        myColors = ["r", "orange", "w"]
        colorText = ["Decreased", "No Change", "Increased or NA"]
        cmap = colors.ListedColormap(myColors)
        sns.heatmap(
            heatmap_df.T,
            annot=annot_df.T,
            mask=annot_df.T.isnull(),
            ax=ax,
            linewidths=0.5,
            fmt=".0f",
            cmap=cmap,
            vmin=-1,
            vmax=1,
            cbar=False,
        )
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        patches = [
            mpatches.Patch(color=myColors[i], label=colorText[i])
            for i in range(len(myColors))
        ]

        legend = plt.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.5,
            frameon=True,
        )
        frame = legend.get_frame()
        frame.set_facecolor("lightgrey")
        return ax

    def OverTime(
        self,
        df,
        metric="Performance",
        by="ModelID",
        day_interval: int = 7,
        query: Union[str, dict] = None,
        figsize: tuple = (16, 10),
        **kwargs,
    ) -> plt.figure:
        """Shows success rate of models over time

        Parameters
        ----------
        day_interval (int):
            interval of tick labels along x axis
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

        assert (
            day_interval <= df["SnapshotTime"].nunique()
        ), f"Day interval ({day_interval}) cannot be larger than the number of snapshots ({df['SnapshotTime'].nunique()})"

        fig, ax = plt.subplots(figsize=figsize)
        format = "(%)" if metric != "Positives" else None
        if format == "(%)":
            df.loc[:, metric] *= 100
        sns.pointplot(x="SnapshotTime", y=metric, data=df, hue=by, marker="o", ax=ax)
        print("Pointplot generated")
        modelnames = (
            df[["ModelID", "ModelName"]]
            .drop_duplicates()
            .set_index("ModelID")
            .to_dict()["ModelName"]
        )
        print("Modelnames generated")
        handles, labels = ax.get_legend_handles_labels()
        newlabels = [modelnames[i] for i in labels]
        ax.legend(handles, newlabels, bbox_to_anchor=(1.05, 1), loc=2)
        if format == "(%)":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel(f"{metric} {format}")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        print("Setting rotations")
        for i in ax.get_xmajorticklabels():
            i.set_rotation(90)
        return ax

    def PropositionSuccessRates(
        self,
        df,
        metric,
        by,
        query: Union[str, dict] = None,
        figsize: tuple = (12, 8),
        **kwargs,
    ) -> plt.figure:
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
        figsize : tuple
            Size of graph

        Returns
        -------
        plt.figure
        """

        f, ax = plt.subplots(figsize=figsize)
        df[metric] *= 100
        bplot = sns.barplot(
            x=metric,
            y=by,
            data=df.sort_values(metric, ascending=False),
            ax=ax,
        )
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        for p in bplot.patches:
            bplot.annotate(
                "{:0.2%}".format(p.get_width() / 100.0),
                (p.get_width(), p.get_y() + p.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
            )
        return ax

    def ScoreDistribution(
        self,
        df,
        show_zero_responses: bool = False,
        query: Union[str, dict] = None,
        figsize: tuple = (14, 10),
        **kwargs,
    ) -> plt.figure:
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
        figsize : tuple
            Size of graph

        Returns
        -------
        plt.figure
        """

        for name, group in df:
            if not show_zero_responses:
                if not group["BinResponseCount"].any():  # pragma: no cover
                    pass
            self.distribution_graph(group, f"Model ID:{name}", figsize)

    def PredictorBinning(
        self,
        df,
        query: Union[str, dict] = None,
        figsize: tuple = (10, 5),
        **kwargs,
    ) -> plt.figure:
        """Show predictor graphs for a given model
        
        For a given model (query) shows all its predictors' graphs. If certain predictors
        selected, only those predictor graphs will be shown

        Parameters
        ----------
        predictors : list
            List of predictors to show their graphs, optional
        ModelID : str
            List of model IDs to subset on, optional
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

        for modelid, modelidgroup in df.groupby("ModelID"):
            modelname = modelidgroup.ModelName.unique()[0]
            for predictor, predictorgroup in modelidgroup.groupby("PredictorName"):
                title = f"Model name: {modelname}\nModel ID: {modelid}\nPredictor name: {predictor}"
                self.distribution_graph(predictorgroup, title, figsize)

    def PredictorPerformance(
        self,
        df,
        order,
        query: Union[str, dict] = None,
        figsize: tuple = (6, 12),
        **kwargs,
    ) -> plt.figure:
        """Shows a box plot of predictor performance

        Parameters
        ----------
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

        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(x="PerformanceBin", y="PredictorName", data=df, order=order, ax=ax)
        ax.set_xlabel("Predictor Performance")
        ax.set_ylabel("Predictor Name")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

        norm = colors.Normalize(vmin=0, vmax=len(df["Legend"].unique()) - 1)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.gnuplot_r)
        cl_dict = dict(
            zip(
                df["Legend"].unique(),
                [mapper.to_rgba(v) for v in range(len(df["Legend"].unique()))],
            )
        )
        value_dict = dict(df[["PredictorName", "Legend"]].drop_duplicates().values)
        type_dict = dict(df[["PredictorName", "Type"]].drop_duplicates().values)
        boxes = ax.artists
        for i in range(len(boxes)):  # pragma: no cover
            boxes[i].set_facecolor(cl_dict[value_dict[order[i]]])
            if type_dict[order[i]].lower() == "symbolic":
                boxes[i].set_linestyle("--")

        lines = [
            Line2D([], [], label="Numeric", color="black", linewidth=1.5),
            Line2D(
                [], [], label="Symbolic", color="black", linewidth=1.5, linestyle="--"
            ),
        ]
        legend_type = plt.legend(
            handles=lines,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.5,
            frameon=True,
            title="Predictor Type \n",
        )
        patches = [mpatches.Patch(color=j, label=i) for i, j in cl_dict.items()]
        legend = plt.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 0.85),
            loc=2,
            borderaxespad=0.5,
            frameon=True,
            title="Predictor Source \n",
        )
        plt.gca().add_artist(legend_type)
        legend._legend_box.align = "left"
        legend_type._legend_box.align = "left"
        return ax

    def PredictorPerformanceHeatmap(
        self, df, query: Union[str, dict] = None, figsize=(14, 10), **kwargs
    ) -> plt.figure:
        """Shows a heatmap plot of predictor performance across models

        Parameters
        ----------
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        figsize : tuple
            size of graph

        Returns
        -------
        plt.figure
        """

        cmap = colors.LinearSegmentedColormap.from_list(
            "mycmap",
            [
                (0 / 1, "red"),
                (0.2 / 1, "green"),
                (0.9 / 1, "white"),
                (1 / 1, "white"),
            ],
        )
        f, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df.fillna(0.5).T,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            vmin=0.5,
            vmax=1,
        )
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        return ax

    def ImpactInfluence(
        self,
        df,
        ModelID: str = None,
        query: Union[str, dict] = None,
        figsize: tuple = (12, 5),
        **kwargs,
    ) -> plt.figure:
        """Calculate the impact and influence of a given model's predictors

        Parameters
        ----------
        modelID : str
            The selected model ID
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        figsize : tuple
            size of graph

        Returns
        -------
        plt.figure
        """

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x="PredictorName", y="value", data=df, hue="metric", ax=ax)
        ax.legend(bbox_to_anchor=(1.01, 1), loc=2)
        ax.set_ylabel("Metrics")
        return ax
