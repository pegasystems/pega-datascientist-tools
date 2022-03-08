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

from plot_base import VizBase


class ADMVisualisations(VizBase):
    
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
        required_columns = {
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
        }
        assert required_columns.issubset(df.columns)
        pd.options.mode.chained_assignment = None
        order = df.sort_values("BinIndex")["BinSymbol"]
        fig, ax = plt.subplots(figsize=figsize)
        df.loc[:,"BinPropensity"] *= 100
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

    def plotPerformanceSuccessRateBubbleChart(self, annotate:bool=False, sizes:tuple=(10, 2000), aspect:int=3, b_to_anchor:tuple=(1.1,0.7), last=True, query:Union[str, dict]=None, figsize:tuple=(20, 5)) -> plt.figure:
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

        table = 'modelData'
        required_columns = {'Performance', 'SuccessRate', 'ResponseCount', 'ModelName'}
        df = self._subset_data(table=table, required_columns=required_columns, query=query, last=last)
        df['SuccessRate'] *= 100
        df['Performance'] *= 100
        if annotate:
            gg = sns.relplot(x='Performance', y='SuccessRate', aspect=aspect, data=df, hue='ModelName')
            ax = gg.axes[0,0]
            for idx,row in df[['Performance', 'SuccessRate', 'ResponseCount']].sort_values(
                'ResponseCount').reset_index(drop=True).reset_index().fillna(-1).iterrows():
                if row[1]!=-1 and row[2]!=-1 and row[3]!=-1:
    #                     space = (gg.ax.get_xticks()[2]-gg.ax.get_xticks()[1])/((row[0]+15)/(row[0]+1))
                    ax.text(row[1]+0.003,row[2],str(row[3]).split('.')[0], horizontalalignment='left')
            c = gg._legend.get_children()[0].get_children()[1].get_children()[0]
            c._children = c._children[0:df['ModelName'].count()+1]
        else:
            gg = sns.relplot(x='Performance', y='SuccessRate', size='ResponseCount',
                                data=df, hue='ModelName',  sizes=sizes, aspect=aspect)

        gg.fig.set_size_inches(figsize[0], figsize[1])
        plt.setp(gg._legend.get_texts(), fontsize='10')
        gg.ax.set_xlabel('Performance')
        gg.ax.set_xlim(48, 100)
        gg.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        gg._legend.set_bbox_to_anchor(b_to_anchor)

    def plotPerformanceAndSuccessRateOverTime(self, day_interval:int=7, query:Union[str, dict]=None, figsize:tuple=(16, 10)) -> plt.figure:
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
        table = 'modelData'
        multi_snapshot = True
        required_columns = {'ModelName', 'SnapshotTime', 'ModelID', 'Performance', 'ResponseCount'}
        df = self._subset_data(table, required_columns, query, multi_snapshot=multi_snapshot)

        fig, ax = plt.subplots(figsize=figsize)
        norm = colors.Normalize(vmin=0.5, vmax=1)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.gnuplot_r)
        for ids in df['ModelID'].unique():
            _df = df[df['ModelID']==ids].sort_values('SnapshotTime')
            name = _df['ModelName'].unique()[0]
            ax.plot(_df['SnapshotTime'].values, _df['ResponseCount'].values, color='gray')
            ax.scatter(_df['SnapshotTime'].values, _df['ResponseCount'].values,
                        color=[mapper.to_rgba(v) for v in _df['Performance'].values])
            if _df['ResponseCount'].max()>1:
                ax.text(_df['SnapshotTime'].max(),_df['ResponseCount'].max(),'   '+name, {'fontsize':9})
        for i in ax.get_xmajorticklabels():
            i.set_rotation(90)
        ax.set_ylabel('ResponseCount')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        ax.set_yscale('log')
        mapper._A=[]
        cbar = fig.colorbar(mapper)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('Model Performance (AUC)')
        print('Maximum AUC across all models: %.2f' % df['Performance'].max())
    
    def plotResponseCountMatrix(self, lookback=15, fill_null_days=False, query:Union[str, dict]=None, figsize=(14, 10)) -> plt.figure:
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
        table = 'modelData'
        multi_snapshot = True
        required_columns = {'ModelID', 'ModelName', 'SnapshotTime', 'ResponseCount'}
        df = self._subset_data(table, required_columns, query=query, multi_snapshot=multi_snapshot)
        assert lookback < df['SnapshotTime'].nunique(), f"Lookback ({lookback}) cannot be larger than the number of snapshots {df['SnapshotTime'].nunique()}"

        f, ax = plt.subplots(figsize=figsize)
        annot_df, heatmap_df = self._create_heatmap_df(df, lookback, query=None, fill_null_days=fill_null_days)
        heatmap_df = heatmap_df.reset_index().merge(df[['ModelID', 'ModelName']].drop_duplicates(), 
                                                    on='ModelID', how='left').drop('ModelID', axis=1).set_index('ModelName')
        annot_df = annot_df.reset_index().merge(df[['ModelID', 'ModelName']].drop_duplicates(), 
                                                    on='ModelID', how='left').drop('ModelID', axis=1).set_index('ModelName')
        myColors = ['r', 'orange', 'w']
        colorText = ['Decreased', 'No Change', 'Increased or NA']
        cmap = colors.ListedColormap(myColors)
        sns.heatmap(heatmap_df.T, annot=annot_df.T, mask=annot_df.T.isnull(), ax=ax,
                    linewidths=0.5, fmt='.0f', cmap=cmap, vmin=-1, vmax=1, cbar=False)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        patches = [mpatches.Patch(color=myColors[i], label=colorText[i]) for i in range(len(myColors)) ]

        legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('lightgrey')

    def plotSuccessRateOverTime(self, day_interval:int=7, query:Union[str, dict]=None, figsize:tuple=(16, 10)) -> plt.figure:
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
        table = 'modelData'
        multi_snapshot = True
        required_columns = {'ModelID', 'ModelName', 'SnapshotTime', 'ResponseCount', 'SuccessRate'}
        df = self._subset_data(table, required_columns, query, multi_snapshot=multi_snapshot)
        assert day_interval < df['SnapshotTime'].nunique(), f"Day interval ({day_interval}) cannot be larger than the number of snapshots ({df['SnapshotTime'].nunique()})"
        
        fig, ax = plt.subplots(figsize=figsize)
        df['SuccessRate'] *= 100
        sns.pointplot(x='SnapshotTime', y='SuccessRate', data=df, hue='ModelID', marker="o", ax=ax)
        print('Pointplot generated')
        modelnames = df[['ModelID', 'ModelName']].drop_duplicates().set_index('ModelID').to_dict()['ModelName']
        print('Modelnames generated')
        handles, labels = ax.get_legend_handles_labels()
        newlabels = [modelnames[i] for i in labels]
        ax.legend(handles, newlabels, bbox_to_anchor=(1.05, 1),loc=2)
        #ax.legend(bbox_to_anchor=(1.05, 1),loc=2)
        ax.set_ylabel('Success Rate (%)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        print("Setting rotations")
        for i in ax.get_xmajorticklabels():
            i.set_rotation(90)

    def plotPropositionSuccessRates(self, query:Union[str, dict]=None, figsize:tuple=(12, 8)) -> plt.figure:
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
        table = 'modelData'
        last = True
        required_columns = {'ModelName', 'SuccessRate'}
        df = self._subset_data(table, required_columns, query, last=last)

        f, ax = plt.subplots(figsize=figsize)
        df['SuccessRate'] *= 100
        bplot = sns.barplot(x='SuccessRate', y='ModelName', data=df.sort_values('SuccessRate', ascending=False), ax=ax)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        for p in bplot.patches:
            bplot.annotate("{:0.2%}".format(p.get_width()/100.0), (p.get_width(), p.get_y()+p.get_height()/2),
                xytext=(3, 0), textcoords="offset points", ha='left', va='center')

    def plotScoreDistribution(self, show_zero_responses:bool=False, query:Union[str, dict]=None, figsize:tuple=(14, 10)) -> plt.figure:
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
        table = 'combinedData'
        required_columns = {'PredictorName', 'ModelName', 'BinIndex', 'BinSymbol', 'BinResponseCount', 'BinPropensity'}
        df = self._subset_data(table, required_columns, query)

        df = df[df['PredictorName']=='Classifier']
        if df.ModelName.nunique() > 10:
            if input(f"""WARNING: you are about to create {df.index.nunique()} plots because there are that many models. 
            This will take a while, and will probably slow down your system. Are you sure? Type 'Yes' to proceed.""") != 'Yes': 
                print("Cancelling. Set your 'query' parameter more strictly to generate fewer images")
                return None
        for model in df.ModelName.unique():
            _df = df[df.ModelName==model]
            if not show_zero_responses:
                if _df['BinResponseCount'].nunique() == [0]:
                    continue
            name = _df.ModelName.unique()[0]
            self.distribution_graph(_df, 'Model name: '+name, figsize)

    def plotPredictorBinning(self, predictors:list=None, modelids:str=None, query:Union[str, dict]=None, figsize:tuple=(10, 5)) -> plt.figure:
        """ Show predictor graphs for a given model
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

        table = 'combinedData'
        last = True
        required_columns = {'PredictorName', 'ModelName', 'BinIndex', 'BinSymbol', 'BinResponseCount', 'BinPropensity'}
        df = self._subset_data(table, required_columns, query, last=last).reset_index()
        if predictors:
            df = df.query(f"PredictorName == {predictors}")
        if modelids is not None: 
            df = df.query(f"ModelID == {modelids}")
        model_name = df['ModelName'].unique()[0]
        for pred in df['PredictorName'].unique():
            _df = df.query(f'PredictorName == {[pred]}')
            title = 'Model name: '+model_name+'\n Predictor name: '+pred
            self.distribution_graph(_df, title, figsize)

    def plotPredictorPerformance(self, query:Union[str, dict]=None, figsize:tuple=(6, 12)) -> plt.figure:
        """ Shows a box plot of predictor performance

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
        table = 'combinedData'
        last = True
        required_columns = {'PredictorName', 'ModelName', 'BinIndex', 'BinSymbol', 'BinResponseCount', 'BinPropensity', 'PerformanceBin', 'Type'}
        df = self._subset_data(table, required_columns, query, last=last)


        fig, ax = plt.subplots(figsize=figsize)
        df = df[df['PredictorName']!='Classifier'].reset_index(drop=True)
        df['Legend'] = pd.Series([i.split('.')[0] if len(i.split('.'))>1 else 'Primary' for i in df['PredictorName']])
        order = df.groupby('PredictorName')['PerformanceBin'].mean().fillna(0).sort_values()[::-1].index
        sns.boxplot(x='PerformanceBin', y='PredictorName', data=df, order=order, ax=ax)
        ax.set_xlabel('Predictor Performance')
        ax.set_ylabel('Predictor Name')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

        norm = colors.Normalize(vmin=0, vmax=len(df['Legend'].unique())-1)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.gnuplot_r)
        cl_dict = dict(zip(df['Legend'].unique(), [mapper.to_rgba(v) for v in range(len(df['Legend'].unique()))]))
        value_dict = dict(df[['PredictorName', 'Legend']].drop_duplicates().values)
        type_dict = dict(df[['PredictorName', 'Type']].drop_duplicates().values)
        boxes = ax.artists
        for i in range(len(boxes)):
            boxes[i].set_facecolor(cl_dict[value_dict[order[i]]])
            if type_dict[order[i]].lower()=='symbolic':
                boxes[i].set_linestyle('--')

        lines = [Line2D([], [], label='Numeric', color='black', linewidth=1.5),
                    Line2D([], [], label='Symbolic', color='black', linewidth=1.5, linestyle='--')]
        legend_type = plt.legend(handles=lines, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5,
                    frameon=True, title='Predictor Type \n')
        patches = [mpatches.Patch(color=j, label=i) for i, j in cl_dict.items()]
        legend = plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.5,
                    frameon=True, title='Predictor Source \n')
        plt.gca().add_artist(legend_type)
        legend._legend_box.align = "left"
        legend_type._legend_box.align = "left"

    def plotPredictorPerformanceHeatmap(self, query:Union[str, dict]=None, figsize=(14, 10)) -> plt.figure:
        """ Shows a heatmap plot of predictor performance across models

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
        table = 'combinedData'
        last = True
        required_columns = {'PredictorName', 'ModelName', 'PerformanceBin'}
        df = self._subset_data(table, required_columns, query, last=last)

        df = df[df['PredictorName']!='Classifier'].reset_index(drop=True)
        pivot_df = df.drop_duplicates().pivot_table(
            index='ModelName', columns='PredictorName', values='PerformanceBin')
        order = list(df[[
            'ModelName', 'PredictorName', 'PerformanceBin']].drop_duplicates().groupby(
            'PredictorName')['PerformanceBin'].mean().fillna(0).sort_values()[::-1].index)
        pivot_df = pivot_df[order]*100.0
        x_order = list(df[[
            'ModelName', 'PredictorName', 'PerformanceBin']].drop_duplicates().groupby(
            'ModelName')['PerformanceBin'].mean().fillna(0).sort_values()[::-1].index)
        df_g = pivot_df.reindex(x_order)
        cmap = colors.LinearSegmentedColormap.from_list(
            'mycmap', [(0/100.0, 'red'), (20/100.0, 'green'),
                        (90/100.0, 'white'), (100/100.0, 'white')])
        f, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_g.fillna(50).T, ax=ax, cmap=cmap, annot=True, fmt='.2f', vmin=50, vmax=100)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

    def plotImpactInfluence(self, ModelID:str=None, query:Union[str, dict]=None, figsize:tuple=(12, 5)) -> plt.figure:
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
        table = 'combinedData'
        last = True
        required_columns = {'ModelID', 'PredictorName', 'ModelName', 'PerformanceBin', 'BinPositivesPercentage', 'BinNegativesPercentage', 'BinResponseCountPercentage', 'Issue', 'Group', 'Channel', 'Direction'}
        df = self._subset_data(table, required_columns, query, last=last).reset_index()
        df = self._calculate_impact_influence(df, ModelID=ModelID)[[
            'ModelID', 'PredictorName', 'Impact(%)', 'Influence(%)']].set_index(
            ['ModelID', 'PredictorName']).stack().reset_index().rename(columns={'level_2':'metric', 0:'value'})
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='PredictorName', y='value', data=df, hue='metric', ax=ax)
        ax.legend(bbox_to_anchor=(1.01, 1),loc=2)
        ax.set_ylabel('Metrics')