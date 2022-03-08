from typing import NoReturn, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np
from copy import deepcopy

import plotly.express as px
import plotly.graph_objects as go

from plot_base import VizBase

class ADMVisualisations(VizBase):

    def plotPerformanceSuccessRateBubbleChart(  self, 
                                                last=True, 
                                                add_bottom_left_text=True, 
                                                to_html=False, 
                                                file_title:str=None,
                                                file_path:str=None,
                                                query:Union[str, dict]=None, 
                                                show_each=True, 
                                                facets=None, 
                                                **kwargs):
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

        table = 'modelData'
        required_columns = {'Performance', 'SuccessRate', 'ResponseCount', 'ModelName'}.union(set(self.facets))
        df = self._subset_data(table=table, required_columns=required_columns, query=query, last=last)
        if isinstance(facets, str) or facets is None: facets = [facets]

        figlist = []
        bubble_size = kwargs.pop('bubble_size', 1)
        for facet in facets:
            title = 'over all models' if facet == None else f'per {facet}'
            df1 = deepcopy(df)
            df1[['Performance', 'SuccessRate']] = df1[['Performance', 'SuccessRate']].apply(lambda x : round(x*100,kwargs.pop('round', 5))) #fix to use .loc
            df1 = df1.reset_index()
            
            fig = px.scatter(df1, 
                            x='Performance', 
                            y='SuccessRate', 
                            color='Performance', 
                            size='ResponseCount', 
                            facet_col=facet, 
                            facet_col_wrap=5, 
                            hover_name='ModelName',
                            hover_data=['ModelID']+self.facets,
                            title=f'Bubble Chart {title} {kwargs.get("title","")}',
                            color_continuous_scale='Bluered',
                            template='none')
            fig.update_traces(marker=dict(line=dict(color='black')))

            if query != None: fig.layout.title.text += f"<br><sup>Query: {query}</sup>"

            if add_bottom_left_text:
                if len(fig.layout.annotations) > 0:
                    for i in range(0,len(fig.layout.annotations)):
                        oldtext = fig.layout.annotations[i].text.split('=')
                        subset = df1[df1[oldtext[0]] == oldtext[1]]
                        bottomleft = len(subset.query('Performance == 50 & (SuccessRate.isnull() | SuccessRate == 0)', engine='python'))
                        newtext = f"{len(subset)} models: {bottomleft} ({round(bottomleft/len(subset)*100, 2)}%) at (50,0)"
                        fig.layout.annotations[i].text += f"<br><sup>{newtext}</sup>"
                        fig.data[i].marker.size*=bubble_size

                else:
                    bottomleft = len(df1.query('Performance == 50 & (SuccessRate.isnull() | SuccessRate == 0)', engine='python'))
                    newtext = f"{len(df1)} models: {bottomleft} ({round(bottomleft/len(df1)*100, 2)}%) at (50,0)"
                    fig.layout.title.text += f"<br><sup>{newtext}</sup>"
                    fig.data[0].marker.size*=bubble_size

            filename = f'Bubble_{title}' if file_title==None else f"Bubble_{file_title}_{title}"
            file_path = 'findings' if file_path == None else file_path


            if to_html: fig.write_html(f'{file_path}/{filename}.html')
            
            figlist.append(fig)
            if show_each: fig.show()

        return figlist if len(figlist) > 1 else figlist[0]
        
    # def plotPerformanceAndSuccessRateOverTime(self, day_interval:int=7, query:Union[str, dict]=None, figsize:tuple=(16, 10)):
    #     """Shows responses and performance of models over time
    #     Reponses are on the y axis and the performance of the model is indicated by heatmap.
    #     x axis is date

    #     Parameters
    #     ----------
    #     day_interval : int
    #         Interval of tick labels along x axis
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
    #     required_columns = {'ModelName', 'SnapshotTime', 'ModelID', 'Performance', 'ResponseCount'}
    #     df = self._subset_data(table, required_columns, query, multi_snapshot=multi_snapshot)
        
    #     raise NotImplementedError("This visualisation is not yet implemented.")
    
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

    # def plotSuccessRateOverTime(self, day_interval:int=7, query:Union[str, dict]=None, figsize:tuple=(16, 10)):
    #     """Shows success rate of models over time
    #     Parameters
    #     ----------
    #     day_interval (int): 
    #         interval of tick labels along x axis
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
    #     required_columns = {'ModelID', 'ModelName', 'SnapshotTime', 'ResponseCount', 'SuccessRate'}
    #     df = self._subset_data(table, required_columns, query, multi_snapshot=multi_snapshot)
    #     assert day_interval < df['SnapshotTime'].nunique(), f"Day interval ({day_interval}) cannot be larger than the number of snapshots ({df['SnapshotTime'].nunique()})"
        
    #     raise NotImplementedError("This visualisation is not yet implemented.")

    # def plotPropositionSuccessRates(self, query:Union[str, dict]=None, figsize:tuple=(12, 8)):
    #     """Shows all latest proposition success rates
    #     A bar plot to show the success rate of all latest model instances (propositions)
    #     For reading simplicity, latest success rate is also annotated next to each bar

    #     Parameters
    #     ----------
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
    #     last = True
    #     required_columns = {'ModelName', 'SuccessRate'}
    #     df = self._subset_data(table, required_columns, query, last=last)

    #     raise NotImplementedError("This visualisation is not yet implemented.")

    # def plotScoreDistribution(self, show_zero_responses:bool=False, query:Union[str, dict]=None, figsize:tuple=(14, 10)):
    #     """Show score distribution similar to ADM out-of-the-box report
    #     Shows a score distribution graph per model. If certain models selected,
    #     only those models will be shown.
    #     the only difference between this graph and the one shown on ADM
    #     report is that, here the raw number of responses are shown on left y-axis whereas in
    #     ADM reports, the percentage of responses are shown
        
    #     Parameters
    #     ----------
    #     show_zero_responses:bool
    #         Whether to include bins with no responses at all
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
    #     table = 'combinedData'
    #     required_columns = {'PredictorName', 'ModelName', 'BinIndex', 'BinSymbol', 'BinResponseCount', 'BinPropensity'}
    #     df = self._subset_data(table, required_columns, query)

    #     df = df[df['PredictorName']=='Classifier']
    #     if df.ModelName.nunique() > 10:
    #         if input(f"""WARNING: you are about to create {df.index.nunique()} plots because there are that many models. 
    #         This will take a while, and will probably slow down your system. Are you sure? Type 'Yes' to proceed.""") != 'Yes': 
    #             print("Cancelling. Set your 'query' parameter more strictly to generate fewer images")
    #             return None
    #     raise NotImplementedError("This visualisation is not yet implemented.")

    # def plotPredictorBinning(self, predictors:list=None, modelids:str=None, query:Union[str, dict]=None, figsize:tuple=(10, 5)):
    #     """ Show predictor graphs for a given model
    #     For a given model (query) shows all its predictors' graphs. If certain predictors
    #     selected, only those predictor graphs will be shown
        
    #     Parameters
    #     ----------          
    #     predictors : list
    #         List of predictors to show their graphs, optional
    #     ModelID : str
    #         List of model IDs to subset on, optional
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

    #     table = 'combinedData'
    #     last = True
    #     required_columns = {'PredictorName', 'ModelName', 'BinIndex', 'BinSymbol', 'BinResponseCount', 'BinPropensity'}
    #     df = self._subset_data(table, required_columns, query, last=last).reset_index()
    #     if predictors:
    #         df = df[df['PredictorName'].isin(predictors)]
    #     if modelids is not None: 
    #         df = df[df['ModelID'].isin(modelids)] 
    #     model_name = df['ModelName'].unique()[0]
    #     raise NotImplementedError("This visualisation is not yet implemented.")

    def plotPredictorPerformance(self, 
                                top_n=0, 
                                to_html=False, 
                                file_title=None,
                                file_path=None, 
                                show_each=True, 
                                query=None, 
                                facets=None, 
                                **kwargs):
        """ Shows a box plot of predictor performance

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
        table = 'combinedData'
        last = True
        required_columns = {'Channel', 'PredictorName', 'PerformanceBin'}
        df = self._subset_data(table, required_columns, query, last=last)
        if isinstance(facets, str) or facets is None: facets = [facets]

        figlist = []
        if top_n>0:
            topn = df.groupby(['Channel', 'PredictorName'])['PerformanceBin'].mean().sort_values(ascending=False).head(top_n).index.get_level_values(1).tolist()
            df = df.query(f'PredictorName == {topn}')


        for facet in facets:
            title = 'over all models' if facet == None else f'per {facet}'
            

            df = df[df['PredictorName']!='Classifier'].reset_index(drop=True)
            df['Legend'] = pd.Series([i.split('.')[0] if len(i.split('.'))>1 else 'Primary' for i in df['PredictorName']])
            order = df.groupby('PredictorName')['PerformanceBin'].mean().fillna(0).sort_values(ascending=False)[::-1].index
            df = df.sort_values('PerformanceBin')
            
            fig = px.box(df, 
                         x='PerformanceBin', 
                         y='PredictorName', 
                         color='Legend', 
                         color_discrete_map={'Primary':'Yellow', 'Param':'Black'}, 
                         template='none', 
                         title=f"Predictor performance {title} {kwargs.get('title','')}", 
                         facet_col=facet, 
                         facet_col_wrap=5,
                         labels={'PerformanceBin':'Performance', 'PredictorName':'Predictor Name'})

            fig.update_yaxes(categoryorder='array', categoryarray=order, automargin=True, dtick=1)
            fig.update_traces(marker=dict(color='rgb(0,0,0)'), width=0.6)
            
            colors = ['rgb(14,94,165)','rgb(28,168,154)','rgb(254,183,85)','rgb(45,130,66)','rgb(252,136,72)','rgb(125,94,187)','rgb(252,139,130)','rgb(140,81,43)','rgb(175,161,156)']
            
            if len(fig.data)>9: colors = px.colors.qualitative.Alphabet 
            
            for i in range(len(fig.data)):
                fig.data[i].fillcolor = colors[i]
                
            fig.update_layout(boxgap=0, boxgroupgap=0, legend_title_text='Predictor type')

            if query != None: fig.layout.title.text += f"<br><sup>Query: {query}</sup>"
            filename = f'predictor_box_{title}' if file_title==None else f"predictor_box_{file_title}_{title}"
            file_path = 'findings' if file_path == None else file_path

            if to_html: fig.write_html(f'{file_path}/{filename}.html')
            
            figlist.append(fig)
            if show_each: fig.show()

        return figlist if len(figlist) > 1 else figlist[0]


    def plotPredictorPerformanceHeatmap(self, 
                                        top_n=0,
                                        to_html=False, 
                                        file_title=None,
                                        file_path=None, 
                                        show_each=True, 
                                        query=None, 
                                        facets=None, 
                                        **kwargs):
        """ Shows a heatmap plot of predictor performance across models

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
        table = 'combinedData'
        last = kwargs.get('last', True)
        required_columns = {'PredictorName', 'ModelName', 'PerformanceBin'}
        df = self._subset_data(table, required_columns, query, last=last)
        pivot_df = self.pivot_df(df)
        if top_n > 0: pivot_df = pivot_df.iloc[:,:top_n]
        if isinstance(facets, str) or facets is None: facets = [facets]

        figlist = []
        for facet in facets:
            title = 'over all models' if facet == None else f'per {facet}'
    
           
            from packaging import version
            import plotly
            assert version.parse(plotly.__version__)>=version.parse("5.5.0"), f"Visualisation requires plotly version 5.5.0 or later (you have version {plotly.__version__}): please upgrade to a newer version."
            
            fig = px.imshow(pivot_df.T, 
                            text_auto='.0%', 
                            aspect='auto', 
                            color_continuous_scale=[(0,'#d91c29'), (kwargs.get('midpoint', 0.01),'#F76923'), (kwargs.get('acceptable', 0.6)/2, '#20aa50'), (0.8,'#20aa50'), (1, '#0000FF')], 
                            facet_col=facet,
                            facet_col_wrap=5,
                            title=f'Top predictors {title} {kwargs.get("title","")}',
                            range_color=[0.5, 1])
            fig.update_yaxes(dtick=1, automargin=True)
            fig.update_xaxes(dtick=1, tickangle=kwargs.get('tickangle', None))
            

            if query != None: fig.layout.title.text += f"<br><sup>Query: {query}</sup>"
            filename = f'predictor_heatmap_{title}' if file_title == None else f"predictor_heatmap_{file_title}_{title}"
            file_path = 'findings' if file_path == None else file_path
            if to_html: fig.write_html(f'{file_path}/{filename}.html')
            figlist.append(fig)
            if show_each: fig.show()

        return figlist if len(figlist) > 1 else figlist[0]

    # def plotImpactInfluence(self, ModelID:str=None, query:Union[str, dict]=None, figsize:tuple=(12, 5)):
    #     """Calculate the impact and influence of a given model's predictors

    #     Parameters
    #     ----------
    #     modelID : str
    #         The selected model ID
    #     query : Union[str, dict]
    #         The query to supply to _apply_query
    #         If a string, uses the default Pandas query function
    #         Else, a dict of lists where the key is column name in the dataframe
    #         and the corresponding value is a list of values to keep in the dataframe
    #     figsize : tuple
    #         size of graph
        
    #     Returns
    #     -------
    #     plt.figure
    #     """
    #     table = 'combinedData'
    #     last = True
    #     required_columns = {'ModelID', 'PredictorName', 'ModelName', 'PerformanceBin', 'BinPositivesPercentage', 'BinNegativesPercentage', 'BinResponseCountPercentage', 'Issue', 'Group', 'Channel', 'Direction'}
    #     df = self._subset_data(table, required_columns, query, last=last).reset_index()
    #     df = self._calculate_impact_influence(df, ModelID=ModelID)[[
    #         'ModelID', 'PredictorName', 'Impact(%)', 'Influence(%)']].set_index(
    #         ['ModelID', 'PredictorName']).stack().reset_index().rename(columns={'level_2':'metric', 0:'value'})
    #     raise NotImplementedError("This visualisation is not yet implemented.")
    
    def plotResponseGain(self, by='Channel', to_html=False, file_title=None, file_path=None, show=True, query=None, **kwargs):
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
        
        table = 'modelData'
        last = True
        required_columns = {by, 'ResponseCount'}
        df = self._subset_data(table, required_columns, query, last=last)
        responseGainData = self.response_gain_df(df, by=by)
        title = "Cumulative Responses by Models"
        fig = px.line(  responseGainData, 
                        x='TotalModelsFraction', 
                        y='TotalResponseFraction', 
                        color=by,
                        labels={'TotalResponseFraction':'Percentage of Responses', 'TotalModelsFraction':'Percentage of Models'},
                        title=f'{title} {kwargs.get("title","")}<br><sup>by {by}</sup>',
                        template='none'
                        )
        fig.layout.yaxis.tickformat=',.0%'
        fig.layout.xaxis.tickformat=',.0%'
        if query != None: fig.layout.title.text += f"<br><sup>Query: {query}</sup>"
        filename = f'responseGain{title}' if file_title == None else f"responseGain{file_title}_{title}"
        file_path = 'findings' if file_path == None else file_path
        if to_html: fig.write_html(f'{file_path}/{filename}.html')
        if show: fig.show()
        return fig
    
    def plotModelsByPositives(self, by='Channel', to_html=False, file_title=None, file_path=None, show=True, query=None, **kwargs):
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

        table = 'modelData'
        last = True
        required_columns = {by, 'Positives'}
        df = self._subset_data(table, required_columns, query, last=last)
        modelsByPositives = self.models_by_positives_df(df.reset_index(), by=by)
        title = "Percentage of models vs number of positive responses"
        fig = px.line(  modelsByPositives.query('ModelCount>0'), 
                        x='PositivesBin', 
                        y='cumModels', 
                        color=by,
                        markers=True,
                        title=f'Percentage of models vs number of positive responses {kwargs.get("title","")}<br><sup>By {by}</sup>',
                        labels={'cumModels':'Percentage of Models', 'PositivesBin':'Positives'},
                        template='none',
                        category_orders={"PositivesBin":modelsByPositives['PositivesBin'].unique().tolist()}
                        )
        fig.layout.yaxis.tickformat=',.0%'
        if query != None: fig.layout.title.text += f"<br><sup>Query: {query}</sup>"
        filename = f'modelsByPositives{title}' if file_title == None else f"modelsByPositives{file_title}_{title}"
        file_path = 'findings' if file_path == None else file_path
        if to_html: fig.write_html(f'{file_path}/{filename}.html')
        if show: fig.show()
        return fig
    
    def plotTreeMap(self, color='performance_weighted', by='ModelID', value_in_text=True, midpoint=None, to_html=False, file_title=None, file_path=None, show=True, query=None, **kwargs):
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

        plotfacets = [px.Constant("All contexts")] + kwargs.get('facets', self.facets)

        summary = self.model_summary(by=by, query=query)
        
        
        plotsummary = summary[[(by, 'count'), (by, 'percentage_without_responses'), ('ResponseCount', 'sum'), ('SuccessRate', 'mean'), ('Performance', 'weighted_mean'), ('Positives', 'sum')]]
        plotsummary = plotsummary.reset_index()
        plotsummary.columns = self.facets + ['Model count', 'Percentage without responses', 'Response Count sum', 'Success Rate mean', 'Performance weighted mean', 'Positives sum']
        colorscale = ['#d91c29', '#F76923', '#20aa50']
        if 'OmniChannel' in plotsummary['Issue'].unique(): 
            print("WARNING: This plot does not work for OmniChannel models. For that reason, we filter those out by default.")
            plotsummary = plotsummary.query('Issue != "OmniChannel"')


        options = { 'responsecount':['Response Count sum', 'Model count', 'Responses per model, per context key combination', False, False, None],
                    'responsecount_log':['Response Count sum', 'Model count', 'Log responses per model, per context key combination', False, True, None],
                    'positives':['Positives sum', 'Model count', 'Positives per model, per context key combination', False, False, None],
                    'positives_log':['Positives sum', 'Model count', 'Log Positives per model, per context key combination', False, True, None],
                    'percentage_without_responses':['Percentage without responses', 'Model count', 'Percentage without responses, per context key combination', True, False, None],
                    'performance_weighted':['Performance weighted mean', 'Model count', 'Weighted mean performance, per context key combination', False, False, None],
                    'successrate':['Success Rate mean', 'Model count', 'Success rate, per context key combination', False, False, 0.5]}
        if isinstance(color, int): color=list(options.keys())[color]

        #manual override section, supply kwargs to override one or multiple of these options
        options[color]    = kwargs.get('override', options[color])
        options[color][0] = kwargs.get('color_col', options[color][0])
        options[color][1] = kwargs.get('groupby_col', options[color][1])
        options[color][2] = kwargs.get('title', options[color][2])
        options[color][3] = kwargs.get('reverse_scale', options[color][3])
        options[color][4] = kwargs.get('log', options[color][4])
        
        format = '%' if color in list(options.keys())[4:] else ''

        #log scale for colors
        if options[color][4]: options[color][0] = np.where(np.log(plotsummary[options[color][0]]) == -np.inf, 0, np.log(plotsummary[options[color][0]]))
        else: options[color][0] = plotsummary[options[color][0]]
        
        if midpoint is not None: options[color][5] = midpoint
        if options[color][5] is not None:
            midpoint = options[color][0].quantile(options[color][5])
            colorscale = [(0,'#d91c29'), (midpoint,'#F76923'), (1,'#20aa50')]
        elif color == 'performance_weighted':
            colorscale = [(0,'#d91c29'), (kwargs.get('midpoint', 0.01),'#F76923'), (kwargs.get('acceptable', 0.6), '#20aa50'), (0.8,'#20aa50'), (1,'#20aa50')]

        hover_data = {'Model count':':.d', 'Percentage without responses':':.0%', 'Response Count sum':':.d', 'Success Rate mean':':.3%', 'Performance weighted mean':':.0%', 'Positives sum':':.d'}

        fig = px.treemap(plotsummary, 
                    path=plotfacets, 
                    color=options[color][0], 
                    values=options[color][1], 
                    title=f'{options[color][2]}', 
                    hover_data=hover_data,
                    color_continuous_scale=colorscale,
                    )
        fig.update_coloraxes(reversescale=options[color][3])
        if query != None: fig.layout.title.text += f"<br><sup>Query: {query}</sup>"
        if value_in_text: 
            fig.update_traces(text = fig.data[0].marker.colors.round(3))
            fig.data[0].textinfo = 'label+text'
            if format == '%':
                fig.data[0].texttemplate='%{label}<br>%{text:.2%}'

        if kwargs.get('min_text_size', None) is not None:
            fig.update_layout(uniformtext_minsize=kwargs.get('min_text_size'), uniformtext_mode='hide')

        if to_html: 
            filename = f'modelTreemap{options[color][2]}' if file_title == None else f"modelTreemap{file_title}_{options[color][2]}"
            file_path = 'findings' if file_path == None else file_path
            fig.write_html(f'{file_path}/{filename}.html')
        
        
        if show: fig.show()
        return fig