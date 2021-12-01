from abc import ABC, abstractmethod
import pandas as pd
import plotly
import matplotlib

class BasePlot(ABC):
    def __init__(self):
        if not all(hasattr(super(), i) for i in self.required_data):
            raise f"Can't use that visualisation because {[not hasattr(super(), i) for i in self.required_data]} is not found"
        if not all(self.required_columns in df.columns):
            raise f"Can't use that visualisation because columns {[self.required_columns not in df.columns]} not in the data."
    
    @abstractmethod
    def plotly_plot(self) -> plotly.graph_objs._figure.Figure:
        """Generate a plotly plot"""
    
    @abstractmethod
    def matplotlib_plot(self) -> matplotlib.axes._subplots.AxesSubplot:
        """Generate a matplotlib plot"""

class plotADMVarImp(BasePlot):
    def __init__(self, df: pd.DataFrame = None):
        self.required_data = {'modelData'}
        self.required_columns = {'PredictorName', 'Importance', 'Rank'}
        self.requires_multiple_snapshots = False
        
        if df is None or not all(required_columns in df.columns):
            df = get_var_imp()

class plotADMPerformanceSuccesRateBubbleChart(BasePlot):
    def __init__(self):
        self.required_data = {'modelData'}
        self.required_columns = {'Performance', 'Positives', 'ResponseCount', 'Name'}

class plotADMPerformanceSuccesRateBoxPlot(BasePlot):
    def __init__(self):
        self.required_data = {'modelData'}
        self.required_columns = {'Performance', 'Positives', 'ResponseCount', 'Name'}

class plotADMModelPerformanceOverTime(BasePlot):
    def __init__(self):
        self.required_data = {'modelData'}
        self.required_columns = {'SnapshotTime', 'Performance', 'ResponseCount'}

class plotADMModelSuccesRateOverTime(BasePlot):
    def __init__(self):
        self.required_data = {'modelData'}
        self.required_columns = {'SnapshotTime', 'Performance', 'Positives', 'ResponseCount'}

class plotADMPredictorPerformance(BasePlot):
    def __init__(self, nPredictors):
        self.required_data = {'modelData', 'predictorData'}
        self.required_columns = {'EntryType', 'Positives', 'Performance', 'SnapshotTime', 'Negatives', 'ResponseCount', 'PredictorType', 'ModelID'}
    
class plotADMPredictorPerformanceMatrix(BasePlot):
    def __init__(self, nPredictors):
        self.required_data = {'modelData', 'predictorData'}
        self.required_columns = {'EntryType', 'ModelID', 'Performance', 'ResponseCount', 'PredictorName'}

class plotADMPredictorPerformanceByGroup(BasePlot):
    def __init__(self):
        self.required_data = {'modelData', 'predictorData'}
        self.required_columns = {'EntryType', 'Performance', 'Positives', 'Negatives', 'ResponseCount', 'ModelID', 'PredictorName', 'PredictorType'}

class plotADMPropositionSuccesRates(BasePlot):
    def __init__(self):
        self.required_data = {'modelData'}
        self.required_columns = {'ModelID', 'SuccesRate', 'SnapshotTime', 'Positives', 'ResponseCount'}

class plotADMCumulativeGains(BasePlot):
    def __init__(self):
        self.required_data = {'predictorData'}
        self.required_columns = {'BinIndex', 'BinPositives', 'BinResponseCount'}

class plotADMCumulativeLift(BasePlot):
    def __init__(self):
        self.required_data = {'predictorData'}
        self.required_columns = {'BinIndex', 'BinPositives', 'BinResponseCount', 'BinNegatives'}

class plotADMBinning(BasePlot):
    def __init__(self):
        self.required_data = {'predictorData'}
        self.required_columns = {'BinIndex', 'BinPositives', 'BinResponseCount', 'BinSymbol', 'EntryType', 'BinNegatives', 'Positives', 'Negatives'}