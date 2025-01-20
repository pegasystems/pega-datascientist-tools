"""
Testing the functionality of the IH class
"""

import os
import pathlib

import polars as pl
import pytest
from pdstools import IH
import plotly.express as px
from plotly.graph_objs import Figure

def test_mockdata():
    ih = IH.from_mock_data()
    assert ih.data.collect().height > 100000 # interactions
    assert ih.data.collect().width == 13 # nr of IH properties in the sample data

    summary = ih.aggregates._summary_interactions().collect()
    assert summary.height == 100000

def test_plots():
    ih = IH.from_mock_data()
    assert isinstance(ih.plot.overall_gauges(condition="ExperimentGroup"), Figure)
    assert isinstance(ih.plot.response_count_tree_map(), Figure)
    assert isinstance(ih.plot.success_rate_tree_map(), Figure)
    assert isinstance(ih.plot.action_distribution(), Figure)
    assert isinstance(ih.plot.success_rate(), Figure)
    assert isinstance(ih.plot.response_count(), Figure)
    assert isinstance(ih.plot.model_performance_trend(), Figure)
    assert isinstance(ih.plot.model_performance_trend(by="ModelTechnique"), Figure)
