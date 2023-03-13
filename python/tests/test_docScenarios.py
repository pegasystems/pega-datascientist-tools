"""
Testing that none of the docs examples produce errors.
"""


import pytest
import polars as pl
import sys

sys.path.append("python")
from pdstools import datasets


@pytest.fixture
def data():
    return datasets.CDHSample()


def test_GraphGallery(data):
    fig = data.plotModelsByPositives()
    fig = data.plotOverTime(query=pl.col("Channel") == "Web", by="Name")
    fig = data.plotPerformanceSuccessRateBubbleChart()
    fig = data.plotPredictorBinning(
        query=(pl.col("ModelID") == "08ca1302-9fc0-57bf-9031-d4179d400493")
        & pl.col("PredictorName").is_in(
            [
                "Customer.Age",
                "Customer.AnnualIncome",
                "IH.Email.Outbound.Accepted.pxLastGroupID",
            ]
        ),
        show_each=False,
    )
    fig = data.plotPredictorPerformance(top_n=30)
    fig = data.plotPredictorPerformanceHeatmap(top_n=20)
    fig = data.plotPropositionSuccessRates(query=pl.col("Channel") == "Web")
    fig = data.plotResponseGain()
    fig = data.plotScoreDistribution(show_each=False)
    fig = data.plotTreeMap()


def test_HelloPdstools():
    pass


def test_ExampleDataAnonymization():
    pass


def test_ExampleADMAnalysis():
    pass


def test_AnalyzingADMTrees():
    pass


def test_ValueFinder():
    pass
