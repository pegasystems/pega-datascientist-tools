import pytest
import sys

sys.path.append("python")
from cdhtools import ADMDatamart


@pytest.fixture
def test():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart(
        path="data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip",
        context_keys=["Issue", "Group", "Channel"],
        plotting_engine="plotly",
        verbose=True,
    )


def test_plotPerformanceSuccessRateBubbleChart(test):
    test.plotPerformanceSuccessRateBubbleChart(add_bottom_left_text=True)


def test_plotOverTime(test):
    test.plotOverTime(hide_legend=True)


def test_plotPropositionSuccessRates(test):
    test.plotPropositionSuccessRates()


def test_plotScoreDistribution(test, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Yes")
    test.plotScoreDistribution()


def test_plotPredictorBinning(test):
    test.plotPredictorBinning(
        modelid="03053052-28c8-5af8-901e-87e857979b3c",
        predictors=["AGE"],
    )


def test_PredictorPerformance(test):
    test.plotPredictorPerformance(top_n=5)


def test_plotPredictorPerformanceHeatmap(test):
    test.plotPredictorPerformanceHeatmap(top_n=5)


def test_plotResponseGain(test):
    test.plotResponseGain()


def test_plotModelsByPositives(test):
    test.plotModelsByPositives()


def test_plotTreeMap(test):
    test.plotTreeMap()


def test_plotTreeMap_options(test):
    test.plotTreeMap("performance_weighted", log=True, midpoint=0.55, min_text_size=11)
