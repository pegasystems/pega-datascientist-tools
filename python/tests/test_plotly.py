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


@pytest.fixture
def Sample():
    return ADMDatamart(path="data")


def test_plotPerformanceSuccessRateBubbleChart(test):
    test.plotPerformanceSuccessRateBubbleChart(add_bottom_left_text=True)


def test_plotPerformanceSuccessRateBubbleChart_facetted(test):
    test.plotPerformanceSuccessRateBubbleChart(
        add_bottom_left_text=True, facets=["Issue", "Group"]
    )


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

def test_plotPredictorBinning_none_found(test):
    with pytest.raises(ValueError):
        test.plotPredictorBinning(
            modelid="unknown",
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
    for i in range(0, 6):
        test.plotTreeMap(i, value_in_text=True)


def test_plotTreeMap_options(test):
    test.plotTreeMap(
        "performance_weighted", log=True, min_text_size=11, value_in_text=True
    )


def test_plotTreeMap_options2(test):
    test.plotTreeMap("successrate", midpoint=0.5, min_text_size=11, value_in_text=True)


def test_post_plot(test):
    test.plotPerformanceSuccessRateBubbleChart(query='Channel=="Web"')


def test_bubble_cdhsample(Sample):
    Sample.plotPerformanceSuccessRateBubbleChart(add_bottom_left_text=True)


def test_overtime_cdhsample(Sample):
    Sample.plotOverTime()


def test_scoredist_cdhsample(Sample, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Yes")
    Sample.plotScoreDistribution()
