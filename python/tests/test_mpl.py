# Since we've now made the transition to plotly,
# i'll only do basic testing on the currently available
# matplotlib plots. The only checks here are that they
# don't give an error, which is still useful.
# Since the base plot class is fully tested, this
# should mean matplotlib will be okay for a while.

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
        plotting_engine="mpl",
    )


def test_plotPerformanceSuccessRateBubbleChart(test):
    test.plotPerformanceSuccessRateBubbleChart(annotate=True)


def test_plotPerformanceAndSuccessRateOverTime(test):
    test.plotPerformanceAndSuccessRateOverTime()


def test_plotResponseCountMatrix(test):
    test.plotResponseCountMatrix(lookback=2)


def test_plotOverTime(test):
    test.plotOverTime(day_interval=2)


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


def test_plotImpactInfluence(test):
    test.plotImpactInfluence()
