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
import seaborn
import matplotlib


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


def mpl_checks(fun, **args):
    df = fun(return_df=True, **args)
    plt = fun(**args)
    if isinstance(plt, seaborn.axisgrid.FacetGrid):
        assert df.equals(plt.data)
    else:
        assert hasattr(plt, "plot")


def test_plotPerformanceSuccessRateBubbleChart(test):
    mpl_checks(test.plotPerformanceSuccessRateBubbleChart, annotate=True)

def test_plotPerformanceSuccessRateBubbleChart_not_annotated(test):
    mpl_checks(test.plotPerformanceSuccessRateBubbleChart)


def test_plotPerformanceAndSuccessRateOverTime(test):
    mpl_checks(test.plotPerformanceAndSuccessRateOverTime)


def test_plotResponseCountMatrix(test):
    mpl_checks(test.plotResponseCountMatrix, lookback=2)


def test_plotOverTime(test):
    mpl_checks(test.plotOverTime, day_interval=2)


def test_plotPropositionSuccessRates(test):
    mpl_checks(test.plotPropositionSuccessRates)


def test_plotScoreDistribution(test, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Yes")
    test.plotScoreDistribution()


def test_plotPredictorBinning(test):
    test.plotPredictorBinning(
        modelid="03053052-28c8-5af8-901e-87e857979b3c",
        predictors=["AGE"],
    )


def test_PredictorPerformance(test):
    mpl_checks(test.plotPredictorPerformance, top_n=5)


def test_plotPredictorPerformanceHeatmap(test):
    mpl_checks(test.plotPredictorPerformanceHeatmap, top_n=5)


def test_plotImpactInfluence(test):
    mpl_checks(test.plotImpactInfluence)