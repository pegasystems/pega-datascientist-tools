import pytest
import sys

sys.path.append("python")
from cdhtools import ADMDatamart, datasets


@pytest.fixture
def test():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart(
        path="data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip",
        context_keys=["Issue", "Group", "Channel"],
    )


def test_subset_data(test):
    table = "predictorData"
    required_columns = {"SnapshotTime", "PredictorName", "EntryType"}
    assert (
        test._subset_data(
            table=table,
            required_columns=required_columns,
            multi_snapshot=False,
            last=True,
            active_only=True,
        )
        is not None
    )


def test_subset_data_table_not_present(test):
    table = "predictorData"
    required_columns = {"SnapshotTime", "PredictorName", "EntryType"}
    example = ADMDatamart("data", predictor_filename=None)
    with pytest.raises(example.NotApplicableError):
        example._subset_data(
            table=table,
            required_columns=required_columns,
            multi_snapshot=False,
            last=True,
            active_only=True,
        )


def test_subset_data_not_enough_snapshots(test):
    table = "modelData"
    required_columns = {"SnapshotTime", "ModelID", "ModelName"}
    query = 'SnapshotTime == "2021-02-01 11:50:42.258000+00:00"'
    with pytest.raises(test.NotApplicableError):
        test._subset_data(
            table=table,
            required_columns=required_columns,
            multi_snapshot=True,
            last=False,
            query=query,
            active_only=True,
        )


def test_PerformanceSuccessRateBubbleChart_data(test):
    df = test.plotPerformanceSuccessRateBubbleChart(return_df=True)
    assert df.shape == (10, 8)
    assert "Configuration" not in df.columns
    assert set(df.columns) == {
        "Performance",
        "ModelID",
        "ModelName",
        "Channel",
        "Issue",
        "SuccessRate",
        "Group",
        "ResponseCount",
    }
    assert set(test.context_keys).issubset(set(df.columns))
    assert list(df.loc[1, ["Performance", "SuccessRate"]]) == [64.3397, 29.0396]


def test_PerformanceSuccessRateBubbleChart_data(test):
    df = test.plotPerformanceSuccessRateBubbleChart(
        return_df=True, plotting_engine="matplotlib"
    )


def test_PerformanceAndSuccessRateOverTime_data(test):
    df = test.plotPerformanceAndSuccessRateOverTime(return_df=True)
    assert df.SnapshotTime.nunique() > 1
    # TODO: finish.


def test_OverTime_data(test):
    df = test.plotOverTime(return_df=True)


def test_ResponseCountMatrix_data(test):
    df = datasets.CDHSample().plotResponseCountMatrix(return_df=True, lookback=5)


def test_ResponseCountMatrix_data(test):
    df = test.plotResponseCountMatrix(return_df=True, lookback=2, fill_null_days=True)


def test_ResponseCountMatrix_data_lookback_too_big(test):
    with pytest.raises(AssertionError):
        df = test.plotResponseCountMatrix(return_df=True, lookback=20)


def test_ResponseCountMatrix_data_lookback_too_big(test):
    with pytest.raises(ValueError):
        df = test.plotResponseCountMatrix(return_df=True, lookback=0)


def test_PropositionSuccessRates_data(test):
    df = test.plotPropositionSuccessRates(return_df=True)


def test_ScoreDistribution_data(test, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Yes")
    assert test.plotScoreDistribution(return_df=True) is not None


def test_ScoreDistribution_no(test, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "No")
    assert test.plotScoreDistribution(return_df=True) == None


def test_PredictorBinning_data(test):
    df = test.plotPredictorBinning(
        return_df=True,
        modelid="03053052-28c8-5af8-901e-87e857979b3c",
        predictors=["AGE"],
    )


def test_PredictorPerformance_data(test):
    df = test.plotPredictorPerformance(return_df=True, top_n=5)


def test_PredictorPerformanceHeatmap_data(test):
    df = test.plotPredictorPerformanceHeatmap(return_df=True, top_n=5)


def test_ImpactInfluence_data(test):
    df = test.plotImpactInfluence(return_df=True)


def test_ImpactInfluence_data(test):
    df = test.plotImpactInfluence(
        return_df=True, ModelID="03053052-28c8-5af8-901e-87e857979b3c"
    )


def test_ResponseGain_data(test):
    df = test.plotResponseGain(return_df=True)


def test_ModelsByPositives_data(test):
    df = test.plotModelsByPositives(return_df=True)


def test_TreeMap_data(test):
    df = test.plotTreeMap(4, return_df=True, midpoint=0.6)

def test_TreeMap_2(test):
    df = test.plotTreeMap('SuccessRate', plotting_engine = 'matplotlib')