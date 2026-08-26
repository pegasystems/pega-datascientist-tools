"""Testing the functionality of the ADMDatamart active_ranges function"""

import pathlib

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.utils import cdh_utils
from polars.testing import assert_frame_equal

basePath = pathlib.Path(__file__).parent.parent.parent.parent


@pytest.fixture
def sample():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data/active_range/CDHSample-Pega8",
    )


def test_active_ranges_basic(sample):
    """Test the basic functionality of active_ranges."""
    # Test with no model_id (all models)
    ar = sample.active_ranges().collect()

    # Check that the expected columns are present
    expected_columns = [
        "ModelID",
        "AUC_Datamart",
        "AUC_FullRange",
        "AUC_ActiveRange",
        "AUC_ActiveRange_CI_Estimate",
        "AUC_ActiveRange_CI_Variance",
        "AUC_ActiveRange_CI_Lower",
        "AUC_ActiveRange_CI_Upper",
        "AUC_ActiveRange_CI_Safe_Lower",
        "AUC_ActiveRange_CI_Safe_Upper",
        "AUC_ActiveRange_CI_Available",
        "AUC_ActiveRange_CI_Reason",
        "AUC_ActiveRange_CI_Scope",
        "AUC_ActiveRange_CI_IncludesModelFitUncertainty",
        "Bins",
        "nActivePredictors",
        "classifierLogOffset",
        "sumMinLogOdds",
        "sumMaxLogOdds",
        "score_min",
        "score_max",
        "idx_min",
        "idx_max",
    ]
    for col in expected_columns:
        assert col in ar.columns

    # Check that the number of rows matches the expected number of models
    assert ar.height == 4

    # Check that the values are within expected ranges
    assert all(0 <= auc <= 1 for auc in ar["AUC_Datamart"])
    assert all(0 <= auc <= 1 for auc in ar["AUC_FullRange"])
    assert all(0 <= auc <= 1 for auc in ar["AUC_ActiveRange"])
    assert all(variance is None or variance >= 0 for variance in ar["AUC_ActiveRange_CI_Variance"])
    for row in ar.iter_rows(named=True):
        if row["AUC_ActiveRange_CI_Available"]:
            expected_lower, expected_upper = cdh_utils.safe_range_interval(
                row["AUC_ActiveRange_CI_Lower"],
                row["AUC_ActiveRange_CI_Upper"],
            )
            assert row["AUC_ActiveRange_CI_Safe_Lower"] == pytest.approx(expected_lower)
            assert row["AUC_ActiveRange_CI_Safe_Upper"] == pytest.approx(expected_upper)
    assert all(flag in (True, False) for flag in ar["AUC_ActiveRange_CI_Available"])
    assert all(bins > 0 for bins in ar["Bins"])
    assert all(n >= 0 for n in ar["nActivePredictors"])
    assert all(idx_min >= 0 for idx_min in ar["idx_min"])
    assert all(idx_max > 0 for idx_max in ar["idx_max"])
    assert all(idx_max >= idx_min for idx_min, idx_max in zip(ar["idx_min"], ar["idx_max"], strict=False))


def test_active_ranges_uses_native_polars_expressions(sample):
    """Keep active-range calculations visible to the Polars optimizer."""
    assert "python_udf" not in sample.active_ranges().explain()


def test_active_ranges_streaming_matches_default_engine(sample):
    """Return identical values through the streaming engine."""
    query = sample.active_ranges()

    assert_frame_equal(
        query.collect(engine="streaming"),
        query.collect(),
        check_row_order=True,
        check_column_order=True,
    )


def test_active_ranges_single_model(sample):
    """Test active_ranges with a single model ID."""
    # Get all model IDs
    all_models = sample.active_ranges().collect()
    model_id = all_models["ModelID"][0]

    # Test with a single model_id as string
    ar_single = sample.active_ranges(model_id).collect()
    assert ar_single.height == 1
    assert ar_single["ModelID"][0] == model_id

    # Test with a single model_id in a list
    ar_single_list = sample.active_ranges([model_id]).collect()
    assert ar_single_list.height == 1
    assert ar_single_list["ModelID"][0] == model_id

    # Compare results from both methods
    for col in ar_single.columns:
        assert ar_single[col][0] == ar_single_list[col][0]


def test_active_ranges_multiple_models(sample):
    """Test active_ranges with multiple model IDs."""
    # Get all model IDs
    all_models = sample.active_ranges().collect()
    model_ids = all_models["ModelID"][:2].to_list()

    # Test with multiple model_ids
    ar_multiple = sample.active_ranges(model_ids).collect()
    assert ar_multiple.height == 2
    assert set(ar_multiple["ModelID"]) == set(model_ids)


def test_active_ranges_nonexistent_model(sample):
    """Test active_ranges with a nonexistent model ID."""
    # Test with a nonexistent model_id
    ar_nonexistent = sample.active_ranges("nonexistent_model_id").collect()
    assert ar_nonexistent.height == 0


def test_active_ranges_edge_cases():
    """Test active_ranges with edge cases."""
    # Test with empty data
    dm_empty = ADMDatamart(model_df=None, predictor_df=None)
    with pytest.raises(ValueError, match="requires predictor data"):
        dm_empty.active_ranges().collect()

    # Test with model data but no predictor data
    test_data_mdls = f"{basePath}/data/active_range/all_1_mdls.csv"
    dm_no_predictors = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls),
        predictor_df=None,
    )
    with pytest.raises(ValueError, match="requires predictor data"):
        dm_no_predictors.active_ranges().collect()


def test_active_ranges_pega7():
    """Test active_ranges with Pega 7 data."""
    test_data_mdls = f"{basePath}/data/active_range/dmModels.csv.gz"
    test_data_preds = f"{basePath}/data/active_range/dmPredictors.csv.gz"
    dm = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls),
        predictor_df=pl.scan_csv(test_data_preds),
    )

    # Test specific model with known values
    model_id = "664cc653-279f-54ae-926f-694652d89a54"
    ar = dm.active_ranges(model_id).collect()

    # Check that the values match expected values
    assert ar["idx_min"].item() == 6
    assert ar["idx_max"].item() == 7
    assert round(ar["AUC_Datamart"].item(), 6) == 0.760333
    assert round(ar["AUC_FullRange"].item(), 6) == 0.760333
    assert round(ar["AUC_ActiveRange"].item(), 6) == 0.5

    # Check that AUC_ActiveRange is different from AUC_FullRange when idx_min and idx_max don't span the full range
    assert ar["AUC_ActiveRange"].item() != ar["AUC_FullRange"].item()

    # Test another model where AUC_ActiveRange equals AUC_FullRange
    model_id = "4574f1fd-13a7-5703-bf38-9374641f370f"
    ar = dm.active_ranges(model_id).collect()
    assert round(ar["AUC_ActiveRange"].item(), 6) == round(
        ar["AUC_FullRange"].item(),
        6,
    )


def test_active_ranges_empty_classifier_slice_returns_unavailable_ci(sample, monkeypatch):
    """Return unavailable CI metadata when no classifier bin is active."""
    model_id = sample._require_predictor_data().select("ModelID").unique().collect()["ModelID"][0]
    original_min_max_scores = ADMDatamart._minMaxScoresPerModel

    def scores_below_classifier_bounds(cls, data):
        return original_min_max_scores(data).with_columns(
            pl.lit(1e9).alias("score_min"),
            pl.lit(-1e9).alias("score_max"),
        )

    monkeypatch.setattr(
        ADMDatamart,
        "_minMaxScoresPerModel",
        classmethod(scores_below_classifier_bounds),
    )

    result = sample.active_ranges(model_id).collect()

    assert result["AUC_ActiveRange"].item() is None
    assert result["AUC_ActiveRange_CI_Lower"].item() is None
    assert result["AUC_ActiveRange_CI_Upper"].item() is None
    assert result["AUC_ActiveRange_CI_Safe_Lower"].item() is None
    assert result["AUC_ActiveRange_CI_Safe_Upper"].item() is None
    assert result["AUC_ActiveRange_CI_Available"].item() is False
    assert result["AUC_ActiveRange_CI_Reason"].item() == "empty_active_range"


def test_active_ranges_missing_score_range_returns_unavailable_ci(sample, monkeypatch):
    """Return unavailable CI metadata when predictor scores are missing."""
    model_id = sample._require_predictor_data().select("ModelID").unique().collect()["ModelID"][0]
    original_min_max_scores = ADMDatamart._minMaxScoresPerModel

    def scores_with_missing_range(cls, data):
        return original_min_max_scores(data).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("score_min"),
            pl.lit(None, dtype=pl.Float64).alias("score_max"),
        )

    monkeypatch.setattr(
        ADMDatamart,
        "_minMaxScoresPerModel",
        classmethod(scores_with_missing_range),
    )

    result = sample.active_ranges(model_id).collect()

    assert result["AUC_ActiveRange"].item() is None
    assert result["AUC_ActiveRange_CI_Available"].item() is False
    assert result["AUC_ActiveRange_CI_Reason"].item() == "missing_score_range"


def _agb_datamart(sample, *, duplicate_classifier_rows=False):
    model_ids = (
        sample._require_model_data()
        .filter(pl.col("Configuration") == "OmniAdaptiveModel")
        .select("ModelID")
        .unique()
        .collect()["ModelID"]
        .to_list()
    )
    model_data = sample._require_model_data().with_columns(
        ModelTechnique=pl.when(pl.col("ModelID").is_in(model_ids))
        .then(pl.lit("GradientBoost"))
        .otherwise(pl.col("ModelTechnique")),
    )
    predictor_data = sample._require_predictor_data().with_columns(
        BinPositives=pl.when(
            pl.col("ModelID").is_in(model_ids)
            & (pl.col("EntryType") == "Classifier")
            & (
                pl.col("BinIndex")
                == pl.col("BinIndex").filter(pl.col("EntryType") == "Classifier").min().over("ModelID")
            ),
        )
        .then(0.0)
        .otherwise(pl.col("BinPositives")),
        BinNegatives=pl.when(
            pl.col("ModelID").is_in(model_ids)
            & (pl.col("EntryType") == "Classifier")
            & (
                pl.col("BinIndex")
                == pl.col("BinIndex").filter(pl.col("EntryType") == "Classifier").min().over("ModelID")
            ),
        )
        .then(0.0)
        .otherwise(pl.col("BinNegatives")),
    )
    if duplicate_classifier_rows:
        duplicates = predictor_data.filter(
            pl.col("ModelID").is_in(model_ids),
            pl.col("EntryType") == "Classifier",
        ).with_columns(BinIndex=pl.col("BinIndex") + 100)
        predictor_data = pl.concat([predictor_data, duplicates])

    return (
        ADMDatamart(
            model_df=model_data,
            predictor_df=predictor_data,
            extract_pyname_keys=False,
        ),
        model_ids,
    )


def test_active_ranges_uses_occupied_bins_and_configuration_ci_for_agb(sample, monkeypatch):
    """Pool AGB validation counts without applying Naive Bayes score math."""
    datamart, model_ids = _agb_datamart(sample)
    original_min_max_scores = ADMDatamart._minMaxScoresPerModel

    def assert_agb_excluded(cls, data):
        assert data.select("ModelID").unique().collect().height == 0
        return original_min_max_scores(data)

    monkeypatch.setattr(
        ADMDatamart,
        "_minMaxScoresPerModel",
        classmethod(assert_agb_excluded),
    )

    result = datamart.active_ranges(model_ids).collect().sort("ModelID")
    pooled_bins = (
        datamart._require_predictor_data()
        .filter(
            pl.col("ModelID").is_in(model_ids),
            pl.col("EntryType") == "Classifier",
            (pl.col("BinPositives") + pl.col("BinNegatives")) > 0,
        )
        .group_by("BinLowerBound")
        .agg(
            pl.col("BinPositives").sum(),
            pl.col("BinNegatives").sum(),
        )
        .collect()
    )
    expected_ci = cdh_utils.auc_ci_from_bincounts(
        pooled_bins["BinPositives"],
        pooled_bins["BinNegatives"],
    )

    assert result["idx_min"].to_list() == [1, 1]
    assert result["score_min"].to_list() == [None, None]
    assert result["score_max"].to_list() == [None, None]
    assert result["AUC_ActiveRange_CI_Scope"].to_list() == ["configuration", "configuration"]
    assert result["AUC_ActiveRange_CI_IncludesModelFitUncertainty"].to_list() == [False, False]
    assert result["AUC_ActiveRange_CI_Estimate"].to_list() == pytest.approx([expected_ci["auc"]] * 2)
    assert result["AUC_ActiveRange_CI_Variance"].to_list() == pytest.approx([expected_ci["variance"]] * 2)
    assert result["AUC_ActiveRange_CI_Lower"].to_list() == pytest.approx([expected_ci["ci_lower"]] * 2)
    assert result["AUC_ActiveRange_CI_Upper"].to_list() == pytest.approx([expected_ci["ci_upper"]] * 2)


def test_active_ranges_ignores_malformed_duplicate_agb_classifier_rows(sample):
    """Keep AGB occupied ranges and pooled intervals invariant to duplicate bins."""
    baseline, model_ids = _agb_datamart(sample)
    duplicated, _ = _agb_datamart(sample, duplicate_classifier_rows=True)

    assert_frame_equal(
        duplicated.active_ranges(model_ids).collect().sort("ModelID"),
        baseline.active_ranges(model_ids).collect().sort("ModelID"),
        check_row_order=True,
        check_column_order=True,
    )


def test_active_ranges_uses_pooled_agb_volume_for_ci_availability(sample):
    """Keep availability and reason consistent with configuration-level counts."""
    datamart, model_ids = _agb_datamart(sample)
    starved_model = model_ids[0]
    datamart.predictor_data = datamart._require_predictor_data().with_columns(
        BinPositives=pl.when(
            (pl.col("ModelID") == starved_model) & (pl.col("EntryType") == "Classifier") & (pl.col("BinPositives") > 0),
        )
        .then(1.0 / pl.col("BinPositives").count().over("ModelID"))
        .otherwise(pl.col("BinPositives")),
        BinNegatives=pl.when(
            (pl.col("ModelID") == starved_model) & (pl.col("EntryType") == "Classifier") & (pl.col("BinNegatives") > 0),
        )
        .then(1.0 / pl.col("BinNegatives").count().over("ModelID"))
        .otherwise(pl.col("BinNegatives")),
    )

    result = datamart.active_ranges(model_ids).collect()

    assert result["AUC_ActiveRange_CI_Available"].to_list() == [True, True]
    assert result["AUC_ActiveRange_CI_Reason"].to_list() == [None, None]
