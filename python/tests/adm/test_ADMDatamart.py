"""Testing the functionality of the ADMDatamart functions"""

import os
import pathlib

import polars as pl
import pytest
from pdstools import ADMDatamart

basePath = pathlib.Path(__file__).parent.parent.parent.parent


@pytest.fixture
def sample():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
    )


def test_cdh_sample_init(sample):
    assert sample


def test_cached_properties(sample: ADMDatamart):
    assert sample.unique_channel_direction == [
        "Email/Outbound",
        "SMS/Outbound",
        "Web/Inbound",
    ]

    assert sample.unique_channels == ["Email", "SMS", "Web"]

    assert sample.unique_configurations == ["OmniAdaptiveModel"]

    assert sample.unique_predictor_categories == ["Customer", "IH", "Param"]


def test_write_then_load(sample: ADMDatamart):
    modeldata_cache, predictordata_cache = sample.save_data("cache")

    assert ADMDatamart(
        model_df=pl.scan_ipc(modeldata_cache),
        predictor_df=pl.scan_ipc(predictordata_cache),
    )
    os.remove(modeldata_cache)
    os.remove(predictordata_cache)


def test_init_without_model_data(sample: ADMDatamart):
    modeldata_cache, predictordata_cache = sample.save_data("cache2")

    ADMDatamart(model_df=pl.scan_ipc(modeldata_cache))
    ADMDatamart(predictor_df=pl.scan_ipc(predictordata_cache))
    os.remove(modeldata_cache)
    os.remove(predictordata_cache)


def test_active_range_Pega7():
    # In this test data, the datamart is often wrong. This data pre-dates (and inspired) the fixes to the calculations in Pega.
    test_data_mdls = f"{basePath}/data/active_range/dmModels.csv.gz"
    test_data_preds = f"{basePath}/data/active_range/dmPredictors.csv.gz"
    dm = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls),
        predictor_df=pl.scan_csv(test_data_preds),
    )

    # These tests are identical to the tests in the old R version
    # be careful changing values

    active_ranges = dm.active_ranges("664cc653-279f-54ae-926f-694652d89a54").collect()
    print(active_ranges)
    assert active_ranges["idx_min"].item() == 6
    assert active_ranges["idx_max"].item() == 7
    assert round(active_ranges["AUC_Datamart"].item(), 6) == 0.760333
    assert round(active_ranges["AUC_FullRange"].item(), 6) == 0.760333
    assert round(active_ranges["AUC_ActiveRange"].item(), 6) == 0.5

    active_ranges = dm.active_ranges("4574f1fd-13a7-5703-bf38-9374641f370f").collect()
    print(active_ranges)
    assert active_ranges["idx_min"].item() == 0
    assert active_ranges["idx_max"].item() == 2
    assert round(active_ranges["AUC_Datamart"].item(), 6) == 0.557292
    assert round(active_ranges["AUC_FullRange"].item(), 6) == 0.557292
    assert round(active_ranges["AUC_ActiveRange"].item(), 6) == 0.557292


def test_active_range_newer_single():
    test_data_mdls = f"{basePath}/data/active_range/all_1_mdls.csv"
    test_data_preds = f"{basePath}/data/active_range/all_1_preds.csv"
    dm = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls),
        predictor_df=pl.scan_csv(test_data_preds),
    )

    ar = dm.active_ranges().collect()
    assert ar["idx_min"].item() == 0
    assert ar["idx_max"].item() == 3
    assert round(ar["AUC_Datamart"].item(), 6) == 0.544661
    assert round(ar["AUC_FullRange"].item(), 6) == 0.544661
    assert round(ar["AUC_ActiveRange"].item(), 6) == 0.534542


def test_active_range_Pega8():
    dm = ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data/active_range/CDHSample-Pega8",
    )

    ar = dm.active_ranges().collect()

    assert ar["nActivePredictors"].to_list() == [19, 17, 11, 3]
    assert ar["idx_min"].to_list() == [2, 0, 0, 1]  # the R version is +1 so 3, 1, 1, 2
    assert ar["idx_max"].to_list() == [10, 10, 2, 4]
    assert [round(x, 6) for x in ar["AUC_Datamart"].to_list()] == [
        0.533401,
        0.562353,
        0.559777,
        0.628571,
    ]
    assert [round(x, 7) for x in ar["AUC_ActiveRange"].to_list()] == [
        0.5334013,
        0.5623530,
        0.5597765,
        0.5476054,
    ]
    assert [round(x, 7) for x in ar["AUC_FullRange"].to_list()] == [
        0.5460856,
        0.5652007,
        0.5597765,
        0.5781145,
    ]
    assert [round(x, 6) for x in ar["score_min"].to_list()] == [
        -0.620929,
        -0.600759,
        -0.749120,
        -1.425557,
    ]
    assert [round(x, 6) for x in ar["score_max"].to_list()] == [
        0.468903,
        0.681244,
        1.454179,
        -0.425495,
    ]


def _check_cat(dm, pred_name):
    return (
        dm.predictor_data.filter(PredictorName=pred_name).select(pl.col("PredictorCategory").unique()).collect().item()
    )


def test_predictor_categorization_default(sample):
    default_cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    assert default_cats == ["Customer", "IH", "Param"]

    # print(sample.predictor_data.select("PredictorName", "PredictorCategory").unique().sort("PredictorName").collect().to_pandas())

    assert _check_cat(sample, "Customer.HealthMatter") == "Customer"
    assert _check_cat(sample, "IH.SMS.Outbound.Loyal.pxLastOutcomeTime.DaysSince") == "IH"
    assert _check_cat(sample, "Classifier") is None


def test_predictor_categorization_custom_expression(sample):
    categorization = pl.when(
        pl.col("PredictorName").cast(pl.Utf8).str.contains("Score"),
    ).then(pl.lit("External Model"))

    sample.apply_predictor_categorization(categorization)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    # print(
    #     sample.predictor_data.select("PredictorName", "PredictorCategory")
    #     .unique()
    #     .sort("PredictorName")
    #     .collect()
    # )
    assert cats == ["Customer", "External Model", "IH", "Param"]
    assert _check_cat(sample, "Customer.RiskScore") == "External Model"


def test_predictor_categorization_dictionary(sample):
    categorization = {"XGBoost Model": "Score"}

    sample.apply_predictor_categorization(categorization)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    # print(
    #     sample.predictor_data.select("PredictorName", "PredictorCategory")
    #     .unique()
    #     .sort("PredictorName")
    #     .collect()
    # )
    assert cats == ["Customer", "IH", "Param", "XGBoost Model"]
    assert _check_cat(sample, "Customer.CreditScore") == "XGBoost Model"


def test_predictor_categorization_dictionary_regexps(sample):
    # Using a reg exp w/o setting the flag should not match anything
    categorization = {"XGBoost Model": "Score$"}
    sample.apply_predictor_categorization(categorization)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    assert "XGBoost Model" not in cats

    # But with the flag we should get some more results
    categorization = {"XGBoost Model": "Score$"}
    sample.apply_predictor_categorization(categorization, use_regexp=True)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    assert "XGBoost Model" in cats


def test_get_last_data_for_report(sample: ADMDatamart):
    """Test get_last_data_for_report formatting."""
    report_data = sample.get_last_data_for_report()

    # Should return a collected DataFrame
    assert isinstance(report_data, pl.DataFrame)
    assert report_data.height > 0

    # Check that nulls are filled with "NA" for string columns
    string_cols = [col for col in report_data.columns if report_data[col].dtype == pl.Utf8]
    for col in string_cols:
        # Should not have null values in string columns
        if col in ["Channel", "Direction", "Configuration"]:
            assert report_data[col].null_count() == 0

    # Check SuccessRate and Performance are filled with 0 for null/nan
    if "SuccessRate" in report_data.columns:
        success_rates = report_data["SuccessRate"].to_list()
        assert all(v is not None or v == 0 for v in success_rates)

    if "Performance" in report_data.columns:
        performances = report_data["Performance"].to_list()
        assert all(v is not None or v == 0 for v in performances)

    # Check Channel/Direction concatenated column exists
    assert "Channel/Direction" in report_data.columns

    # Verify no categorical columns remain (all should be cast to string)
    for col in report_data.columns:
        assert report_data[col].dtype != pl.Categorical


def test_has_single_snapshot_multi(sample: ADMDatamart):
    """Sample data has multiple snapshots — has_single_snapshot must be False."""
    assert sample.has_single_snapshot is False


def test_has_single_snapshot_single():
    """When model_df contains exactly one SnapshotTime, has_single_snapshot is True."""
    import datetime

    single_ts = datetime.datetime(2024, 1, 1)
    dm = ADMDatamart(
        model_df=pl.LazyFrame(
            {
                "ModelID": ["m1", "m2"],
                "SnapshotTime": [single_ts, single_ts],
                "Name": ["Action A", "Action B"],
                "Channel": ["Web", "Web"],
                "Direction": ["Inbound", "Inbound"],
                "Issue": ["Sales", "Sales"],
                "Group": ["Cards", "Loans"],
                "Type": ["adaptive", "adaptive"],
                "Configuration": ["OmniAdaptiveModel", "OmniAdaptiveModel"],
                "ResponseCount": [100, 200],
                "Positives": [10, 20],
                "Performance": [0.7, 0.72],
            }
        )
    )
    assert dm.has_single_snapshot is True


def test_has_single_snapshot_two_snapshots():
    """When model_df contains two distinct SnapshotTimes, has_single_snapshot is False."""
    import datetime

    ts1 = datetime.datetime(2024, 1, 1)
    ts2 = datetime.datetime(2024, 1, 2)
    dm = ADMDatamart(
        model_df=pl.LazyFrame(
            {
                "ModelID": ["m1", "m1"],
                "SnapshotTime": [ts1, ts2],
                "Name": ["Action A", "Action A"],
                "Channel": ["Web", "Web"],
                "Direction": ["Inbound", "Inbound"],
                "Issue": ["Sales", "Sales"],
                "Group": ["Cards", "Cards"],
                "Type": ["adaptive", "adaptive"],
                "Configuration": ["OmniAdaptiveModel", "OmniAdaptiveModel"],
                "ResponseCount": [100, 150],
                "Positives": [10, 15],
                "Performance": [0.7, 0.71],
            }
        )
    )
    assert dm.has_single_snapshot is False


def test_from_s3_downloads_and_delegates(monkeypatch, tmp_path):
    """from_s3 downloads model + predictor objects and delegates to from_ds_export."""
    pytest.importorskip("moto")
    pytest.importorskip("boto3")
    import boto3
    from moto import mock_aws

    model_src = f"{basePath}/data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
    pred_src = (
        f"{basePath}/data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"
    )

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        client.upload_file(model_src, "test-bucket", "exports/models.zip")
        client.upload_file(pred_src, "test-bucket", "exports/predictors.zip")

        dm = ADMDatamart.from_s3(
            bucket="test-bucket",
            model_key="exports/models.zip",
            predictor_key="exports/predictors.zip",
            boto3_client=client,
        )

    assert dm.model_data is not None
    assert dm.predictor_data is not None
    assert dm.model_data.collect().height > 0


def test_from_s3_model_only(monkeypatch):
    """from_s3 works without a predictor key."""
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    model_src = f"{basePath}/data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        client.upload_file(model_src, "test-bucket", "models.zip")

        dm = ADMDatamart.from_s3(
            bucket="test-bucket",
            model_key="models.zip",
            boto3_client=client,
        )

    assert dm.model_data is not None
    assert dm.predictor_data is None


# ---------------------------------------------------------------------------
# Helper-method unit tests (synthetic minimal fixtures, exact-value assertions)
# ---------------------------------------------------------------------------


def _minimal_model_df(**overrides):
    """Tiny model-snapshot LazyFrame (2 rows, 1 model, 2 snapshots)."""
    import datetime

    base = {
        "ModelID": ["m1", "m1"],
        "SnapshotTime": [
            datetime.datetime(2024, 1, 1),
            datetime.datetime(2024, 1, 2),
        ],
        "Name": ["Action A", "Action A"],
        "Channel": ["Web", "Web"],
        "Direction": ["Inbound", "Inbound"],
        "Issue": ["Sales", "Sales"],
        "Group": ["Cards", "Cards"],
        "Type": ["adaptive", "adaptive"],
        "Configuration": ["OmniAdaptiveModel", "OmniAdaptiveModel"],
        "ResponseCount": [100, 200],
        "Positives": [10, 25],
        "Performance": [0.7, 0.8],
    }
    base.update(overrides)
    return pl.LazyFrame(base)


# ---- _normalize_performance_scale (static) --------------------------------


def test_normalize_performance_scale_above_one_divides_by_100():
    df = pl.LazyFrame({"Performance": [55.0, 80.0, 100.0]})
    out = ADMDatamart._normalize_performance_scale(df).collect()
    assert out["Performance"].to_list() == [0.55, 0.8, 1.0]


def test_normalize_performance_scale_below_or_equal_one_unchanged():
    df = pl.LazyFrame({"Performance": [0.5, 0.75, 1.0]})
    out = ADMDatamart._normalize_performance_scale(df).collect()
    assert out["Performance"].to_list() == [0.5, 0.75, 1.0]


def test_normalize_performance_scale_exactly_one_is_no_op():
    df = pl.LazyFrame({"Performance": [1.0, 1.0]})
    out = ADMDatamart._normalize_performance_scale(df).collect()
    assert out["Performance"].to_list() == [1.0, 1.0]


def test_normalize_performance_scale_all_nan_no_op():
    import math

    df = pl.LazyFrame({"Performance": [float("nan"), float("nan")]})
    out = ADMDatamart._normalize_performance_scale(df).collect()
    vals = out["Performance"].to_list()
    assert all(math.isnan(v) for v in vals)


def test_normalize_performance_scale_all_null_no_op():
    df = pl.LazyFrame({"Performance": [None, None]}, schema={"Performance": pl.Float64})
    out = ADMDatamart._normalize_performance_scale(df).collect()
    assert out["Performance"].to_list() == [None, None]


def test_normalize_performance_scale_no_performance_column():
    df = pl.LazyFrame({"Other": [1, 2, 3]})
    out = ADMDatamart._normalize_performance_scale(df).collect()
    assert out["Other"].to_list() == [1, 2, 3]
    assert "Performance" not in out.columns


# ---- _validate_model_data -------------------------------------------------


def test_validate_model_data_none_returns_none():
    dm = ADMDatamart()
    assert dm._validate_model_data(None) is None


def test_validate_model_data_adds_success_rate_and_modeltechnique():
    dm = ADMDatamart(model_df=_minimal_model_df())
    out = dm._require_model_data().collect()

    # Sorted by SnapshotTime, ModelID
    assert out["ResponseCount"].to_list() == [100, 200]
    assert out["Positives"].to_list() == [10, 25]
    # SuccessRate = Positives / ResponseCount
    assert out["SuccessRate"].to_list() == [0.10, 0.125]
    # ModelTechnique synthesised when absent
    assert "ModelTechnique" in out.columns
    assert out["ModelTechnique"].to_list() == [None, None]


def test_validate_model_data_success_rate_zero_when_no_responses():
    df = _minimal_model_df(ResponseCount=[0, 100], Positives=[0, 5])
    dm = ADMDatamart(model_df=df)
    out = dm._require_model_data().collect().sort("SnapshotTime")
    # 0/0 → NaN → filled with 0
    assert out["SuccessRate"].to_list() == [0.0, 0.05]


def test_validate_model_data_normalizes_performance_above_one():
    df = _minimal_model_df(Performance=[70.0, 80.0])
    dm = ADMDatamart(model_df=df)
    out = dm._require_model_data().collect()
    assert out["Performance"].to_list() == pytest.approx([0.7, 0.8])


def test_validate_model_data_is_updated_flag():
    df = _minimal_model_df(ResponseCount=[100, 100], Positives=[10, 10])
    dm = ADMDatamart(model_df=df)
    out = dm._require_model_data().collect().sort("SnapshotTime")
    # First row of model is always IsUpdated=True (null diff filled True);
    # second row has identical counts → not updated
    assert out["IsUpdated"].to_list() == [True, False]


# ---- _validate_predictor_data --------------------------------------------


def _minimal_predictor_df():
    import datetime

    return pl.LazyFrame(
        {
            "ModelID": ["m1", "m1", "m1"],
            "PredictorName": ["Customer.Age", "Customer.Age", "Classifier"],
            "EntryType": ["Active", "Active", "Classifier"],
            "BinIndex": [1, 2, 1],
            "BinPositives": [4.0, 6.0, 10.0],
            "BinNegatives": [16.0, 14.0, 30.0],
            "BinResponseCount": [20.0, 20.0, 40.0],
            "Performance": [0.65, 0.65, 0.65],
            "SnapshotTime": [datetime.datetime(2024, 1, 1)] * 3,
            "Type": ["numeric", "numeric", "symbolic"],
        }
    )


def test_validate_predictor_data_none_returns_none():
    dm = ADMDatamart()
    assert dm._validate_predictor_data(None) is None


def test_validate_predictor_data_adds_propensity_columns_and_categorization():
    dm = ADMDatamart(predictor_df=_minimal_predictor_df())
    out = dm._require_predictor_data().collect().sort("PredictorName", "BinIndex")

    # BinPropensity = BinPositives / BinResponseCount
    assert out["BinPropensity"].to_list() == [10 / 40, 4 / 20, 6 / 20]
    # BinAdjustedPropensity = (BinPositives + 0.5) / (BinResponseCount + 1)
    assert out["BinAdjustedPropensity"].to_list() == [
        (10 + 0.5) / (40 + 1),
        (4 + 0.5) / (20 + 1),
        (6 + 0.5) / (20 + 1),
    ]
    # Default categorization → "Customer.Age" maps to Customer; Classifier left null
    cats = dict(
        zip(
            out["PredictorName"].to_list(),
            out["PredictorCategory"].to_list(),
            strict=True,
        )
    )
    assert cats["Customer.Age"] == "Customer"
    assert cats["Classifier"] is None


def test_validate_predictor_data_synthesises_bin_response_count_when_absent():
    import datetime

    df = pl.LazyFrame(
        {
            "ModelID": ["m1", "m1"],
            "PredictorName": ["Customer.Age", "Classifier"],
            "EntryType": ["Active", "Classifier"],
            "BinIndex": [1, 1],
            "BinPositives": [4.0, 10.0],
            "BinNegatives": [16.0, 30.0],
            "Performance": [0.65, 0.65],
            "SnapshotTime": [datetime.datetime(2024, 1, 1)] * 2,
            "Type": ["numeric", "symbolic"],
        }
    )
    dm = ADMDatamart(predictor_df=df)
    out = dm._require_predictor_data().collect().sort("PredictorName")
    # BinResponseCount synthesised as BinPositives + BinNegatives
    assert out["BinResponseCount"].to_list() == [40.0, 20.0]
    assert out["BinPropensity"].to_list() == [10 / 40, 4 / 20]


# ---- apply_predictor_categorization (df= passthrough mode) ----------------


def test_apply_predictor_categorization_df_mode_dict():
    dm = ADMDatamart()
    df = pl.LazyFrame(
        {
            "PredictorName": ["Customer.Score", "IH.Foo", "Classifier"],
            "EntryType": ["Active", "Active", "Classifier"],
        }
    )
    out = dm.apply_predictor_categorization(
        categorization={"Risk": "Score", "History": "IH"},
        df=df,
    ).collect()
    cats = dict(
        zip(
            out["PredictorName"].to_list(),
            out["PredictorCategory"].to_list(),
            strict=True,
        )
    )
    assert cats == {"Customer.Score": "Risk", "IH.Foo": "History", "Classifier": None}


def test_apply_predictor_categorization_invalid_type_returns_df_unchanged():
    dm = ADMDatamart()
    df = pl.LazyFrame({"PredictorName": ["X"], "EntryType": ["Active"]})
    # Non-expr / non-callable / non-dict → method returns df as-is
    out = dm.apply_predictor_categorization(categorization=42, df=df)  # type: ignore[arg-type]
    assert out is df


# ---- _unique_sorted_from_model_data ---------------------------------------


def test_unique_sorted_from_model_data_dedupes_and_sorts():
    df = _minimal_model_df(
        ModelID=["m1", "m2"],
        Channel=["Web", "Email"],
    )
    dm = ADMDatamart(model_df=df)
    assert dm._unique_sorted_from_model_data(pl.col("Channel"), "Channel") == [
        "Email",
        "Web",
    ]


def test_unique_sorted_from_model_data_raises_without_model_data():
    dm = ADMDatamart()
    with pytest.raises(ValueError, match="requires model data"):
        dm._unique_sorted_from_model_data(pl.col("Channel"), "Channel")


# ---- _get_first_action_dates ---------------------------------------------


def test_get_first_action_dates_none_returns_none():
    dm = ADMDatamart()
    assert dm._get_first_action_dates(None) is None


def test_get_first_action_dates_returns_min_per_action():
    import datetime

    df = pl.LazyFrame(
        {
            "Name": ["A", "A", "B", "B"],
            "SnapshotTime": [
                datetime.datetime(2024, 1, 5),
                datetime.datetime(2024, 1, 1),
                datetime.datetime(2024, 2, 1),
                datetime.datetime(2024, 1, 20),
            ],
        }
    )
    dm = ADMDatamart()
    out = dm._get_first_action_dates(df).collect()
    assert out["Name"].to_list() == ["A", "B"]
    assert out["ActionFirstSnapshotTime"].to_list() == [
        datetime.datetime(2024, 1, 1),
        datetime.datetime(2024, 1, 20),
    ]


# ---- _require_* error paths ----------------------------------------------


def test_require_model_data_raises_when_missing():
    dm = ADMDatamart()
    with pytest.raises(
        ValueError,
        match="This operation requires model data, but no model_df was provided",
    ):
        dm._require_model_data()


def test_require_predictor_data_raises_when_missing():
    dm = ADMDatamart()
    with pytest.raises(
        ValueError,
        match="This operation requires predictor data, but no predictor_df",
    ):
        dm._require_predictor_data()


def test_require_first_action_dates_raises_when_missing():
    dm = ADMDatamart()
    with pytest.raises(
        ValueError,
        match="This operation requires first action dates",
    ):
        dm._require_first_action_dates()


def test_require_model_data_returns_lazyframe_when_present():
    dm = ADMDatamart(model_df=_minimal_model_df())
    assert isinstance(dm._require_model_data(), pl.LazyFrame)
    assert isinstance(dm._require_first_action_dates(), pl.LazyFrame)
