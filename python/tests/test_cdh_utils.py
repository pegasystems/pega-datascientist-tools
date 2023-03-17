"""
Testing the functionality of the cdh_utils functions
"""

"""
Testing the functionality of the cdh_utils functions
"""
import sys
import os
import pytest
import zipfile
import pandas as pd
from io import BytesIO
import numpy as np
import urllib.request
import datetime
from pytz import timezone
import polars as pl
import pathlib

pathlib.Path(__file__).parent.parent

sys.path.append("python")
from pdstools import cdh_utils
from pdstools import datasets


@pl.api.register_lazyframe_namespace("shape")
class Shape:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (ldf.select(pl.first().count()).collect().item(), len(ldf.columns))


@pytest.fixture
def test_data():
    return cdh_utils.readDSExport(
        "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        "data",
    )


# Tests for get_latest_file function
def test_find_default_model():
    file = cdh_utils.get_latest_file(path="data", target="modelData")
    assert os.path.join(file) == os.path.join("data", "pr_data_dm_admmart_mdl_fact.csv")


def test_find_default_predictors():
    file = cdh_utils.get_latest_file(path="data", target="predictorData")
    assert os.path.join(file) == os.path.join("data", "pr_data_dm_admmart_pred.csv")


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        cdh_utils.get_latest_file(path="data1", target="predictorData")


def test_wrong_filename():
    file = cdh_utils.get_latest_file(path="data", target="combinedData")
    assert file == "Target not found"


# Tests for readZippedFile function
def test_only_imports_zips():
    with pytest.raises(zipfile.BadZipFile):
        cdh_utils.readZippedFile(
            os.path.join("data", "pr_data_dm_admmart_mdl_fact.csv")
        )


def test_import_produces_polars():
    assert isinstance(
        cdh_utils.readDSExport(
            "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
            "data",
        ),
        pl.LazyFrame,
    )


def test_import_produces_bytes():
    ret = cdh_utils.readZippedFile(
        os.path.join(
            "data",
            "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        ),
        verbose=True,
    )

    assert isinstance(ret[0], BytesIO)
    assert ret[1] == ".json"


def test_read_json(test_data):
    temp_filename = "data.json"
    test_data.collect().write_ndjson(temp_filename)
    assert cdh_utils.readDSExport(
        "data.json",
        ".",
    ).shape == (20, 23)
    os.remove(temp_filename)


def test_read_modelData():
    assert cdh_utils.readDSExport(
        "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        "data",
    ).shape == (20, 23)


def test_read_predictorData():
    assert cdh_utils.readDSExport(
        "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip",
        "data",
    ).shape == (1755, 35)


def test_polars_zip_from_url():
    df = cdh_utils.readDSExport(
        "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data",
    )

    assert isinstance(
        df,
        pl.LazyFrame,
    )
    assert df.shape == (20, 23)


def polars_checks(df):
    "Very simple convienence function to check if it is a dataframe with rows."
    if isinstance(df, pl.LazyFrame) and df.shape[0] > 0:
        return True


def test_lazyframe_returns_itself(test_data):
    input = test_data
    assert cdh_utils.readDSExport(input).collect().frame_equal(input.collect())


def test_dataframe_returns_itself(test_data):
    input = test_data
    assert (
        cdh_utils.readDSExport(input.collect()).collect().frame_equal(input.collect())
    )


def test_pandasdataframe_returns_lazyframe(test_data):
    input = test_data.collect().to_pandas()
    assert cdh_utils.readDSExport(input).collect().frame_equal(test_data.collect())


def test_import_parquet(test_data):
    temp_filename = "data.parquet"
    test_data.collect().write_parquet(temp_filename)
    path = "."
    assert polars_checks(cdh_utils.readDSExport(path=path, filename=temp_filename))
    os.remove(temp_filename)


def test_import_csv():
    path = "data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert polars_checks(cdh_utils.readDSExport(path=path, filename=models))


def test_csv_from_url():
    path = "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert polars_checks(cdh_utils.readDSExport(path=path, filename=models))


def test_cdh_sample_models_from_url():
    path = "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert polars_checks(cdh_utils.readDSExport(path=path, filename=models))


def test_cdh_sample_models_locally():
    path = "data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert polars_checks(cdh_utils.readDSExport(path=path, filename=models))


def test_cdh_sample_autodiscovered_locally():
    path = "data"
    assert polars_checks(cdh_utils.readDSExport(path=path, filename="modelData"))


def test_file_not_found_in_good_dir():
    assert cdh_utils.readDSExport(path="data", filename="models") == None


def test_file_not_found_in_bad_dir():
    with pytest.raises(FileNotFoundError):
        cdh_utils.readDSExport(path="data1", filename="modelData")


def test_import_csv_pandas():
    path = "data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert polars_checks(cdh_utils.readDSExport(path=path, filename=models))


def test_import_zipped_json_pandas():
    path = "data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert polars_checks(cdh_utils.readDSExport(path=path, filename=models))


def test_import_not_supported_extension(test_data):
    with pytest.raises(ValueError):
        temp_filename = "data.xlsx"
        test_data.collect().write_excel(temp_filename)
        path = "."
        cdh_utils.readDSExport(filename=temp_filename, path=path)
    os.remove(temp_filename)


def test_import_file_verbose(test_data):
    temp_filename = "data.arrow"
    test_data.collect().write_ipc(temp_filename)
    path = "."
    assert polars_checks(cdh_utils.readDSExport(filename=temp_filename, path=path))
    os.remove(temp_filename)


def test_getMatches():
    files_dir = "data"
    target = "ValueFinder"
    expected_outcome = "Data-Insights_pyValueFinder_20210824T112615_GMT.zip"
    outcome = pathlib.Path(cdh_utils.get_latest_file(files_dir, target)).name
    assert outcome == expected_outcome

    target = "wrong_target_name"
    with pytest.raises(ValueError):
        cdh_utils.getMatches(files_dir, target)


def test_cache_to_file():
    input_df = pl.LazyFrame(
        {
            "PredictorName": ["Customer.Variable", "Variable"],
            "date": ["2023-01-01", "2023-01-02"],
        }
    )

    cached_path = cdh_utils.cache_to_file(input_df, path=os.getcwd(), name=f"cahe_test")

    assert (
        cdh_utils.readDSExport(input_df)
        .collect()
        .frame_equal(cdh_utils.readDSExport(cached_path).collect())
    )

    # test for cache_type="ipc"
    cached_path_arrow = cdh_utils.cache_to_file(
        input_df, path=os.getcwd(), name=f"cahe_test", cache_type="ipc"
    )

    assert (
        cdh_utils.readDSExport(input_df)
        .collect()
        .frame_equal(cdh_utils.readDSExport(cached_path_arrow).collect())
    )
    # test for cache_type="parquet"
    cached_path_parquet = cdh_utils.cache_to_file(
        input_df, path=os.getcwd(), name=f"cahe_test", cache_type="parquet"
    )

    assert (
        cdh_utils.readDSExport(input_df)
        .collect()
        .frame_equal(cdh_utils.readDSExport(cached_path_parquet).collect())
    )
    for file in [cached_path_parquet, cached_path_arrow]:
        os.remove(file)


def test_safe_range_auc():
    assert cdh_utils.safe_range_auc(0.8) == 0.8
    assert cdh_utils.safe_range_auc(0.4) == 0.6
    assert cdh_utils.safe_range_auc(np.nan) == 0.5


def test_auc_from_probs():
    assert cdh_utils.auc_from_probs([1, 1, 0], [0.6, 0.2, 0.2]) == 0.75
    assert cdh_utils.auc_from_probs([1, 1, 1], [0.6, 0.2, 0.2]) == 0.5
    assert (
        cdh_utils.auc_from_probs(
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [20, 19, 18, 17, 16, 15, 14, 13, 11.5, 11.5, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        )
        == 0.825
    )
    assert (
        abs(
            cdh_utils.auc_from_probs(
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                [
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                ],
            )
            - 0.7637363
        )
        < 1e-6
    )


def test_auc_from_bincounts():
    assert cdh_utils.auc_from_bincounts([3, 1, 0], [2, 0, 1]) == 0.75
    positives = [50, 70, 75, 80, 85, 90, 110, 130, 150, 160]
    negatives = [1440, 1350, 1170, 990, 810, 765, 720, 675, 630, 450]
    assert abs(cdh_utils.auc_from_bincounts(positives, negatives) - 0.6871) < 1e-6


def test_aucpr_from_probs():
    assert (
        abs(cdh_utils.aucpr_from_probs([1, 1, 0], [0.6, 0.2, 0.2]) - 0.4166667) < 1e-6
    )
    assert cdh_utils.aucpr_from_probs([1, 1, 1], [0.6, 0.2, 0.2]) == 0.0
    # assert abs(cdh_utils.aucpr_from_probs([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    #    [0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875]) - 0.8610687) < 1e-6


def test_aucpr_from_bincounts():
    assert cdh_utils.aucpr_from_bincounts([3, 1, 0], [2, 0, 1]) == 0.625
    assert (
        abs(
            cdh_utils.aucpr_from_bincounts(
                [50, 70, 75, 80, 85, 90, 110, 130, 150, 160],
                [1440, 1350, 1170, 990, 810, 765, 720, 675, 630, 450],
            )
            - 0.1489611
        )
        < 1e-6
    )


def test_auc2gini():
    assert abs(cdh_utils.auc2GINI(0.8232) - 0.6464) < 1e-6


def test_fromPRPCDateTime():
    assert (
        cdh_utils.fromPRPCDateTime("20180316T134127.847 GMT", True)
        == "2018-03-16 13:41:27 GMT"
    )
    assert cdh_utils.fromPRPCDateTime(
        "20180316T134127.847 GMT", False
    ) == datetime.datetime(2018, 3, 16, 13, 41, 27, 847, tzinfo=timezone("GMT"))
    assert (
        cdh_utils.fromPRPCDateTime("20180316T134127.847345", True)
        == "2018-03-16 13:41:27 "
    )
    assert (
        cdh_utils.fromPRPCDateTime("20180316T134127.8", True) == "2018-03-16 13:41:27 "
    )
    assert cdh_utils.fromPRPCDateTime("20180316T134127", True) == "2018-03-16 13:41:27 "


def test_toPRPCDateTime():
    assert (
        cdh_utils.toPRPCDateTime(
            datetime.datetime(2018, 3, 16, 13, 41, 27, 847000, tzinfo=timezone("GMT"))
        )
        == "20180316T134127.847 GMT+0000"
    )
    assert (
        cdh_utils.toPRPCDateTime(
            datetime.datetime(
                2018, 3, 16, 13, 41, 27, 847000, tzinfo=timezone("US/Eastern")
            )
        )
        == "20180316T134127.847 GMT-0456"
    )
    assert (
        cdh_utils.toPRPCDateTime(datetime.datetime(2018, 3, 16, 13, 41, 27, 847000))[:-3]
        == "20180316T134127.847 GMT+0000"[:-3]
    )


def test_weighted_average_polars():
    input = pl.DataFrame(
        {
            "SuccessRate": [1, 3, 10, 40],
            "Channel": ["SMS", "SMS", "Web", "Web"],
            "ResponseCount": [3, 1, 10, 5],
        }
    )
    output = (
        input.groupby("Channel")
        .agg(
            cdh_utils.weighed_average_polars("SuccessRate", "ResponseCount").alias(
                "SuccessRate_weighted"
            ),
        )
        .sort("Channel")
    )

    expected_output = pl.DataFrame(
        {"Channel": ["SMS", "Web"], "SuccessRate_weighted": [1.5, 20]}
    ).sort("Channel")

    assert output.frame_equal(expected_output)

    output = (
        input.filter(pl.col("Channel") == "SMS")
        .with_columns(
            cdh_utils.weighed_average_polars(
                vals="SuccessRate", weights="ResponseCount"
            ).alias("weighted_average")
        )
        .sort("SuccessRate")
    )

    expected_output = pl.DataFrame(
        {
            "SuccessRate": [1, 3],
            "Channel": ["SMS", "SMS"],
            "ResponseCount": [3, 1],
            "weighted_average": [1.5, 1.5],
        }
    ).sort("SuccessRate")

    assert output.frame_equal(expected_output)


def test_weighed_performance_polars():
    input = pl.DataFrame(
        {
            "Performance": [0.5, 0.8, 0.75, 0.5],  # 0.6, 0.6
            "Channel": ["SMS", "SMS", "Web", "Web"],
            "ResponseCount": [2, 1, 2, 3],
        }
    )

    output = (
        input.groupby("Channel")
        .agg(cdh_utils.weighed_performance_polars())
        .sort("Channel")
    )

    expected_output = pl.DataFrame(
        {"Channel": ["SMS", "Web"], "Performance": [0.6, 0.6]}
    ).sort("Channel")

    assert output.frame_equal(expected_output)


def test_zRatio():
    input = pl.DataFrame(
        {
            "Predictor_range": [
                "MISSING",
                "<7.08",
                "[7.08, 12.04>",
                "[12.04, 18.04>",
                ">=18.04",
            ],
            "BinPositives": [0, 7, 11, 12, 6],
            "BinNegatives": [5, 2208, 1919, 1082, 352],
        }
    )

    output = (
        input.with_columns(cdh_utils.zRatio())
        .with_columns(pl.col("ZRatio").round(2))
        .sort("Predictor_range")
    )

    Zratios = [-3.05, 1.66, -2.24, 1.76, -0.51]
    expected_output = input.sort("Predictor_range").with_columns(
        pl.Series(name="ZRatio", values=Zratios)
    )

    assert output.frame_equal(expected_output)


def test_log_odds():
    input = pl.DataFrame(
        {
            "Predictor_range": [
                "MISSING",
                "<7.08",
                "[7.08, 12.04>",
                "[12.04, 18.04>",
                ">=18.04",
            ],
            "Positives": [0, 7, 11, 12, 6],
            "ResponseCount": [5, 2215, 1930, 1094, 358],
        }
    )
    output = input.with_columns(cdh_utils.LogOdds().round(2)).sort("Predictor_range")

    log_odds_list = [1.65, -0.81, -0.23, -0.43, -0.87]
    expected_output = input.sort("Predictor_range").with_columns(
        pl.Series(name="LogOdds", values=log_odds_list)
    )

    assert output.frame_equal(expected_output)


def test_featureImportance():
    input = pl.DataFrame(
        {
            "ModelID": ["001", "001", "002", "002"],
            "PredictorName": ["Age", "Age", "CreditScore", "CreditScore"],
            "BinResponseCount": [20, 30, 12, 10],
            "BinPositives": [2, 7, 11, 6],
        }
    )

    output = input.with_columns(cdh_utils.featureImportance().round(2)).sort(
        "BinPositives"
    )
    importance_list = [-1.4, 1.13, -1.4, 1.13]
    expected_output = input.sort("BinPositives").with_columns(
        pl.Series(name="FeatureImportance", values=importance_list)
    )

    assert output.frame_equal(expected_output)


# Test _capitalize function
def test_capitalize_behavior():
    assert set(cdh_utils._capitalize(["pyTest", "pzTest", "pxTest"])) == {"Test"}
    assert cdh_utils._capitalize(["test"]) == ["Test"]
    assert cdh_utils._capitalize(["RESPONSEcOUNT"]) == ["ResponseCount"]
    assert cdh_utils._capitalize(["responsecounter"]) == ["ResponseCounter"]
    assert cdh_utils._capitalize(["responsenumber"]) == ["Responsenumber"]
    assert cdh_utils._capitalize(["Response Count"]) == ["Response Count"]
    assert cdh_utils._capitalize(["Response_count1"]) == ["Response_Count1"]
    assert cdh_utils._capitalize("Response_count1") == ["Response_Count1"]


def test_PredictorCategorization():
    df = (
        pl.LazyFrame({"PredictorName": ["Customer.Variable", "Variable"]})
        .with_columns(cdh_utils.defaultPredictorCategorization())
        .collect()
    )
    assert df.get_column("PredictorCategory").to_list() == ["Customer", "Primary"]

    df = (
        pl.LazyFrame({"Predictor": ["Customer.Variable", "Variable"]})
        .with_columns(cdh_utils.defaultPredictorCategorization(x="Predictor"))
        .collect()
    )
    assert df.get_column("PredictorCategory").to_list() == ["Customer", "Primary"]


def test_legend_color_order():
    input_fig = datasets.CDHSample().plotPerformanceSuccessRateBubbleChart()
    output_fig = cdh_utils.legend_color_order(input_fig)

    assert output_fig.data[0].marker.color == "#001F5F"
