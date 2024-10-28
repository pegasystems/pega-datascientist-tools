"""
Testing the functionality of the io functions
"""

import os
import pathlib
import zipfile
from io import BytesIO

import polars as pl
import pytest
from pdstools import pega_io


@pl.api.register_lazyframe_namespace("shape")
class Shape:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (ldf.select(pl.first().len()).collect().item(), len(ldf.columns))


@pytest.fixture
def test_data() -> pl.LazyFrame:
    return pega_io.read_ds_export(
        "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        f"{basePath}/data",
    )


# Tests for get_latest_file function
def test_find_default_model():
    file = pega_io.get_latest_file(path=f"{basePath}/data", target="model_data")
    assert os.path.join(file) == os.path.join(
        f"{basePath}/data", "pr_data_dm_admmart_mdl_fact.csv"
    )


def test_find_default_predictors():
    file = pega_io.get_latest_file(path=f"{basePath}/data", target="predictor_data")
    assert os.path.join(file) == os.path.join(
        f"{basePath}/data",
        "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip",
    )


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        pega_io.get_latest_file(path="data1", target="predictor_data")


def test_wrong_filename():
    file = pega_io.get_latest_file(path=f"{basePath}/data", target="combined_data")
    assert file == "Target not found"


# Tests for readZippedFile function
def test_only_imports_zips():
    with pytest.raises(zipfile.BadZipFile):
        pega_io.read_zipped_file(
            os.path.join(f"{basePath}/data", "pr_data_dm_admmart_mdl_fact.csv")
        )


def test_import_produces_polars():
    assert isinstance(
        pega_io.read_ds_export(
            "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
            f"{basePath}/data",
        ),
        pl.LazyFrame,
    )


def test_import_produces_bytes():
    ret = pega_io.read_zipped_file(
        os.path.join(
            f"{basePath}/data",
            "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        ),
        verbose=True,
    )

    assert isinstance(ret[0], BytesIO)
    assert ret[1] == ".json"


def test_read_json(test_data):
    temp_filename = "data.json"
    test_data.collect().write_ndjson(temp_filename)
    assert pega_io.read_ds_export(
        "data.json",
        ".",
    ).shape == (20, 23)
    os.remove(temp_filename)


def test_read_modelData():
    assert pega_io.read_ds_export(
        "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        f"{basePath}/data",
    ).shape == (20, 23)


def test_read_predictorData():
    assert pega_io.read_ds_export(
        "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip",
        f"{basePath}/data",
    ).shape == (1755, 35)


def test_polars_zip_from_url():
    df = pega_io.read_ds_export(
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


def test_import_parquet(test_data):
    temp_filename = "data.parquet"
    test_data.collect().write_parquet(temp_filename)
    path = "."
    assert polars_checks(pega_io.read_ds_export(path=path, filename=temp_filename))
    os.remove(temp_filename)


def test_import_csv():
    path = f"{basePath}/data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert polars_checks(pega_io.read_ds_export(path=path, filename=models))


def test_csv_from_url():
    path = "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert polars_checks(pega_io.read_ds_export(path=path, filename=models))


def test_cdh_sample_models_from_url():
    path = "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert polars_checks(pega_io.read_ds_export(path=path, filename=models))


def test_cdh_sample_models_locally():
    path = f"{basePath}/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert polars_checks(pega_io.read_ds_export(path=path, filename=models))


def test_cdh_sample_autodiscovered_locally():
    path = f"{basePath}/data"
    assert polars_checks(pega_io.read_ds_export(path=path, filename="model_data"))


def test_file_not_found_in_good_dir():
    assert pega_io.read_ds_export(path=f"{basePath}/data", filename="models") == None


def test_file_not_found_in_bad_dir():
    with pytest.raises(FileNotFoundError):
        pega_io.read_ds_export(path="data1", filename="model_data")


def test_import_csv_pandas():
    path = f"{basePath}/data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert polars_checks(pega_io.read_ds_export(path=path, filename=models))


def test_import_zipped_json_pandas():
    path = f"{basePath}/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert polars_checks(pega_io.read_ds_export(path=path, filename=models))


def test_import_not_supported_extension(test_data):
    with pytest.raises(ValueError):
        temp_filename = "data.xlsx"
        test_data.collect().write_excel(temp_filename)
        path = "."
        pega_io.read_ds_export(filename=temp_filename, path=path)
    os.remove(temp_filename)


def test_import_file_verbose(test_data):
    temp_filename = "data.arrow"
    test_data.collect().write_ipc(temp_filename)
    path = "."
    assert polars_checks(pega_io.read_ds_export(filename=temp_filename, path=path))
    os.remove(temp_filename)


def test_getMatches():
    files_dir = f"{basePath}/data"
    target = "value_finder"
    expected_outcome = "Data-Insights_pyValueFinder_20210824T112615_GMT.zip"
    outcome = pathlib.Path(pega_io.get_latest_file(files_dir, target)).name
    assert outcome == expected_outcome

    target = "wrong_target_name"
    with pytest.raises(ValueError):
        pega_io.find_files(files_dir, target)


def test_cache_to_file():
    input_df = pl.LazyFrame(
        {
            "PredictorName": ["Customer.Variable", "Variable"],
            "date": ["2023-01-01", "2023-01-02"],
        }
    )

    cached_path = pega_io.cache_to_file(input_df, path=os.getcwd(), name="cache_test")

    assert input_df.collect().equals(pl.scan_ipc(cached_path).collect())

    # test for cache_type="ipc"
    cached_path_arrow = pega_io.cache_to_file(
        input_df, path=os.getcwd(), name="cache_test", cache_type="ipc"
    )

    assert input_df.collect().equals(pl.read_ipc(cached_path_arrow))
    # test for cache_type="parquet"
    cached_path_parquet = pega_io.cache_to_file(
        input_df, path=os.getcwd(), name="cache_test", cache_type="parquet"
    )

    assert input_df.collect().equals(pl.read_parquet(cached_path_parquet))
    for file in [cached_path_parquet, cached_path_arrow]:
        os.remove(file)
