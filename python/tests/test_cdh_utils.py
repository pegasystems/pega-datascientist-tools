import sys
import pytest
import zipfile
import pandas as pd
from io import BytesIO
import urllib.request

sys.path.append("python")
from cdhtools import cdh_utils

# Tests for get_latest_file function
def test_find_default_model():
    file = cdh_utils.get_latest_file(path="data", target="modelData")
    assert file == "data/pr_data_dm_admmart_mdl_fact.csv"


def test_find_default_predictors():
    file = cdh_utils.get_latest_file(path="data", target="predictorData")
    assert file == "data/pr_data_dm_admmart_pred.csv"


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        cdh_utils.get_latest_file(path="data1", target="predictorData")


def test_wrong_filename():
    file = cdh_utils.get_latest_file(path="data", target="combinedData")
    assert file == "Target not found"


# Tests for readZippedFile function
def test_only_imports_zips():
    with pytest.raises(zipfile.BadZipFile):
        cdh_utils.readZippedFile("data/pr_data_dm_admmart_mdl_fact.csv")


def test_import_produces_dataframe():
    assert isinstance(
        cdh_utils.readZippedFile(
            "data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
        ),
        pd.DataFrame,
    )


def test_read_modelData():
    assert cdh_utils.readZippedFile(
        "data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    ).shape == (20, 23)


def test_read_predictorData():
    assert cdh_utils.readZippedFile(
        "data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip"
    ).shape == (1755, 35)


def test_read_zip_from_url():
    assert checks(
        cdh_utils.readZippedFile(
            BytesIO(
                urllib.request.urlopen(
                    "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
                ).read()
            )
        )
    )


# Tests for ReadDSExport function
@pytest.fixture
def test_data():
    return cdh_utils.readZippedFile(
        "data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    )


def checks(df):
    "Very simple convienence function to check if it is a dataframe with rows."
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        return True
    else:
        return False


def test_dataframe_returns_itself(test_data):
    input = test_data
    assert cdh_utils.readDSExport(input).equals(input)


def test_import_csv():
    path = "data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert checks(cdh_utils.readDSExport(path=path, filename=models))


def test_csv_from_url():
    path = "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert checks(cdh_utils.readDSExport(path=path, filename=models))


def test_cdh_sample_models_from_url():
    path = "https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert checks(cdh_utils.readDSExport(path=path, filename=models))


def test_cdh_sample_models_locally():
    path = "data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert checks(cdh_utils.readDSExport(path=path, filename=models))


def test_cdh_sample_autodiscovered_locally():
    path = "data"
    assert checks(cdh_utils.readDSExport(path=path, filename="modelData"))


def test_file_not_found_in_good_dir():
    assert cdh_utils.readDSExport(path="data", filename="models") == None


def test_file_not_found_in_bad_dir():
    with pytest.raises(FileNotFoundError):
        cdh_utils.readDSExport(path="data1", filename="modelData")


def test_import_csv_pandas():
    path = "data"
    models = "pr_data_dm_admmart_mdl_fact.csv"
    assert checks(cdh_utils.readDSExport(path=path, filename=models, force_pandas=True))


def test_import_zipped_json_pandas():
    path = "data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    assert checks(cdh_utils.readDSExport(path=path, filename=models, force_pandas=True))
