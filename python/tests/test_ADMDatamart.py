from copy import deepcopy
import sys
import zipfile
import pytest
from numpy import NaN
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

import pathlib
pathlib.Path(__file__).parent.parent
sys.path.append("python")
from pdstools import ADMDatamart, cdh_utils


@pytest.fixture
def test():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart(
        path="data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip",
        context_keys=["Issue", "Group", "Channel"],
    )


def test_basic_available_columns(test):
    input = pd.DataFrame(columns=["ModelID", "Issue", "Unknown", "Auc", "Response"])
    output, variables, missing = test._available_columns(input)
    assert "Unknown" not in variables.values()
    assert "Unknown" not in missing
    assert "ResponseCount" in variables.values()
    assert ("Response", "ResponseCount") in variables.items()
    assert {"ModelID", "Issue", "ResponseCount"} not in missing
    assert "ModelName" in missing
    assert "Auc" not in variables.values()
    assert "Performance" not in variables.values()
    assert set(output.columns) == {
        "ModelID",
        "Issue",
        "Unknown",
        "Auc",
        "ResponseCount",
    }


def test_overwrite_mapping(test):
    input = pd.DataFrame(columns=["ModelID", "Issue", "Unknown", "Auc", "Response"])
    mapping = {"Unknown": "Unknown", "Performance": "Auc"}
    output, variables, missing = test._available_columns(
        input, overwrite_mapping=mapping
    )
    assert "Unknown" in set(variables.values())
    assert "Unknown" not in missing
    assert "ResponseCount" in variables.values()
    assert ("Response", "ResponseCount") in variables.items()
    assert "Auc" not in set(variables.values())
    assert "Performance" in set(variables.values())
    assert ("Auc", "Performance") in variables.items()
    assert {"ModelID", "Issue", "ResponseCount"} not in missing
    assert "ModelName" in missing
    assert set(output.columns) == {
        "ModelID",
        "Issue",
        "Unknown",
        "Performance",
        "ResponseCount",
    }


def test_import_utils_with_importing(test):
    output, renamed, missing = test._import_utils(
        path="data",
        name="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
    )
    assert output.shape == (20, 10)


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "pymodelid": ["model1", "model2", "model3"],
            "AUC": ["0.5000", 0.700, 0.9],
            "Junk": ["blabla", "blabla2", "blabla3"],
            "pyname": [
                '{"pyName": "ABC", "pyTreatment": "XYZ"}',
                '{"pyName": "abc", "pyTreatment": "xyz"}',
                "NormalName",
            ],
            "SnapshotTime": [
                "2022-03-01 05:00:00",
                "2022-03-01 05:00:00",
                "2022-03-01 05:00:00",
            ],
        }
    )


def test_import_utils(test, data):
    output, renamed, missing = test._import_utils(name=deepcopy(data))
    assert output.shape == (3, 3)
    assert "ModelName" in output.columns
    assert list(output["ModelID"]) == list(data["pymodelid"])
    assert is_datetime(output["SnapshotTime"])
    assert "Performance" not in output.columns
    assert renamed == {
        "ModelID": "ModelID",
        "Name": "ModelName",
        "SnapshotTime": "SnapshotTime",
    }
    assert "Junk" not in output.columns
    assert "Treatment" not in output.columns


def test_extract_treatment(test, data):
    mapping = {"Performance": "auc"}
    output, renamed, missing = test._import_utils(
        name=data, overwrite_mapping=mapping, extract_treatment="pyname"
    )
    assert output.shape == (3, 5)
    assert output["Performance"].dtype == "float64"
    assert list(output["Treatment"]) == ["XYZ", "xyz", NaN]
    assert list(output["ModelName"]) == ["ABC", "abc", "NormalName"]


def test_extract_treatment_all_json(test, data):
    df = deepcopy(data).reset_index(drop=True)
    df.loc[:, "pyname"] = [
        '{"pyName": "ABC", "pyTreatment": "XYZ"}',
        '{"pyName": "abc", "pyTreatment": "xyz"}',
        '{"pyName": "ABCD", "pyTreatment": "XYZ1"}',
    ]
    output, renamed, missing = test._import_utils(name=df, extract_treatment="pyname")
    assert list(output["ModelName"]) == ["ABC", "abc", "ABCD"]


def test_import_with_prequery(test, data):
    prequery = "pymodelid != 'model2'"
    output, renamed, missing = test._import_utils(name=data, prequery=prequery)
    assert output.shape == (2, 3)
    assert "model2" not in output["ModelID"]


def test_import_with_wrong_prequery(test, data):
    prequery = "non-existant-column != 'unknown'"
    output, renamed, missing = test._import_utils(
        name=data, prequery=prequery, verbose=True
    )
    assert output.shape == (3, 3)


def test_import_with_query(test, data):
    """Is actually also dependent on mapping and typing, but a good test nonetheless."""
    query = "Performance > 0.5"
    mapping = {"Performance": "auc"}

    output, renamed, missing = test._import_utils(
        name=data, overwrite_mapping=mapping, query=query
    )
    assert output.shape == (2, 4)
    assert list(output["Performance"]) == [0.7, 0.9]


def test_old_query(test):
    filtered = test._apply_query(deepcopy(test.modelData), query={"Channel": ["Email"]})
    assert filtered.shape == (11, 11)
    assert filtered.shape != test.modelData.shape


def test_error_old_query_1(test):
    with pytest.raises(TypeError):
        test._apply_query(deepcopy(test.modelData), query=["Email"])


def test_error_old_query_2(test):
    with pytest.raises(ValueError):
        test._apply_query(deepcopy(test.modelData), query={"Channel": "Email"})


def test_import_with_wrong_query(test, data):
    query = "non-existant-column != 'unknown'"
    output, renamed, missing = test._import_utils(name=data, query=query)
    assert output.shape == (3, 3)


def test_no_subset(test, data):
    output, renamed, missing = test._import_utils(name=data, subset=False)
    assert "Junk" in output.columns
    assert output.shape == (3, 5)


def test_set_types_fails(test):
    df = pd.DataFrame(
        {
            "Positives": [1, 2, "3b"],
            "Negatives": ["0", 2, 4],
            "SnapshotTime": [
                "2022-03-01 05:00:00",
                "2022-03-01 05:00:00",
                "2022-14-01 05:00:00",
            ],
        }
    )
    output = test._set_types(df)
    assert output.shape == (3, 3)


def test_pdc_fix(test):
    pdc_extract = pd.DataFrame(
        {
            "pxObjClass": "Code-Pega-List",
            "pxResults": [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Group": "Cards",
                    "Issue": "Sales",
                    "ModelClass": "Data-Decision-Request-Customer",
                    "ModelID": "abcdefgh",
                    "ModelName": "OmniAdaptiveModel",
                    "ModelType": "AdaptiveModel",
                    "Name": "ExampleModelName",
                    "Negatives": "100.000",
                    "Performance": "0.800",
                    "Positives": "200.000",
                    "pxObjClass": "Pega-Data-CDHSnapshot-Models",
                    "pzInsKey": "NotUsed",
                    "ResponseCount": "300.000",
                    "SnapshotTime": "20220301T000000.000 GMT",
                    "TotalPositives": "200.000",
                    "TotalResponses": "300.000",
                }
            ],
        }
    )
    output = ADMDatamart(model_df=pdc_extract)
    assert output.modelData.shape == (1, 12)


# More of the end-to-end main function tests


@pytest.fixture
def cdhsample_models():
    with zipfile.ZipFile(
        "data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip"
    ) as zip:
        with zip.open("data.json") as zippedfile:
            return pd.read_json(zippedfile, lines=True)


@pytest.fixture
def cdhsample_predictors():
    with zipfile.ZipFile(
        "data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip"
    ) as zip:
        with zip.open("data.json") as zippedfile:
            return pd.read_json(zippedfile, lines=True)


def test_import_models_only(test, cdhsample_models):
    assert cdhsample_models.shape == (20, 23)
    models, preds = test.import_data(model_df=deepcopy(cdhsample_models))
    assert preds == None
    assert models.shape == (20, 11)

    assert "SuccessRate" in models.columns
    # of course not strictly necessary, but just there as a reminder
    assert "SuccessRate" not in cdh_utils._capitalize(list(cdhsample_models.columns))


def test_import_predictors_only(test, cdhsample_predictors):
    assert cdhsample_predictors.shape == (1755, 35)
    models, preds = test.import_data(predictor_df=deepcopy(cdhsample_predictors), model_filename=None)
    assert models == None
    assert preds.shape == (1755, 16)

    assert "BinResponseCount" in preds.columns
    assert "BinPropensity" in preds.columns
    assert "BinAdjustedPropensity" in preds.columns


def test_import_both(test, cdhsample_models, cdhsample_predictors):
    models, preds = test.import_data(
        model_df=deepcopy(cdhsample_models), predictor_df=deepcopy(cdhsample_predictors)
    )
    assert models is not None
    assert preds is not None
    assert preds.shape == (1755, 16)
    assert models.shape == (20, 11)

    assert "SuccessRate" in models.columns

    assert "BinResponseCount" in preds.columns
    assert "BinPropensity" in preds.columns
    assert "BinAdjustedPropensity" in preds.columns


def test_import_both_from_file_autodiscovered(test):
    models, preds = test.import_data(path="data")
    assert models is not None
    assert preds is not None
    assert preds.shape == (70735, 16)
    assert models.shape == (1047, 12)

    assert "SuccessRate" in models.columns

    assert "BinResponseCount" in preds.columns
    assert "BinPropensity" in preds.columns
    assert "BinAdjustedPropensity" in preds.columns


def test_import_both_from_file_manual(test):
    models, preds = test.import_data(
        path="data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip",
    )

    assert models is not None
    assert preds is not None
    assert preds.shape == (1755, 16)
    assert models.shape == (20, 11)

    assert "SuccessRate" in models.columns

    assert "BinResponseCount" in preds.columns
    assert "BinPropensity" in preds.columns
    assert "BinAdjustedPropensity" in preds.columns


def test_drop_BinResponseCount(test):
    models, preds = test.import_data(path="data", drop_cols=["BinResponseCount"])
    assert preds.shape == (70735, 16)


def test_init_models_only(cdhsample_models):
    output = ADMDatamart(model_df=cdhsample_models)
    assert output.modelData is not None
    assert output.modelData.shape == (20, 11)
    assert not hasattr(output, "combinedData")
    assert output.context_keys == ["Channel", "Direction", "Issue", "Group"]


def test_init_preds_only(cdhsample_predictors):
    output = ADMDatamart(model_filename=None, predictor_df=cdhsample_predictors)
    assert output.predictorData is not None
    assert output.predictorData.shape == (1755, 16)
    assert not hasattr(output, "combinedData")
    assert output.context_keys == ["Channel", "Direction", "Issue", "Group"]


def test_init_both(cdhsample_models, cdhsample_predictors):
    output = ADMDatamart(model_df=cdhsample_models, predictor_df=cdhsample_predictors)
    assert output.modelData is not None
    assert output.modelData.shape == (20, 11)
    assert output.predictorData is not None
    assert output.predictorData.shape == (1755, 16)
    assert output.combinedData.shape == (1648, 26)
    assert output.context_keys == ["Channel", "Direction", "Issue", "Group"]


def test_describe_models(test):
    test.describe_models()


def test_describe_models_without_models():
    test = ADMDatamart("data", model_filename=None)
    assert test.modelData is None
    with pytest.raises(ValueError):
        test.describe_models()


def test_model_summary(test):
    assert test.model_summary().shape == (3, 15)

