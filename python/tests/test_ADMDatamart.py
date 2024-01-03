"""
Testing the functionality of the ADMDatamart functions
"""


import sys
import pytest
import zipfile
import polars as pl
import itertools
from pandas.errors import UndefinedVariableError
import pathlib

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import ADMDatamart, cdh_utils
from pdstools import errors
from polars.testing import assert_frame_equal


@pl.api.register_lazyframe_namespace("shape")
class Shape:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (ldf.select(pl.first().count()).collect().item(), len(ldf.columns))


@pl.api.register_lazyframe_namespace("frame")
class Equals:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __init__(self, ldf: pl.LazyFrame):
        self.ldf = ldf

    def equal(self, other):
        df = self.ldf.with_columns(pl.col(pl.Categorical).cast(pl.Utf8)).collect()
        other = other.with_columns(pl.col(pl.Categorical).cast(pl.Utf8)).collect()
        return all(
            df.get_column(col).series_equal(other.get_column(col)) for col in df.columns
        )


@pytest.fixture
def test():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart(
        path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
        context_keys=["Issue", "Group", "Channel"],
    )


def test_basic_available_columns(test):
    df = pl.DataFrame(
        schema=["ModelID", "Issue", "Unknown", "Auc", "ResponseCount", "ModelName"]
    )
    include_cols = ["Auc"]
    drop_cols = ["Issue"]

    variables, missing = test._available_columns(
        df, include_cols=include_cols, drop_cols=drop_cols
    )
    # Basic functionality of missing & variables
    assert "Unknown" not in variables
    assert "Unknown" not in missing
    assert {"ModelID", "Issue", "ResponseCount"} not in missing
    assert "ResponseCount" in variables
    assert "Performance" not in variables
    assert "Name" not in missing

    # Testing include_cols & drop_cols
    assert "Issue" not in variables
    assert "Auc" in variables


def test_import_utils_with_importing(test):
    output, renamed, missing = test._import_utils(
        path=f"{basePath}/data",
        name="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
    )
    assert isinstance(output, pl.LazyFrame)
    assert len(output.columns) == 10
    assert output.select(pl.count()).collect().item() == 20
    assert renamed == {
        "Channel",
        "Configuration",
        "Group",
        "Issue",
        "ModelID",
        "Name",
        "Performance",
        "Positives",
        "ResponseCount",
        "SnapshotTime",
    }
    assert missing == {
        "BinIndex",
        "BinNegatives",
        "BinPositives",
        "BinResponseCount",
        "BinSymbol",
        "BinType",
        "Contents",
        "Direction",
        "EntryType",
        "PredictorName",
        "Treatment",
        "Type",
        "GroupIndex"
    }


@pytest.fixture
def data():
    return pl.LazyFrame(
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
    output, renamed, missing = test._import_utils(
        name=data,
        timestamp_fmt="%Y-%m-%d %H:%M:%S",
        typesetting_table="ADMModelSnapshot",
    )
    assert len(missing) == 20
    assert isinstance(output, pl.LazyFrame)
    output = output.collect()
    assert output.shape == (3, 3)
    assert "Name" in output.columns
    assert all(output["ModelID"] == data.collect()["pymodelid"])
    assert output.schema["SnapshotTime"].base_type() == pl.Datetime
    assert "Performance" not in output.columns
    assert renamed == {"Name", "ModelID", "SnapshotTime"}
    assert "Junk" not in output.columns
    assert "Treatment" not in output.columns


def test_import_no_subset(test, data):
    output = test._import_utils(
        name=data,
        subset=False,
        typesetting_table="ADMModelSnapshot",
    )[0]
    assert "Junk" in output.columns


def test_extract_treatment(test, data):
    mapping = {"pyname": "Name"}
    output = cdh_utils._extract_keys(data.rename(mapping).lazy()).collect()
    assert output.shape == (3, 6)
    assert list(output["Treatment"]) == ["XYZ", "xyz", None]
    assert list(output["Name"]) == ["ABC", "abc", "NormalName"]
    jsonnames = pl.LazyFrame(
        {
            "Name": [
                '{"pyName": "ABC", "pyTreatment": "XYZ"}',
                '{"pyName": "abc", "pyTreatment": "xyz"}',
                '{"pyName": "ABCD", "pyTreatment": "XYZ1"}',
            ]
        }
    )
    out = cdh_utils._extract_keys(jsonnames).collect()
    assert list(out["Name"]) == ["ABC", "abc", "ABCD"]
    assert list(out["Treatment"]) == ["XYZ", "xyz", "XYZ1"]
    # Just checking that this'll work without raising errors
    ADMDatamart(
        path="data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename=None,
        extract_keys=True,
        verbose=True,
    )


def test_apply_query(test, data):
    mods = ["model1", "model2"]
    frames = [
        # with lazyframe
        test._apply_query(data, query=pl.col("pymodelid").is_in(mods)),
        # with eager frame
        test._apply_query(data.collect(), query=pl.col("pymodelid").is_in(mods)),
        # with dict-based query
        test._apply_query(data, query={"pymodelid": mods}),
        # with pandas query
        test._apply_query(data, query=f"pymodelid.isin({mods})"),
    ]

    [
        assert_frame_equal(frame1, frame2, check_column_order=False)
        for frame1, frame2 in itertools.combinations(frames, 2)
    ]

    with pytest.raises(pl.ColumnNotFoundError):
        test._apply_query(data, pl.col("TEST") > 0)

    with pytest.raises(ValueError):
        test._apply_query(data, query={"Channel": "Email"})

    with pytest.raises(UndefinedVariableError):
        test._apply_query(data, query="UnknownCol>0").collect()


def test_set_types(test):
    df = pl.DataFrame(
        {
            "Positives": [1, 2, "3"],
            "Negatives": ["0", 2, 4],
            "Issue": ["Issue1", "Issue2", None],
            "SnapshotTime": [
                "2022-03-01 05:00:00",
                "2022-03-01 05:00:00",
                "2022-14-01 05:00:00",
            ],
        }
    )
    with pytest.raises(pl.ComputeError):
        test._set_types(
            df,
            timestamp_fmt="%Y%m%dT%H%M%S",
            strict_conversion=True,
            table="ADMModelSnapshot",
        )

    df2 = test._set_types(
        df,
        timestamp_fmt="%Y-%m-%d %H:%M:%S",
        strict_conversion=False,
        table="ADMModelSnapshot",
    )
    assert df2.shape == (3, 4)
    assert df2.dtypes == [
        pl.Float32,
        pl.Float32,
        pl.Categorical,
        pl.Datetime,
    ]

    assert df2["Positives"].to_list() == [None, None, 3]
    assert df2["Negatives"].to_list() == [0.0, None, None]
    assert df2["Issue"].to_list() == ["Issue1", "Issue2", None]
    import datetime

    assert df2["SnapshotTime"].to_list() == [
        datetime.datetime(2022, 3, 1, 5, 0),
        datetime.datetime(2022, 3, 1, 5, 0),
        None,
    ]


@pytest.fixture
def cdhsample_models():
    with zipfile.ZipFile(
        f"{basePath}/data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
        mode="r",
    ) as zip:
        with zip.open("data.json") as zippedfile:
            from io import BytesIO

            return pl.read_ndjson(BytesIO(zippedfile.read()))


@pytest.fixture
def cdhsample_predictors():
    with zipfile.ZipFile(
        f"{basePath}/data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip"
    ) as zip:
        with zip.open("data.json") as zippedfile:
            from io import BytesIO

            return pl.read_ndjson(BytesIO(zippedfile.read()))


def test_import_models_only(cdhsample_models):
    assert cdhsample_models.shape == (20, 23)
    output = ADMDatamart(
        model_df=cdhsample_models,
        predictor_filename=None,
        verbose=True,
        context_keys=["Channel", "Issue", "Group"],
    )
    assert output.modelData is not None
    assert output.modelData.shape == (20, 13)
    assert output.predictorData is None
    assert not hasattr(output, "combinedData")
    assert output.context_keys == ["Channel", "Issue", "Group"]


def test_init_preds_only(cdhsample_predictors):
    output = ADMDatamart(model_filename=None, predictor_df=cdhsample_predictors)
    assert output.predictorData is not None
    assert output.predictorData.shape == (1755, 18)
    assert not hasattr(output, "combinedData")
    assert output.context_keys == ["Channel", "Direction", "Issue", "Group"]
    with pytest.raises(ValueError):
        output.get_model_stats()


def test_init_both(cdhsample_models, cdhsample_predictors):
    output = ADMDatamart(
        model_df=cdhsample_models.lazy(),
        predictor_df=cdhsample_predictors.lazy(),
        model_filename=None,
        predictor_filename=None,
        context_keys=["Channel", "Issue", "Group"],
    )
    assert output.modelData is not None
    assert output.modelData.shape == (20, 13)
    assert output.predictorData is not None
    assert output.predictorData.shape == (1755, 18)
    assert output.combinedData.shape == (1648, 30)
    assert output.context_keys == ["Channel", "Issue", "Group"]


def test_filter_also_filters_predictorData():
    assert ADMDatamart(
        path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
        context_keys=["Issue", "Group", "Channel", "Direction"],
        verbose=True,
        query=pl.col("Channel") == "Email",
    ).predictorData.shape == (18742, 19)


def test_lazy_strategy():
    pass


def test_eagerFunctionalityFailsInLazy(test):
    with pytest.raises(errors.NotEagerError):
        ADMDatamart(
            path=f"{basePath}/data",
            model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
            predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
            import_strategy="lazy",
            extract_keys=True,
        )
    with pytest.raises(errors.NotEagerError):
        ADMDatamart(
            path=f"{basePath}/data",
            model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
            predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
            import_strategy="lazy",
            query="Channel=='Web'",
        )
    lazyADM = ADMDatamart(
        path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
        import_strategy="lazy",
    )
    with pytest.raises(errors.NotEagerError):
        lazyADM.discover_modelTypes(test.modelData)
    with pytest.raises(errors.NotEagerError):
        lazyADM.models_by_positives_df(test.modelData, allow_collect=False)
    with pytest.raises(errors.NotEagerError):
        lazyADM.pivot_df(lazyADM.modelData, allow_collect=False)


def test_explicit_plotting_engine(test):
    from pdstools.plots.plots_plotly import ADMVisualisations as plotly

    test.get_engine(plotly)
    with pytest.raises(ValueError):
        test.get_engine("UNDEFINED ENGINE")


def test_last():
    pass


def test_last_timestamp():
    pass


def test_save_data(test):
    out = test.save_data()
    imported = ADMDatamart(".")
    assert_frame_equal(imported.modelData, test.modelData, check_column_order=False)
    from os import remove

    try:
        [remove(f) for f in out]
    except PermissionError as e:
        print("Could not remove file: ", e)


@pytest.fixture
def sample_with_agb():
    return ADMDatamart(
        model_df=pl.scan_ipc(f"{basePath}/data/sample_datamart_with_AGB.arrow"),
        predictor_filename=None,
        include_cols="Modeldata",
    )


def test_discover_modelTypes(sample_with_agb):
    assert sample_with_agb.discover_modelTypes(sample_with_agb.modelData) == {
        "WebClickthroughAGB": "GbModel",
        "Web_Click_Through_Rate": "NbModel",
    }


def test_get_AGB_models(sample_with_agb):
    trees = sample_with_agb.get_AGB_models()
    assert "WebClickthroughAGB" in trees
    assert len(trees["WebClickthroughAGB"]) == 35


def test_create_sign_df():
    from pdstools import datasets

    dm = datasets.CDHSample()
    dm._create_sign_df(dm.combinedData).collect()
    dm._create_sign_df(dm.combinedData, pivot=False).collect()
    dm._create_sign_df(dm.combinedData, mask=False).collect()
    # TODO: make this a good test rather than test for no fail


def test_model_summary():
    pass


def test_pivot_df(test):
    test.pivot_df(test.combinedData, by="Configuration")
    # TODO: make this a good test rather than test for no fail


def test_response_gain_df():
    pass


def test_models_by_positives():
    pass


def test_get_model_stats():
    pass


def test_describe_models(test):
    test.describe_models()
    # TODO: make this a good test rather than test for no fail
