import zipfile

import polars as pl
import pytest
from pdstools import IH, ADMDatamart, Prediction, demo_data
from pdstools.sample_data import DemoData
from pdstools.sample_data._adm import model_snapshots
from pdstools.sample_data._prediction import prediction_snapshots
from polars.testing import assert_frame_equal


@pytest.fixture(scope="module")
def demo():
    return demo_data(end="canonical", interactions=5_000)


def test_bundle_views(demo):
    assert isinstance(demo, DemoData)
    assert isinstance(demo.datamart, ADMDatamart)
    assert isinstance(demo.prediction, Prediction)
    assert isinstance(demo.ih, IH)
    assert demo.prediction.predictions.select(pl.col("SnapshotTime").n_unique()).collect().item() == 182
    assert demo.ih.data.select(pl.col("InteractionID").n_unique()).collect().item() == 5_000


def test_views_match_from_mock_data(demo):
    assert_frame_equal(
        demo.datamart.model_data.collect(),
        ADMDatamart.from_mock_data(end="canonical").model_data.collect(),
    )
    assert_frame_equal(
        demo.prediction.predictions.collect(),
        Prediction.from_mock_data(days=182, end="canonical").predictions.collect(),
    )
    assert_frame_equal(
        demo.ih.data.collect(),
        IH.from_mock_data(n=5_000, end="canonical").data.collect(),
    )


def test_cached_per_process(demo):
    assert demo_data(end="canonical", interactions=5_000) is demo
    assert demo_data(end="canonical", interactions=5_000, seed=7) is not demo
    assert demo_data(end="canonical", seed=None) is not demo_data(end="canonical", seed=None)


def test_views_are_lazy():
    fresh = DemoData(demo_data(end="canonical").world)
    fresh.prediction
    assert "datamart" not in vars(fresh)
    assert "ih" not in vars(fresh)


def test_minimum_window():
    with pytest.raises(ValueError, match="56"):
        demo_data(days=30)


def test_write_ds_export_round_trip(demo, tmp_path):
    files = demo.write_ds_export(tmp_path / "exports")
    assert {k: v.name for k, v in files.items()} == {
        "model_data": "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20000702T235959_GMT.zip",
        "predictor_data": "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20000702T235959_GMT.zip",
        "prediction_data": "Data-DM-Snapshot_pyGetSnapshot_20000702T235959_GMT.zip",
    }
    for file in files.values():
        assert zipfile.ZipFile(file).namelist() == ["data.json"]

    raw = pl.read_ndjson(zipfile.ZipFile(files["model_data"]).read("data.json"))
    assert_frame_equal(raw, model_snapshots(demo.world).collect())

    dm = ADMDatamart.from_ds_export(files["model_data"].name, files["predictor_data"].name, tmp_path / "exports")
    assert_frame_equal(dm.model_data.collect(), demo.datamart.model_data.collect())
    assert_frame_equal(dm.predictor_data.collect(), demo.datamart.predictor_data.collect())
    score_categories = (
        dm.predictor_data.filter(pl.col("PredictorName") == "ExternalModel.CustomerChurnRiskScore")
        .select(pl.col("PredictorCategory").unique())
        .collect()["PredictorCategory"]
        .to_list()
    )
    assert score_categories == ["External Model"]

    pred = Prediction.from_ds_export(files["prediction_data"].name, tmp_path / "exports")
    assert_frame_equal(pred.predictions.collect(), demo.prediction.predictions.collect())
    assert (prediction_snapshots(demo.world).collect().height, pred.predictions.collect().height) == (3458, 910)
