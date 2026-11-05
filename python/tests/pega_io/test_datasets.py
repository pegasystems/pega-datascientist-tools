"""Testing the functionality of the built-in datasets"""

from datetime import datetime

import polars as pl
from pdstools import datasets, demo_data
from pdstools.utils import cdh_utils


def test_demo_datamart():
    sample = demo_data(end="canonical").datamart
    assert sample.model_data.collect().shape == (131845, 20)
    assert sample.predictor_data.collect().height == 244764

    snapshot_times = (
        sample.model_data.select("SnapshotTime").unique().sort("SnapshotTime").collect().get_column("SnapshotTime")
    )
    assert len(snapshot_times) == 182
    assert snapshot_times[0] == datetime(2000, 1, 3, 22, 3, 49)
    assert snapshot_times[-1] == datetime(2000, 7, 2, 22, 3, 49)
    predictor_snapshot_times = (
        sample.predictor_data.select("SnapshotTime").unique().collect().get_column("SnapshotTime")
    )
    assert predictor_snapshot_times.to_list() == [snapshot_times[-1]]

    period_ctr = sample.plot.over_time(
        metric="SuccessRate",
        by="ModelID",
        query=pl.col("Channel") == "Web",
        every="auto",
        cumulative=False,
        return_df=True,
    ).collect()
    assert period_ctr.get_column("SnapshotTime").n_unique() == 25

    trend = sample.plot.over_time(
        metric="SuccessRate",
        by="ModelID",
        query=pl.col("Channel") == "Web",
        every="auto",
        cumulative=False,
    )
    assert trend.layout.yaxis.title.text == "Weekly Success Rate"


def test_demo_action_trend():
    sample = demo_data(end="canonical").datamart
    assert cdh_utils.resolve_trend_interval(sample.model_data, "auto") == "1w"

    trend = (
        sample.aggregates.overall_summary(every="4w")
        .select("Actions", "Used Actions", "New Actions", "Stopped Actions")
        .collect()
    )
    assert trend["Actions"].to_list() == [84, 89, 93, 98, 103, 108, 111]
    assert trend["Used Actions"].to_list() == [84, 89, 93, 98, 99, 100, 101]
    assert trend["New Actions"].to_list() == [84, 5, 4, 5, 5, 5, 3]
    assert trend["Stopped Actions"].to_list() == [None, 0, 0, 0, 4, 4, 2]


def test_import_SampleTrees():
    datasets.sample_trees()


def test_import_SampleValueFinder():
    vf = datasets.sample_value_finder()
    assert vf.df.shape == (27133, 98)


# ---------------------------------------------------------------------------
# Exception-branch coverage
# ---------------------------------------------------------------------------


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def test_sample_trees_raises_runtime_error(monkeypatch):
    import pytest
    from pdstools.utils import datasets as ds_mod

    monkeypatch.setattr(ds_mod, "ADMTreesModel", _raise)
    with pytest.raises(RuntimeError, match="Error importing the Sample Trees"):
        ds_mod.sample_trees()


def test_sample_value_finder_raises_runtime_error(monkeypatch):
    import pytest
    from pdstools.valuefinder.ValueFinder import ValueFinder

    monkeypatch.setattr(ValueFinder, "from_ds_export", _raise)
    with pytest.raises(RuntimeError, match="Error importing the Value Finder"):
        datasets.sample_value_finder()


def test_sample_explanations_is_exported():
    """sample_explanations is importable from pdstools without a network call."""
    import pdstools

    assert callable(pdstools.sample_explanations)


def test_sample_explanations_raises_runtime_error(monkeypatch):
    import pytest
    from pdstools.explanations import Explanations

    monkeypatch.setattr(Explanations, "from_aggregates", _raise)
    with pytest.raises(RuntimeError, match="Error importing the sample Explanations dataset"):
        datasets.sample_explanations()


def test_sample_explanations_passes_args_through(monkeypatch):
    """model_name and the date window reach Explanations.from_aggregates."""
    from datetime import datetime

    from pdstools.explanations import Explanations

    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "explanations"

    monkeypatch.setattr(Explanations, "from_aggregates", _capture)
    result = datasets.sample_explanations(
        model_name="MyModel",
        from_date=datetime(2024, 3, 28),
        to_date=datetime(2025, 3, 28),
    )

    assert result == "explanations"
    assert captured["model_name"] == "MyModel"
    assert captured["from_date"] == datetime(2024, 3, 28)
    assert captured["to_date"] == datetime(2025, 3, 28)
    assert captured["base_path"].startswith("https://")
