"""Testing the functionality of the built-in datasets"""

from pdstools import datasets


def test_import_CDHSample():
    Sample = datasets.cdh_sample()
    assert Sample.model_data.shape == (1047, 30)


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


def test_cdh_sample_raises_runtime_error(monkeypatch):
    import pytest
    from pdstools.adm.ADMDatamart import ADMDatamart

    monkeypatch.setattr(ADMDatamart, "from_ds_export", _raise)
    with pytest.raises(RuntimeError, match="Error importing CDH Sample"):
        datasets.cdh_sample()


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
