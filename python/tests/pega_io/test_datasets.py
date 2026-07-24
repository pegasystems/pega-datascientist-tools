"""Testing the functionality of the built-in datasets"""

import tempfile
from datetime import datetime
from pathlib import Path

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


def test_sample_explanations_downloads_expected_files(monkeypatch, tmp_path):
    from pdstools.utils import datasets as ds_mod
    from pdstools.explanations.Explanations import Explanations

    downloaded_urls: list[str] = []

    def fake_urlretrieve(url, destination):
        downloaded_urls.append(url)
        Path(destination).write_text("placeholder")
        return (str(destination), None)

    call_kwargs: dict = {}

    def fake_from_aggregates(cls, **kwargs):
        call_kwargs.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(ds_mod, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(Explanations, "from_aggregates", classmethod(fake_from_aggregates))

    result = ds_mod.sample_explanations(target_dir=tmp_path / "agg")

    assert result == "sentinel"
    assert downloaded_urls == [
        "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/explanations/aggregated_data/OVERVIEW.parquet",
        "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/explanations/aggregated_data/BY_CONTEXT.parquet",
    ]
    assert call_kwargs["data_folder"] == tmp_path / "agg"


def test_sample_explanations_skips_download_when_cached(monkeypatch, tmp_path):
    from pdstools.utils import datasets as ds_mod
    from pdstools.explanations.Explanations import Explanations

    target_dir = tmp_path / "agg"
    target_dir.mkdir(parents=True)
    (target_dir / "OVERVIEW.parquet").write_text("cached")
    (target_dir / "BY_CONTEXT.parquet").write_text("cached")

    downloaded_urls: list[str] = []

    def fake_urlretrieve(url, destination):
        downloaded_urls.append(url)
        return (str(destination), None)

    def fake_from_aggregates(cls, **kwargs):
        return "sentinel"

    monkeypatch.setattr(ds_mod, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(Explanations, "from_aggregates", classmethod(fake_from_aggregates))

    result = ds_mod.sample_explanations(target_dir=target_dir)

    assert result == "sentinel"
    assert downloaded_urls == []


def test_sample_explanations_passes_optional_args(monkeypatch, tmp_path):
    from pdstools.utils import datasets as ds_mod
    from pdstools.explanations.Explanations import Explanations

    def fake_urlretrieve(url, destination):
        Path(destination).write_text("placeholder")
        return (str(destination), None)

    call_kwargs: dict = {}

    def fake_from_aggregates(cls, **kwargs):
        call_kwargs.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(ds_mod, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(Explanations, "from_aggregates", classmethod(fake_from_aggregates))

    from_date = datetime(2025, 1, 1)
    to_date = datetime(2025, 1, 31)
    result = ds_mod.sample_explanations(
        target_dir=tmp_path / "agg",
        model_name="MyModel",
        from_date=from_date,
        to_date=to_date,
    )

    assert result == "sentinel"
    assert call_kwargs["model_name"] == "MyModel"
    assert call_kwargs["from_date"] == from_date
    assert call_kwargs["to_date"] == to_date


def test_sample_explanations_refresh_redownloads_cached_files(monkeypatch, tmp_path):
    from pdstools.utils import datasets as ds_mod
    from pdstools.explanations.Explanations import Explanations

    target_dir = tmp_path / "agg"
    target_dir.mkdir(parents=True)
    (target_dir / "OVERVIEW.parquet").write_text("cached")
    (target_dir / "BY_CONTEXT.parquet").write_text("cached")

    downloaded_urls: list[str] = []

    def fake_urlretrieve(url, destination):
        downloaded_urls.append(url)
        Path(destination).write_text("fresh")
        return (str(destination), None)

    def fake_from_aggregates(cls, **kwargs):
        return "sentinel"

    monkeypatch.setattr(ds_mod, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(Explanations, "from_aggregates", classmethod(fake_from_aggregates))

    result = ds_mod.sample_explanations(target_dir=target_dir, refresh=True)

    assert result == "sentinel"
    assert downloaded_urls == [
        "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/explanations/aggregated_data/OVERVIEW.parquet",
        "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/explanations/aggregated_data/BY_CONTEXT.parquet",
    ]


def test_sample_explanations_uses_tempdir_when_target_dir_not_provided(monkeypatch):
    from pdstools.utils import datasets as ds_mod
    from pdstools.explanations.Explanations import Explanations

    downloaded_destinations: list[Path] = []

    def fake_urlretrieve(url, destination):
        downloaded_destinations.append(Path(destination))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text("placeholder")
        return (str(destination), None)

    call_kwargs: dict = {}

    def fake_from_aggregates(cls, **kwargs):
        call_kwargs.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(ds_mod, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(Explanations, "from_aggregates", classmethod(fake_from_aggregates))

    result = ds_mod.sample_explanations()

    expected_target = Path(tempfile.gettempdir()) / "pdstools" / "aggregated_data"
    assert result == "sentinel"
    assert call_kwargs["data_folder"] == expected_target
    assert all(destination.parent == expected_target for destination in downloaded_destinations)


def test_sample_explanations_raises_runtime_error(monkeypatch, tmp_path):
    import pytest

    from pdstools.utils import datasets as ds_mod

    monkeypatch.setattr(ds_mod, "urlretrieve", _raise)

    with pytest.raises(RuntimeError, match="Error importing the Sample Explanations"):
        ds_mod.sample_explanations(target_dir=tmp_path / "agg")
