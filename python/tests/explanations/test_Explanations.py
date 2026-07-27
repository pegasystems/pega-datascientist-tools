"""Testing the functionality of the Explanations class."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from pdstools.explanations import Explanations
from pdstools.explanations.Aggregates import Aggregates
from pdstools.explanations.Plots import Plots
from pdstools.explanations.Reports import Reports

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


def make_explanations(**kwargs) -> Explanations:
    """Build an Explanations over empty frames, for tests that don't touch data."""
    return Explanations(pl.LazyFrame(), pl.LazyFrame(), data_folderpath=".tmp/aggregated_data", **kwargs)


class TestExplanationsDateRange:
    """Test the initialization of the Explanations class."""

    def test_date_range_only_to_date(self):
        to_date = datetime(2023, 1, 8)
        explanations = make_explanations(to_date=to_date)

        expected_from_date = to_date - timedelta(days=7)
        assert explanations.from_date == expected_from_date
        assert explanations.to_date == to_date

    def test_date_range_only_from_date(self):
        from_date = datetime(2023, 1, 1)
        explanations = make_explanations(from_date=from_date)

        expected_to_date = datetime.today().date()
        assert explanations.from_date == from_date
        assert explanations.to_date.date() == expected_to_date

    def test_invalid_date_range(self):
        from_date = datetime(2023, 1, 8)
        to_date = datetime(2023, 1, 1)

        with pytest.raises(ValueError, match="from_date cannot be after to_date"):
            make_explanations(from_date=from_date, to_date=to_date)

    def test_valid_date_range(self):
        from_date = datetime(2023, 1, 1)
        to_date = datetime(2023, 1, 8)

        explanations = make_explanations(from_date=from_date, to_date=to_date)

        assert explanations.from_date == from_date
        assert explanations.to_date == to_date

    def test_same_from_and_to_date(self):
        date = datetime(2023, 1, 1)

        explanations = make_explanations(from_date=date, to_date=date)

        assert explanations.from_date == date
        assert explanations.to_date == date

    def test_default_date_range(self):
        explanations = make_explanations()

        expected_to_date = datetime.today().date()
        expected_from_date = expected_to_date - timedelta(days=7)

        assert explanations.from_date.date() == expected_from_date
        assert explanations.to_date.date() == expected_to_date


class TestPureInit:
    """The Explanations constructor must be pure config (no I/O)."""

    def test_init_does_not_touch_filesystem(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp = make_explanations()

        assert list(tmp_path.iterdir()) == []
        assert exp.root_dir == ".tmp"

    def test_init_requires_both_frames(self):
        with pytest.raises(TypeError):
            Explanations(data_folderpath=".tmp")  # type: ignore[call-arg]

    def test_init_rejects_positional_paths(self):
        with pytest.raises(TypeError):
            Explanations(pl.LazyFrame(), pl.LazyFrame(), "some_root_dir")  # type: ignore[misc]

    def test_frames_are_stored_verbatim(self):
        overall, contextual = pl.LazyFrame({"a": [1]}), pl.LazyFrame({"b": [2]})
        exp = Explanations(overall, contextual, data_folderpath=".tmp")

        assert (exp.overall, exp.contextual) == (overall, contextual)

    def test_namespaces_attached(self):
        exp = make_explanations()

        assert (type(exp.aggregates), type(exp.plot), type(exp.report)) == (Aggregates, Plots, Reports)
        assert all(namespace.explanations is exp for namespace in (exp.aggregates, exp.plot, exp.report))


class TestResolveDataFolder:
    """Path resolution is a pure function, exercised without touching disk."""

    def test_absolute_data_folder_is_used_as_is(self, tmp_path):
        """An absolute data_folder is used verbatim, ignoring root_dir."""
        custom = tmp_path / "mydata"

        assert Explanations._resolve_data_folder(str(tmp_path), str(custom)) == custom

    def test_path_object_accepted(self, tmp_path):
        custom = tmp_path / "mydata"

        assert Explanations._resolve_data_folder(None, custom) == custom

    def test_relative_data_folder_resolves_against_cwd(self, tmp_path, monkeypatch):
        """Without an explicit root_dir, a relative data_folder is CWD-relative."""
        monkeypatch.chdir(tmp_path)

        assert (
            Explanations._resolve_data_folder(None, "custom/path/mydata") == (tmp_path / "custom/path/mydata").resolve()
        )

    def test_explicit_root_dir_with_relative_data_folder(self, tmp_path):
        """A relative aggregates folder resolves under the explicit root_dir."""
        assert Explanations._resolve_data_folder(str(tmp_path), "custom_aggs") == tmp_path / "custom_aggs"

    def test_nested_relative_data_folder_under_explicit_root(self, tmp_path):
        """Multi-segment relative folders keep every segment under root_dir."""
        assert (
            Explanations._resolve_data_folder(str(tmp_path), "nested/aggregated_data")
            == (tmp_path / "nested/aggregated_data").resolve()
        )

    def test_parent_relative_data_folder(self, tmp_path, monkeypatch):
        """A ``../``-prefixed relative folder resolves upward from the CWD."""
        workdir = tmp_path / "a" / "b"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        assert (
            Explanations._resolve_data_folder(None, "../../data/aggregated_data")
            == (tmp_path / "data/aggregated_data").resolve()
        )

    def test_default_data_folder_goes_under_root_dir(self, tmp_path, monkeypatch):
        """The default data_folder is root-relative even without an explicit root_dir."""
        monkeypatch.chdir(tmp_path)

        assert Explanations._resolve_data_folder(None, "aggregated_data") == tmp_path / ".tmp" / "aggregated_data"


class TestFromAggregates:
    """from_aggregates owns all the I/O."""

    def test_reads_both_frames(self, tmp_path):
        for filename in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
            (tmp_path / filename).write_bytes((DATA_DIR / filename).read_bytes())

        exp = Explanations.from_aggregates(data_folder=str(tmp_path))

        assert exp.data_folderpath == tmp_path
        assert exp.overall.collect().height > 0
        assert exp.contextual.collect().height > 0

    def test_missing_folder_raises_immediately(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Explanations.from_aggregates(data_folder=str(tmp_path / "nope"))

    def test_empty_folder_raises_immediately(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Explanations.from_aggregates(data_folder=str(tmp_path))

    def test_contextual_file_selects_a_batch(self, tmp_path):
        (tmp_path / "OVERVIEW.parquet").write_bytes((DATA_DIR / "OVERVIEW.parquet").read_bytes())
        batches = tmp_path / "batches"
        batches.mkdir()
        (batches / "BATCH_1.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())

        exp = Explanations.from_aggregates(
            data_folder=str(tmp_path),
            contextual_file="batches/BATCH_1.parquet",
        )

        assert exp.contextual.collect().height > 0
