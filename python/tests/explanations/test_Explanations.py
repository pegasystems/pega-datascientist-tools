"""Testing the functionality of the Explanations class."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pdstools.explanations import Explanations
from pdstools.explanations.Aggregates import Aggregates
from pdstools.explanations.Plots import Plots
from pdstools.explanations.Reports import Reports

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


class TestExplanationsDateRange:
    """Test the initialization of the Explanations class."""

    def test_date_range_only_to_date(self):
        to_date = datetime(2023, 1, 8)
        explanations = Explanations(to_date=to_date)

        expected_from_date = to_date - timedelta(days=7)
        assert explanations.from_date == expected_from_date
        assert explanations.to_date == to_date

    def test_date_range_only_from_date(self):
        from_date = datetime(2023, 1, 1)
        explanations = Explanations(from_date=from_date)

        expected_to_date = datetime.today().date()
        assert explanations.from_date == from_date
        assert explanations.to_date.date() == expected_to_date

    def test_invalid_date_range(self):
        from_date = datetime(2023, 1, 8)
        to_date = datetime(2023, 1, 1)

        with pytest.raises(ValueError, match="from_date cannot be after to_date"):
            Explanations(from_date=from_date, to_date=to_date)

    def test_valid_date_range(self):
        from_date = datetime(2023, 1, 1)
        to_date = datetime(2023, 1, 8)

        explanations = Explanations(from_date=from_date, to_date=to_date)

        assert explanations.from_date == from_date
        assert explanations.to_date == to_date

    def test_same_from_and_to_date(self):
        date = datetime(2023, 1, 1)

        explanations = Explanations(from_date=date, to_date=date)

        assert explanations.from_date == date
        assert explanations.to_date == date

    def test_default_date_range(self):
        explanations = Explanations()

        expected_to_date = datetime.today().date()
        expected_from_date = expected_to_date - timedelta(days=7)

        assert explanations.from_date.date() == expected_from_date
        assert explanations.to_date.date() == expected_to_date


class TestPureInit:
    """The Explanations constructor must be pure config (no I/O)."""

    def test_init_does_not_touch_filesystem(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp = Explanations()

        assert list(tmp_path.iterdir()) == []
        assert exp.root_dir == ".tmp"
        assert Path(exp.root_dir) / exp.data_folder == Path(".tmp/aggregated_data")

    def test_init_rejects_positional_paths(self):
        with pytest.raises(TypeError):
            Explanations("some_root_dir")  # type: ignore[misc]

    def test_namespaces_attached(self):
        exp = Explanations()

        assert (type(exp.aggregates), type(exp.plot), type(exp.report)) == (Aggregates, Plots, Reports)
        assert all(namespace.explanations is exp for namespace in (exp.aggregates, exp.plot, exp.report))

    def test_absolute_data_folder_is_used_as_is(self, tmp_path):
        """An absolute data_folder is used verbatim, ignoring root_dir."""
        custom_data_path = tmp_path / "mydata"
        custom_data_path.mkdir(parents=True)

        exp = Explanations(data_folder=str(custom_data_path))

        assert exp.data_folderpath == custom_data_path

    def test_relative_data_folder_resolves_against_cwd(self, tmp_path, monkeypatch):
        """Without an explicit root_dir, a relative data_folder is CWD-relative."""
        monkeypatch.chdir(tmp_path)

        exp = Explanations(data_folder="custom/path/mydata")

        assert exp.data_folderpath == (tmp_path / "custom/path/mydata").resolve()

    def test_relative_data_folder_is_cwd_independent_once_built(self, tmp_path, monkeypatch):
        """data_folderpath is resolved once, so later chdir cannot break it."""
        monkeypatch.chdir(tmp_path)
        exp = Explanations(data_folder="custom/path/mydata")
        expected = exp.data_folderpath

        monkeypatch.chdir(tmp_path.parent)

        assert exp.data_folderpath == expected
        assert exp.data_folderpath.is_absolute()

    def test_explicit_root_dir_with_relative_data_folder(self, tmp_path):
        """A relative aggregates folder resolves under the explicit root_dir."""
        data_dir = tmp_path / "custom_aggs"
        data_dir.mkdir()
        for filename in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
            (data_dir / filename).write_bytes((DATA_DIR / filename).read_bytes())

        exp = Explanations.from_aggregates(root_dir=str(tmp_path), data_folder="custom_aggs")

        assert exp.data_folderpath == data_dir
        assert exp.aggregates.data_folderpath == data_dir

    def test_nested_relative_data_folder_under_explicit_root(self, tmp_path):
        """Multi-segment relative folders keep every segment under root_dir."""
        exp = Explanations(root_dir=str(tmp_path), data_folder="nested/aggregated_data")

        assert exp.data_folderpath == (tmp_path / "nested/aggregated_data").resolve()

    def test_parent_relative_data_folder(self, tmp_path, monkeypatch):
        """A ``../``-prefixed relative folder resolves upward from the CWD."""
        workdir = tmp_path / "a" / "b"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        exp = Explanations(data_folder="../../data/aggregated_data")

        assert exp.data_folderpath == (tmp_path / "data/aggregated_data").resolve()

    def test_path_object_accepted(self, tmp_path):
        """Test that Path objects are accepted for data_folder."""
        custom_data_path = tmp_path / "mydata"
        custom_data_path.mkdir(parents=True)

        exp = Explanations(data_folder=custom_data_path)

        assert exp.data_folderpath == custom_data_path
