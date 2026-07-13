"""Testing the functionality of the Explanations class."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pdstools.explanations import Explanations

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
        assert exp.aggregate is not None
        assert exp.plot is not None
        assert exp.report is not None

    def test_absolute_path_splits_correctly(self, tmp_path):
        """Test that absolute data_folder path is split into root_dir and data_folder."""
        custom_data_path = tmp_path / "mydata"
        custom_data_path.mkdir(parents=True)

        exp = Explanations(data_folder=str(custom_data_path))

        assert exp.root_dir == str(tmp_path)
        assert exp.data_folder == "mydata"
        assert Path(exp.root_dir) / exp.data_folder == custom_data_path

    def test_relative_path_splits_correctly(self):
        """Test that relative data_folder path is split into root_dir and data_folder."""
        exp = Explanations(data_folder="custom/path/mydata")

        assert Path(exp.root_dir).as_posix() == "custom/path"
        assert exp.data_folder == "mydata"

    def test_explicit_root_dir_with_relative_data_folder(self, tmp_path):
        """A relative aggregate folder resolves under the explicit root_dir."""
        data_dir = tmp_path / "custom_aggs"
        data_dir.mkdir()
        for filename in ("BY_CONTEXT.parquet", "OVERALL.parquet"):
            (data_dir / filename).write_bytes((DATA_DIR / filename).read_bytes())

        exp = Explanations.from_aggregates(root_dir=str(tmp_path), data_folder="custom_aggs")

        assert exp.root_dir == str(tmp_path)
        assert exp.data_folder == "custom_aggs"
        assert exp.aggregate.data_folderpath == data_dir

    def test_path_object_accepted(self, tmp_path):
        """Test that Path objects are accepted for data_folder."""
        custom_data_path = tmp_path / "mydata"
        custom_data_path.mkdir(parents=True)

        exp = Explanations(data_folder=custom_data_path)

        assert exp.root_dir == str(tmp_path)
        assert exp.data_folder == "mydata"


class TestValidateDataFolder:
    """Test the validate_data_folder method of Explanations."""

    def test_folder_exists_but_no_parquet_files(self, tmp_path):
        """Folder exists but contains no .parquet files raises FileNotFoundError."""
        data_dir = tmp_path / "aggregated_data"
        data_dir.mkdir()
        (data_dir / "readme.txt").write_text("no parquet here")

        exp = Explanations(data_folder=str(data_dir))

        with pytest.raises(FileNotFoundError, match="No parquet files found"):
            exp.validate_data_folder()
