"""Testing the functionality of the Explanations class."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pdstools.explanations import Explanations


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

        assert exp.root_dir == "custom/path"
        assert exp.data_folder == "mydata"

    def test_path_object_accepted(self, tmp_path):
        """Test that Path objects are accepted for data_folder."""
        custom_data_path = tmp_path / "mydata"
        custom_data_path.mkdir(parents=True)

        exp = Explanations(data_folder=custom_data_path)

        assert exp.root_dir == str(tmp_path)
        assert exp.data_folder == "mydata"
