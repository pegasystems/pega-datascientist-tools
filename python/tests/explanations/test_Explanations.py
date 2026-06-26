"""Testing the functionality of the Explanations class"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pdstools.explanations import Explanations

basePath = Path(__file__).parent.parent.parent.parent


def _clean_up(root_dir):
    _root_dir = Path(f"{basePath}/{root_dir}")
    if _root_dir.exists():
        for file in _root_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        _root_dir.rmdir()


@pytest.fixture(scope="module")
def dummy_explanations_data():
    """Fixture to create a dummy explanations data folder."""
    os.makedirs("explanations_data", exist_ok=True)
    file_path = "explanations_data/dummy_explanations.parquet"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("dummy data")
    yield file_path

    # Cleanup after test
    if os.path.exists("explanations_data"):
        shutil.rmtree("explanations_data")


@pytest.fixture(scope="module")
def dummy_aggregate_data():
    """Fixture to create a dummy aggregate data folder."""
    os.makedirs(".tmp/aggregated_data", exist_ok=True)
    file_path = ".tmp/aggregated_data/dummy_aggregate.parquet"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("dummy data")
    yield file_path

    # Cleanup after test
    if os.path.exists(".tmp"):
        shutil.rmtree(".tmp")


class TestExplanationsDateRange:
    """Test the initialization of the Explanations class"""

    def test_date_range_only_to_date(
        self,
        dummy_explanations_data,
        dummy_aggregate_data,
    ):
        """Test initialization with only to_date provided"""
        to_date = datetime(2023, 1, 8)
        explanations = Explanations(to_date=to_date)

        expected_from_date = to_date - timedelta(days=7)
        assert explanations.from_date == expected_from_date
        assert explanations.to_date == to_date

    def test_date_range_only_from_date(
        self,
        dummy_explanations_data,
        dummy_aggregate_data,
    ):
        """Test initialization with only from_date provided"""
        from_date = datetime(2023, 1, 1)
        explanations = Explanations(from_date=from_date)

        expected_to_date = datetime.today().date()
        assert explanations.from_date == from_date
        assert explanations.to_date.date() == expected_to_date

    def test_invalid_date_range(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that invalid date range raises ValueError"""
        from_date = datetime(2023, 1, 8)
        to_date = datetime(2023, 1, 1)  # to_date before from_date

        with pytest.raises(ValueError, match="from_date cannot be after to_date"):
            Explanations(from_date=from_date, to_date=to_date)

    def test_valid_date_range(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that valid date range is accepted"""
        from_date = datetime(2023, 1, 1)
        to_date = datetime(2023, 1, 8)

        explanations = Explanations(from_date=from_date, to_date=to_date)

        assert explanations.from_date == from_date
        assert explanations.to_date == to_date

    def test_same_from_and_to_date(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that same from_date and to_date is valid"""
        date = datetime(2023, 1, 1)

        explanations = Explanations(from_date=date, to_date=date)

        assert explanations.from_date == date
        assert explanations.to_date == date

    def test_default_date_range(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that default date range is set to a week before today"""
        explanations = Explanations()

        expected_to_date = datetime.today().date()
        expected_from_date = expected_to_date - timedelta(days=7)

        assert explanations.from_date.date() == expected_from_date
        assert explanations.to_date.date() == expected_to_date


class TestPureInit:
    """The Explanations constructor must be pure config (no I/O)."""

    def test_init_does_not_touch_filesystem(self, tmp_path, monkeypatch):
        """Default-constructed Explanations must not create any files."""
        monkeypatch.chdir(tmp_path)
        exp = Explanations()
        # No directories should have been created from a bare construction.
        assert list(tmp_path.iterdir()) == []
        # Defaults should be exposed but not acted upon.
        assert exp.root_dir == ".tmp"
        assert exp.data_folder == "explanations_data"
        assert exp.data_file is None

    def test_init_rejects_positional_paths(self):
        """root_dir / data_folder / data_file must not be constructor params."""
        with pytest.raises(TypeError):
            Explanations("some_root_dir")
        with pytest.raises(TypeError):
            Explanations(data_folder="some_folder")

    def test_namespaces_attached(self):
        """The five sub-namespace facades must be present after init."""
        exp = Explanations()
        assert exp.preprocess is not None
        assert exp.aggregate is not None
        assert exp.plot is not None
        assert exp.report is not None
        assert exp.filter is not None


class TestFromLocalDirectory:
    """Exact-value tests for Explanations.from_local_directory."""

    def test_loads_real_sample_parquet(self):
        """Load the shipped sample data and verify aggregates were produced."""
        data_folder = f"{basePath}/data/explanations"
        try:
            exp = Explanations.from_local_directory(
                data_folder=data_folder,
                model_name="AdaptiveBoostCT",
                from_date=datetime(2025, 3, 28),
                to_date=datetime(2025, 3, 28),
            )

            # The classmethod must wire all path config exactly as given.
            assert exp.root_dir == ".tmp"
            assert exp.data_folder == data_folder
            assert exp.data_file is None
            assert exp.model_name == "AdaptiveBoostCT"
            assert exp.from_date == datetime(2025, 3, 28)
            assert exp.to_date == datetime(2025, 3, 28)

            # Pre-aggregation must have produced parquet files in
            # <root_dir>/aggregated_data/.
            agg_dir = Path(exp.root_dir) / "aggregated_data"
            assert agg_dir.is_dir()
            parquets = sorted(p.name for p in agg_dir.glob("*.parquet"))
            assert len(parquets) > 0, "Expected at least one aggregated parquet file"

            # The aggregate namespace must be wired to that folder.
            assert str(exp.aggregate.data_folderpath) == str(agg_dir)
        finally:
            _clean_up(".tmp")

    def test_invalid_data_folder_raises(self):
        with pytest.raises(
            ValueError,
            match="Explanations folder invalid_path does not exist or is empty.",
        ):
            Explanations.from_local_directory(data_folder="invalid_path")

    def test_data_file_overrides_data_folder(self, tmp_path, monkeypatch):
        """When data_file is given, data_folder is not consulted."""
        monkeypatch.chdir(tmp_path)
        sample = basePath / "data" / "explanations" / "AdaptiveBoostCT_20250328064604.parquet"
        try:
            exp = Explanations.from_local_directory(
                data_file=str(sample),
                model_name="AdaptiveBoostCT",
                from_date=datetime(2025, 3, 28),
                to_date=datetime(2025, 3, 28),
            )
            assert exp.data_file == str(sample)
            agg_dir = Path(exp.root_dir) / "aggregated_data"
            assert agg_dir.is_dir()
        finally:
            if Path(".tmp").exists():
                shutil.rmtree(".tmp")


class TestFromAggregates:
    """Reopening already-generated aggregates must not require manual rewiring."""

    def test_reopens_generated_aggregates(self):
        data_folder = f"{basePath}/data/explanations"
        try:
            generated = Explanations.from_local_directory(
                data_folder=data_folder,
                model_name="AdaptiveBoostCT",
                from_date=datetime(2025, 3, 28),
                to_date=datetime(2025, 3, 28),
            )
            agg_dir = Path(generated.root_dir) / "aggregated_data"

            reopened = Explanations.from_aggregates(
                agg_dir,
                data_pattern="*_BATCH_*.parquet",
                model_name="AdaptiveBoostCT",
                from_date=datetime(2025, 3, 28),
                to_date=datetime(2025, 3, 28),
            )

            assert reopened.root_dir == str(agg_dir.parent)
            assert reopened.aggregate.data_folderpath == agg_dir
            assert reopened.preprocess.data_folderpath == agg_dir
            assert reopened.report.aggregate_folder == agg_dir
            assert reopened.aggregate.data_pattern == "*_BATCH_*.parquet"
            assert reopened.model_name == "AdaptiveBoostCT"
            assert reopened.from_date == datetime(2025, 3, 28)
            assert reopened.to_date == datetime(2025, 3, 28)

            assert reopened.aggregate.get_df_overall().collect().height > 0
            assert reopened.aggregate.get_df_contextual().collect().height > 0
        finally:
            _clean_up(".tmp")

    def test_missing_aggregates_folder_raises(self):
        with pytest.raises(
            FileNotFoundError,
            match="does not exist or is not a directory",
        ):
            Explanations.from_aggregates("invalid_aggregates_path")

    def test_empty_aggregates_folder_raises(self, tmp_path):
        empty_dir = tmp_path / "aggregated_data"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="is empty"):
            Explanations.from_aggregates(empty_dir)
