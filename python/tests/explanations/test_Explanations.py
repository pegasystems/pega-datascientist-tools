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
    return Explanations(pl.LazyFrame(), pl.LazyFrame(), **kwargs)


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

        with pytest.raises(ValueError, match=r"from_date \(2023-01-08.*\) cannot be after to_date \(2023-01-01.*\)"):
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
        make_explanations()

        assert list(tmp_path.iterdir()) == []

    def test_init_requires_both_frames(self):
        with pytest.raises(TypeError):
            Explanations()  # type: ignore[call-arg]

    def test_init_rejects_positional_paths(self):
        # Only two positional args (overall, contextual) are accepted.
        with pytest.raises(TypeError):
            Explanations(pl.LazyFrame(), pl.LazyFrame(), "some_root_dir")  # type: ignore[misc]

    def test_frames_are_stored_verbatim(self):
        overall, contextual = pl.LazyFrame({"a": [1]}), pl.LazyFrame({"b": [2]})
        exp = Explanations(overall, contextual)

        assert (exp.overall, exp.contextual) == (overall, contextual)

    def test_namespaces_attached(self):
        exp = make_explanations()

        assert (type(exp.aggregates), type(exp.plot), type(exp.report)) == (Aggregates, Plots, Reports)
        assert all(namespace.explanations is exp for namespace in (exp.aggregates, exp.plot, exp.report))


class TestFromAggregates:
    """from_aggregates owns all the I/O."""

    def test_reads_both_frames(self, tmp_path):
        for filename in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
            (tmp_path / filename).write_bytes((DATA_DIR / filename).read_bytes())

        exp = Explanations.from_aggregates(base_path=str(tmp_path))

        assert exp.overall.collect().height == 1072
        assert exp.contextual.collect().height == 8064

    def test_missing_folder_raises_immediately(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Explanations.from_aggregates(base_path=str(tmp_path / "nope"))

    def test_empty_folder_raises_immediately(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Explanations.from_aggregates(base_path=str(tmp_path))

    def test_contextual_filename_selects_a_batch(self, tmp_path):
        (tmp_path / "OVERVIEW.parquet").write_bytes((DATA_DIR / "OVERVIEW.parquet").read_bytes())
        batches = tmp_path / "batches"
        batches.mkdir()
        (batches / "BATCH_1.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())

        exp = Explanations.from_aggregates(
            base_path=str(tmp_path),
            contextual_filename="batches/BATCH_1.parquet",
        )

        assert exp.contextual.collect().height > 0

    def test_absolute_filename_ignores_base_path(self, tmp_path):
        """An absolute overall_filename / contextual_filename wins over base_path."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        for filename in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
            (other_dir / filename).write_bytes((DATA_DIR / filename).read_bytes())

        base = tmp_path / "empty_base"
        base.mkdir()

        exp = Explanations.from_aggregates(
            overall_filename=other_dir / "OVERVIEW.parquet",
            contextual_filename=other_dir / "BY_CONTEXT.parquet",
            base_path=str(base),
        )

        assert exp.overall.collect().height == 1072
        assert exp.contextual.collect().height == 8064


class TestSaveData:
    """Tests for Explanations.save_data."""

    def test_save_data_returns_paths(self, tmp_path):
        exp = Explanations.from_aggregates(base_path=DATA_DIR)
        overview_path, context_path = exp.save_data(tmp_path)

        assert overview_path == tmp_path / "OVERVIEW.parquet"
        assert context_path == tmp_path / "BY_CONTEXT.parquet"

    def test_save_data_creates_files(self, tmp_path):
        exp = Explanations.from_aggregates(base_path=DATA_DIR)
        overview_path, context_path = exp.save_data(tmp_path)

        assert overview_path.exists()
        assert context_path.exists()

    def test_save_data_round_trips_row_counts(self, tmp_path):
        exp = Explanations.from_aggregates(base_path=DATA_DIR)
        exp.save_data(tmp_path)

        reloaded = Explanations.from_aggregates(base_path=tmp_path)
        assert reloaded.overall.collect().height == 1072
        assert reloaded.contextual.collect().height == 8064

    def test_save_data_creates_directory(self, tmp_path):
        exp = Explanations.from_aggregates(base_path=DATA_DIR)
        new_dir = tmp_path / "new" / "subdir"
        exp.save_data(new_dir)

        assert (new_dir / "OVERVIEW.parquet").exists()
        assert (new_dir / "BY_CONTEXT.parquet").exists()
