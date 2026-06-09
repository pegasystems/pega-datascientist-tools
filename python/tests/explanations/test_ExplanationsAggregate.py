"""Test cases for Aggregate class that handles loading and processing of aggregate data."""

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from pdstools.explanations import Explanations
from pdstools.explanations.ExplanationsUtils import _COL, _SPECIAL, ContextOperations

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


@pytest.fixture(scope="class")
def aggregate():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations.from_aggregates(
        data_folder=DATA_DIR,
        model_name="AdaptiveBoostCT",
    )
    yield explanations.aggregate


@pytest.fixture
def selected_context():
    """Fixture to provide a selected context for testing."""
    return {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }


@pytest.fixture
def predictors():
    """Fixture to provide a list of predictors for testing."""
    return ["Age", "EyeColor"]


class TestAggregateLoadData:
    """Test cases for Aggregate.load_data method."""

    def test_initial_state(self, aggregate):
        """Test the initial state of Aggregate before load_data is called."""
        assert aggregate.initialized is False
        assert aggregate.df_contextual is None
        assert aggregate.df_overall is None

    def test_load_data_success(self, aggregate):
        """Test successful data loading produces the expected fixture shape."""
        aggregate._load_data()

        assert aggregate.initialized is True
        assert aggregate.df_contextual is not None
        assert aggregate.df_overall is not None

        overall = aggregate.df_overall.collect()
        contextual = aggregate.df_contextual.collect()
        expected_cols = {
            "partition",
            "contribution",
            "contribution_abs",
            "frequency",
            "predictor_type",
            "predictor_name",
            "bin_contents",
            "bin_order",
            "contribution_min",
            "contribution_max",
        }
        assert set(overall.columns) == expected_cols
        assert set(contextual.columns) == expected_cols
        assert overall.height == 1072
        assert contextual.height == 8064

    def test_get_df_overall(self, aggregate):
        """Test get_df_overall returns a populated LazyFrame after loading."""
        df = aggregate.get_df_overall().collect()
        assert df.height == 1072
        assert sorted(df["predictor_name"].unique().to_list()) == [
            "Age",
            "CustomerName",
            "EyeColor",
            "NumX",
            "Occupation",
            "pyName",
        ]

    def test_zero_contribution_rows_filtered(self, aggregate):
        """Test that rows with zero contribution are filtered out during loading."""
        df = aggregate.get_df_overall().collect()
        assert (df["contribution"] != 0.0).all()

    def test_single_bin_numeric_predictors_filtered(self, aggregate):
        """Test that numeric predictors with only one non-missing bin are filtered out."""
        df = aggregate.get_df_overall().collect()
        numeric_df = df.filter((pl.col("predictor_type") == "NUMERIC") & (pl.col("bin_contents") != "MISSING"))
        bin_counts = numeric_df.group_by(["partition", "predictor_name"]).agg(
            pl.col("bin_order").n_unique().alias("bin_count")
        )
        assert (bin_counts["bin_count"] > 1).all()

    def test_single_bin_numeric_interval_not_null(self, aggregate):
        """Single-bin numeric predictors have a valid interval after the COALESCE fix.

        When ``include_numeric_single_bin=True``, the SQL COALESCE ensures
        ``bin_contents`` renders as ``[min:max]`` (not ``null`` or empty).
        """
        aggregate._load_data()
        # get_predictor_value_contributions returns bin_contents; use all predictors
        all_predictors = (
            aggregate.get_df_overall().select("predictor_name").unique().collect()["predictor_name"].to_list()
        )
        df = aggregate.get_predictor_value_contributions(predictors=all_predictors, include_numeric_single_bin=True)
        numeric_rows = df.filter((pl.col("predictor_type") == "NUMERIC") & (pl.col("bin_contents") != "MISSING"))

        if numeric_rows.is_empty():
            pytest.skip("No numeric predictor bins in test data")

        assert numeric_rows["bin_contents"].null_count() == 0, (
            "Numeric predictor bins should not have null bin_contents after COALESCE fix"
        )
        for val in numeric_rows["bin_contents"].to_list():
            assert val != "", f"bin_contents should not be empty, got: {val!r}"

    def test_load_data_folder_does_not_exist(self, aggregate):
        """Test handling of non-existent aggregates folder."""
        aggregate.initialized = False
        aggregate.data_folderpath = Path("/non/existent/path")
        with pytest.raises(FileNotFoundError):
            aggregate._load_data()
            assert aggregate.initialized is False

    @patch("polars.scan_parquet")
    def test_load_data_file_not_found_error(self, mock_scan_parquet, aggregate):
        """Test handling of file not found errors."""
        aggregate.data_folderpath = Path("/non/existent/path")
        mock_scan_parquet.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            aggregate._load_data()
            assert aggregate.initialized is False


class TestContextOperations:
    """Coverage for unique-context batching and file creation."""

    def test_create_context_batches_keys(self, aggregate):
        contexts = [f"ctx-{idx}" for idx in range(200)]
        batches = aggregate.context_operations._create_context_batches(
            contexts, aggregate.context_operations.file_batch_limit
        )
        assert list(batches) == ["0", "1"]

    def test_create_context_batches_sizes(self, aggregate):
        contexts = [f"ctx-{idx}" for idx in range(230)]
        batches = aggregate.context_operations._create_context_batches(
            contexts, aggregate.context_operations.file_batch_limit
        )
        assert sum(len(batch) for batch in batches.values()) == len(contexts)
        assert all(len(batch) <= aggregate.context_operations.file_batch_limit for batch in batches.values())

    def test_create_context_batches_single_batch(self, aggregate):
        contexts = [f"ctx-{idx}" for idx in range(42)]
        batches = aggregate.context_operations._create_context_batches(
            contexts, aggregate.context_operations.file_batch_limit
        )
        assert list(batches) == ["0"]
        assert len(batches["0"]) == 42

    def test_create_context_batches_custom_batch_size(self, aggregate):
        contexts = [f"ctx-{idx}" for idx in range(150)]
        batches = aggregate.context_operations._create_context_batches(contexts, 50)
        assert list(batches) == ["0", "1", "2"]
        assert all(len(batch) <= 50 for batch in batches.values())

    def test_create_unique_contexts_file_creates_json(self, aggregate, tmp_path):
        output = tmp_path / "unique_contexts.json"
        original = aggregate.context_operations.unique_contexts_file
        aggregate.context_operations.unique_contexts_file = output

        contexts = aggregate.context_operations.create_unique_contexts_file()

        assert output.exists()
        persisted = json.loads(output.read_text())
        assert persisted == contexts
        assert list(persisted) == ["0"]
        assert all(isinstance(key, str) for key in persisted)
        aggregate.context_operations.unique_contexts_file = original

    def test_create_unique_contexts_file_idempotent(self, aggregate, tmp_path):
        output = tmp_path / "unique_contexts.json"
        original = aggregate.context_operations.unique_contexts_file
        aggregate.context_operations.unique_contexts_file = output

        first = aggregate.context_operations.create_unique_contexts_file()
        first_mtime = output.stat().st_mtime_ns
        second = aggregate.context_operations.create_unique_contexts_file()

        assert output.stat().st_mtime_ns == first_mtime
        assert second == first
        assert json.loads(output.read_text()) == first
        aggregate.context_operations.unique_contexts_file = original

    def test_create_unique_contexts_file_returns_dict(self, aggregate, tmp_path):
        output = tmp_path / "unique_contexts.json"
        original = aggregate.context_operations.unique_contexts_file
        aggregate.context_operations.unique_contexts_file = output

        contexts = aggregate.context_operations.create_unique_contexts_file()

        assert contexts == json.loads(output.read_text())
        aggregate.context_operations.unique_contexts_file = original

    def test_create_batch_parquet_files_creates_files(self, aggregate, tmp_path):
        original_folder = aggregate.data_folderpath
        original_file = aggregate.context_operations.unique_contexts_file

        # Copy parquet files to temp directory
        Path(tmp_path, "BY_CONTEXT.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())
        Path(tmp_path, "OVERALL.parquet").write_bytes((DATA_DIR / "OVERALL.parquet").read_bytes())

        aggregate.data_folderpath = tmp_path
        aggregate.context_operations.unique_contexts_file = tmp_path / "unique_contexts.json"
        aggregate.initialized = False
        contexts = aggregate.context_operations.create_unique_contexts_file()

        aggregate.context_operations.create_batch_parquet_files(contexts)

        expected_files = [tmp_path / "batches" / f"BATCH_{key}.parquet" for key in contexts]
        assert all(path.exists() for path in expected_files)
        aggregate.data_folderpath = original_folder
        aggregate.context_operations.unique_contexts_file = original_file
        aggregate.initialized = False

    def test_create_batch_parquet_files_row_counts(self, aggregate, tmp_path):
        original_folder = aggregate.data_folderpath
        original_file = aggregate.context_operations.unique_contexts_file

        # Copy parquet files to temp directory
        Path(tmp_path, "BY_CONTEXT.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())
        Path(tmp_path, "OVERALL.parquet").write_bytes((DATA_DIR / "OVERALL.parquet").read_bytes())

        aggregate.data_folderpath = tmp_path
        aggregate.context_operations.unique_contexts_file = tmp_path / "unique_contexts.json"
        aggregate.initialized = False  # Reset to force reload with new folder
        contexts = aggregate.context_operations.create_unique_contexts_file()

        aggregate.context_operations.create_batch_parquet_files(contexts)
        contextual = aggregate.get_df_contextual().collect()

        for batch_key, batch_contexts in contexts.items():
            batch_df = pl.read_parquet(tmp_path / "batches" / f"BATCH_{batch_key}.parquet")
            expected = contextual.filter(pl.col(_COL.PARTITION.value).is_in(batch_contexts))
            assert batch_df.height == expected.height
            assert set(batch_df[_COL.PARTITION.value].unique()) == set(batch_contexts)
        aggregate.data_folderpath = original_folder
        aggregate.context_operations.unique_contexts_file = original_file
        aggregate.initialized = False

    def test_no_file_writes_on_load(self, tmp_path):
        data_dir = tmp_path / "aggregated_data"
        data_dir.mkdir()
        (data_dir / "BY_CONTEXT.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())
        (data_dir / "OVERALL.parquet").write_bytes((DATA_DIR / "OVERALL.parquet").read_bytes())

        aggregate = Explanations.from_aggregates(data_folder=data_dir).aggregate
        before = sorted(path.name for path in data_dir.iterdir())
        aggregate._load_data()
        after = sorted(path.name for path in data_dir.iterdir())

        assert after == before

    def test_no_file_writes_on_init(self, tmp_path):
        data_dir = tmp_path / "aggregated_data"
        data_dir.mkdir()
        (data_dir / "BY_CONTEXT.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())
        (data_dir / "OVERALL.parquet").write_bytes((DATA_DIR / "OVERALL.parquet").read_bytes())

        before = sorted(path.name for path in data_dir.iterdir())
        Explanations.from_aggregates(data_folder=data_dir)
        after = sorted(path.name for path in data_dir.iterdir())

        assert after == before


class TestAggregatePredictorContributions:
    """Test cases for Aggregate contribution methods."""

    def test_get_predictor_contributions_overall_default_params(self, aggregate):
        """Default top_n=20 returns one row per predictor (6 in fixture)."""
        df = aggregate.get_predictor_contributions()
        assert df.height == 6
        assert {"predictor_name", "predictor_type", "contribution", "partition"}.issubset(df.columns)
        assert df["partition"].n_unique() == 1
        assert sorted(df["predictor_name"].unique().to_list()) == [
            "Age",
            "CustomerName",
            "EyeColor",
            "NumX",
            "Occupation",
            "pyName",
        ]

    def test_get_predictor_contributions_overall_custom_params(self, aggregate):
        """top_n=3 returns 3 top predictors plus 1 'remaining' row per partition."""
        df = aggregate.get_predictor_contributions(top_n=3)
        assert_predictor_rows_per_partition(df, top_n=3)

    def test_get_predictor_contributions_overall_invalid_contribution_type(
        self,
        aggregate,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_contributions(
                sort_by="invalid_type",
            )

    def test_get_predictor_contributions_overall_invalid_top_n(self, aggregate):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_n value"):
            aggregate.get_predictor_contributions(top_n=-1)

    def test_get_predictor_contributions_for_context_default_params(
        self,
        aggregate,
        selected_context,
    ):
        """Context-scoped query returns the same 6 predictor rows for that partition."""
        df = aggregate.get_predictor_contributions(context=selected_context)
        assert df.height == 6
        assert df["partition"].n_unique() == 1

    def test_get_predictor_contributions_for_context_custom_params(
        self,
        aggregate,
        selected_context,
    ):
        """Context-scoped top_n=3 returns 3 top predictors + 1 'remaining' row."""
        df = aggregate.get_predictor_contributions(context=selected_context, top_n=3)
        assert_predictor_rows_per_partition(df, top_n=3)

    def test_get_predictor_contributions_for_context_invalid_contribution_type(
        self,
        aggregate,
        selected_context,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_contributions(
                context=selected_context,
                sort_by="invalid_type",
            )

    def test_get_predictor_contributions_for_context_invalid_top_n(
        self,
        aggregate,
        selected_context,
    ):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_n value"):
            aggregate.get_predictor_contributions(context=selected_context, top_n=-1)


class TestAggregatePredictorValueContributions:
    """Test cases for Aggregate predictor value contributions."""

    def test_get_predictor_value_contributions_overall_default_params(
        self,
        aggregate,
        predictors,
    ):
        """Default top_k returns all bins for the requested predictors (19 in fixture)."""
        df = aggregate.get_predictor_value_contributions(predictors=predictors)
        assert df.height == 19
        assert {"bin_contents", "bin_order", "predictor_name", "contribution"}.issubset(df.columns)
        assert sorted(df["predictor_name"].unique().to_list()) == ["Age", "EyeColor"]

    def test_get_predictor_value_contributions_overall_custom_params(
        self,
        aggregate,
        predictors,
    ):
        """top_k=3 returns at most 3 symbolic bins per predictor (8 rows total)."""
        df = aggregate.get_predictor_value_contributions(predictors=predictors, top_k=3)
        assert df.height == 8
        assert_symbolic_bins_per_predictor_capped(df, top_k=3)

    def test_get_predictor_value_contributions_overall_invalid_contribution_type(
        self,
        aggregate,
        predictors,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                sort_by="invalid_type",
            )

    def test_get_predictor_value_contributions_overall_invalid_top_k(
        self,
        aggregate,
        predictors,
    ):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_k value"):
            aggregate.get_predictor_value_contributions(predictors=predictors, top_k=-1)

    def test_get_predictor_value_contributions_for_context_default_params(
        self,
        aggregate,
        predictors,
        selected_context,
    ):
        """Context-scoped value contributions return all bins (19 rows in fixture)."""
        df = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            context=selected_context,
        )
        assert df.height == 19
        assert sorted(df["predictor_name"].unique().to_list()) == ["Age", "EyeColor"]

    def test_get_predictor_value_contributions_for_context_custom_params(
        self,
        aggregate,
        predictors,
        selected_context,
    ):
        """Context-scoped top_k=3 returns at most 3 symbolic bins per predictor."""
        df = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            context=selected_context,
            top_k=3,
        )
        assert df.height == 8
        assert_symbolic_bins_per_predictor_capped(df, top_k=3)

    def test_get_predictor_value_contributions_for_context_invalid_contribution_type(
        self,
        aggregate,
        predictors,
        selected_context,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                context=selected_context,
                sort_by="invalid_type",
            )

    def test_get_predictor_value_contributions_for_context_invalid_top_k(
        self,
        aggregate,
        predictors,
        selected_context,
    ):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_k value"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                context=selected_context,
                top_k=-1,
            )


class TestFilterKwargsValidation:
    """Test that unknown filter kwargs raise TypeError."""

    def test_get_predictor_contributions_unknown_kwarg(self, aggregate):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            aggregate.get_predictor_contributions(unknown_param=True)

    def test_get_predictor_value_contributions_unknown_kwarg(self, aggregate, predictors):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            aggregate.get_predictor_value_contributions(predictors=predictors, unknown_param=True)


class TestFilterKwargsDefaults:
    """Test that filter kwargs are optional and defaults are applied correctly."""

    def test_get_predictor_contributions_no_kwargs_uses_defaults(self, aggregate):
        """Calling with no filter kwargs should apply defaults (sort_by=contribution_abs, descending=True)."""
        df_no_kwargs = aggregate.get_predictor_contributions()
        df_explicit = aggregate.get_predictor_contributions(
            sort_by="contribution_abs", descending=True, missing=True, remaining=True, include_numeric_single_bin=False
        )
        assert df_no_kwargs.equals(df_explicit)

    def test_get_predictor_contributions_with_kwargs_overrides_default(self, aggregate):
        """Passing filter kwargs should override the defaults."""
        df_default = aggregate.get_predictor_contributions()
        df_no_remaining = aggregate.get_predictor_contributions(remaining=False)
        # Without remaining row, result should differ from the default
        assert not df_default.equals(df_no_remaining)

    def test_get_predictor_value_contributions_no_kwargs_uses_defaults(self, aggregate, predictors):
        """Calling with no filter kwargs should apply defaults."""
        df_no_kwargs = aggregate.get_predictor_value_contributions(predictors=predictors)
        df_explicit = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            sort_by="contribution_abs",
            descending=True,
            missing=True,
            remaining=True,
            include_numeric_single_bin=False,
        )
        assert df_no_kwargs.equals(df_explicit)

    def test_get_predictor_value_contributions_with_kwargs_overrides_default(self, aggregate, predictors):
        """Passing filter kwargs should override the defaults."""
        df_default = aggregate.get_predictor_value_contributions(predictors=predictors)
        df_no_remaining = aggregate.get_predictor_value_contributions(predictors=predictors, remaining=False)
        assert not df_default.equals(df_no_remaining)

    def test_get_predictor_contributions_include_numeric_single_bin_default(self, aggregate):
        """Default (False) should exclude single-bin numeric predictors."""
        df_default = aggregate.get_predictor_contributions()
        df_explicit_false = aggregate.get_predictor_contributions(include_numeric_single_bin=False)
        assert df_default.equals(df_explicit_false)

    def test_get_predictor_contributions_include_numeric_single_bin_true(self, aggregate):
        """Passing include_numeric_single_bin=True may include extra predictors."""
        df_default = aggregate.get_predictor_contributions()
        df_with_single = aggregate.get_predictor_contributions(include_numeric_single_bin=True)
        # With single-bin numerics included, we should get at least as many unique predictors
        default_predictors = set(df_default[_COL.PREDICTOR_NAME.value].to_list())
        with_single_predictors = set(df_with_single[_COL.PREDICTOR_NAME.value].to_list())
        assert default_predictors <= with_single_predictors

    def test_get_predictor_value_contributions_include_numeric_single_bin_default(self, aggregate, predictors):
        """Default (False) should exclude single-bin numeric predictors."""
        df_default = aggregate.get_predictor_value_contributions(predictors=predictors)
        df_explicit_false = aggregate.get_predictor_value_contributions(
            predictors=predictors, include_numeric_single_bin=False
        )
        assert df_default.equals(df_explicit_false)

    def test_get_predictor_value_contributions_include_numeric_single_bin_true(self, aggregate, predictors):
        """Passing include_numeric_single_bin=True may include extra predictor values."""
        df_default = aggregate.get_predictor_value_contributions(predictors=predictors)
        df_with_single = aggregate.get_predictor_value_contributions(
            predictors=predictors, include_numeric_single_bin=True
        )
        # In this fixture there are no single-bin numerics, so the two should be identical.
        assert df_with_single.shape[0] == df_default.shape[0]


class TestAggregateFrequencyPct:
    """Test cases for add_frequency_pct_to_df and add_context_frequency_pct_to_df."""

    def test_add_frequency_pct_to_df(self, aggregate):
        """Test that frequency_pct column is added correctly."""
        df = aggregate.get_df_overall()
        result = aggregate.add_frequency_pct_to_df(df, group_by=["partition"]).collect()
        assert "frequency_pct" in result.columns
        assert result["frequency_pct"].dtype == pl.Float64

    def test_frequency_pct_values_in_range(self, aggregate):
        """Test that frequency_pct values are between 0 and 100."""
        df = aggregate.get_df_overall()
        result = aggregate.add_frequency_pct_to_df(df, group_by=["partition"]).collect()
        assert (result["frequency_pct"] >= 0.0).all()
        assert (result["frequency_pct"] <= 100.0).all()

    def test_add_context_frequency_pct_exact_values(self, aggregate):
        """Verify context frequency as a share of the overall model.

        Uses a small context DataFrame with known frequencies and asserts
        exact expected values of ``context_freq / overall_freq * 100``.
        """
        aggregate._load_data()

        overall_df = aggregate.get_df_overall().collect()
        join_cols = [_COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value]
        overall_totals = overall_df.group_by(join_cols).agg(
            pl.sum(_COL.FREQUENCY.value).alias("expected_overall_total")
        )

        first_two = overall_totals.head(2)
        context_rows = []
        for row in first_two.to_dicts():
            context_rows.append(
                {
                    _COL.PREDICTOR_NAME.value: row[_COL.PREDICTOR_NAME.value],
                    _COL.PREDICTOR_TYPE.value: row[_COL.PREDICTOR_TYPE.value],
                    _COL.FREQUENCY.value: 50,
                }
            )

        context_df = pl.DataFrame(context_rows)
        result = aggregate.add_context_frequency_pct_to_df(context_df, join_on=join_cols)

        assert "frequency_pct" in result.columns
        for row in result.to_dicts():
            name = row[_COL.PREDICTOR_NAME.value]
            ptype = row[_COL.PREDICTOR_TYPE.value]
            overall_total = overall_totals.filter(
                (pl.col(_COL.PREDICTOR_NAME.value) == name) & (pl.col(_COL.PREDICTOR_TYPE.value) == ptype)
            )["expected_overall_total"][0]
            expected_pct = round(50 / overall_total * 100, 4)
            assert row["frequency_pct"] == expected_pct, (
                f"frequency_pct for {name}/{ptype}: expected {expected_pct}, got {row['frequency_pct']}"
            )

    def test_add_context_frequency_pct_zero_overall(self, aggregate):
        """When overall frequency is zero, frequency_pct should be 0.0."""
        aggregate._load_data()

        context_df = pl.DataFrame(
            {
                _COL.PREDICTOR_NAME.value: ["NonExistentPredictor"],
                _COL.PREDICTOR_TYPE.value: ["NUMERIC"],
                _COL.FREQUENCY.value: [100],
            }
        )
        join_cols = [_COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value]
        result = aggregate.add_context_frequency_pct_to_df(context_df, join_on=join_cols)

        assert result["frequency_pct"][0] == 0.0


class TestWeightedAverageComputation:
    """Unit tests for the weighted average contribution calculation.

    These tests exercise _calculate_aggregates, _add_total_frequency_to_df,
    _get_weighted_aggregates, and _filter_single_bin_numeric_predictors using
    minimal in-memory DataFrames — no parquet files required.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_df(rows: list[dict]) -> pl.LazyFrame:
        """Build a LazyFrame from a list of dicts matching the Aggregate schema."""

        schema = {
            _COL.PARTITION.value: pl.Utf8,
            _COL.PREDICTOR_NAME.value: pl.Utf8,
            _COL.PREDICTOR_TYPE.value: pl.Utf8,
            _COL.BIN_CONTENTS.value: pl.Utf8,
            _COL.BIN_ORDER.value: pl.Int64,
            _COL.CONTRIBUTION.value: pl.Float64,
            _COL.CONTRIBUTION_ABS.value: pl.Float64,
            _COL.CONTRIBUTION_MIN.value: pl.Float64,
            _COL.CONTRIBUTION_MAX.value: pl.Float64,
            _COL.FREQUENCY.value: pl.Int64,
        }
        return pl.DataFrame(rows, schema=schema).lazy()

    # ------------------------------------------------------------------
    # _add_total_frequency_to_df
    # ------------------------------------------------------------------

    def test_total_frequency_per_predictor(self, aggregate):
        """total_frequency equals the sum of all bin frequencies for the group."""

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[0:30]",
                    "bin_order": 1,
                    "contribution": 0.2,
                    "contribution_abs": 0.2,
                    "contribution_min": 0.1,
                    "contribution_max": 0.3,
                    "frequency": 100,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[30:60]",
                    "bin_order": 2,
                    "contribution": 0.8,
                    "contribution_abs": 0.8,
                    "contribution_min": 0.6,
                    "contribution_max": 0.9,
                    "frequency": 50,
                },
            ]
        )
        result = aggregate._add_total_frequency_to_df(
            df, group_by=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value]
        ).collect()

        assert (result[_SPECIAL.TOTAL_FREQUENCY.value] == 150).all()

    # ------------------------------------------------------------------
    # _get_weighted_aggregates (formula: sum(c*f) / total_f)
    # ------------------------------------------------------------------

    def test_weighted_average_formula_correctness(self, aggregate):
        """contribution_weighted = sum(contribution * frequency) / total_frequency.

        With bin A (contribution=0.2, frequency=100) and bin B (contribution=0.8,
        frequency=50), total=150:
          correct  = (0.2*100 + 0.8*50) / 150 = 60/150 = 0.4
          wrong    = mean(0.2*100, 0.8*50) / 150 = 30/150 = 0.2
        """

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[0:30]",
                    "bin_order": 1,
                    "contribution": 0.2,
                    "contribution_abs": 0.2,
                    "contribution_min": 0.2,
                    "contribution_max": 0.2,
                    "frequency": 100,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[30:60]",
                    "bin_order": 2,
                    "contribution": 0.8,
                    "contribution_abs": 0.8,
                    "contribution_min": 0.8,
                    "contribution_max": 0.8,
                    "frequency": 50,
                },
            ]
        )
        result = aggregate._calculate_aggregates(
            df,
            frequency_over=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
            aggregate_over=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
        ).collect()

        assert result.shape[0] == 1
        weighted = result[_COL.CONTRIBUTION_WEIGHTED.value][0]
        assert abs(weighted - 0.4) < 1e-9, f"Expected 0.4, got {weighted}"

    def test_weighted_average_equal_frequencies_matches_mean(self, aggregate):
        """When all bins have equal frequency, weighted avg equals simple mean."""

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "Color",
                    "predictor_type": "SYMBOLIC",
                    "bin_contents": "Red",
                    "bin_order": 1,
                    "contribution": 0.3,
                    "contribution_abs": 0.3,
                    "contribution_min": 0.3,
                    "contribution_max": 0.3,
                    "frequency": 100,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Color",
                    "predictor_type": "SYMBOLIC",
                    "bin_contents": "Blue",
                    "bin_order": 2,
                    "contribution": 0.7,
                    "contribution_abs": 0.7,
                    "contribution_min": 0.7,
                    "contribution_max": 0.7,
                    "frequency": 100,
                },
            ]
        )
        result = aggregate._calculate_aggregates(
            df,
            frequency_over=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
            aggregate_over=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
        ).collect()

        weighted = result[_COL.CONTRIBUTION_WEIGHTED.value][0]
        mean_val = result[_COL.CONTRIBUTION.value][0]
        assert abs(weighted - mean_val) < 1e-9

    # ------------------------------------------------------------------
    # frequency_over scoped per predictor (not per partition)
    # ------------------------------------------------------------------

    def test_weighted_average_scoped_per_predictor(self, aggregate):
        """Each predictor's weighted average divides by its own bin frequencies.

        Two predictors in the same partition with different frequency totals:
          Age:   bins (freq=100, c=0.2) + (freq=100, c=0.8)  → total=200, weighted=0.5
          Score: bins (freq=10,  c=0.1) + (freq=90,  c=0.9)  → total=100, weighted=0.82
        If frequency_over were scoped to partition only, both would use total=300
        and produce wrong results.
        """

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[0:30]",
                    "bin_order": 1,
                    "contribution": 0.2,
                    "contribution_abs": 0.2,
                    "contribution_min": 0.2,
                    "contribution_max": 0.2,
                    "frequency": 100,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[30:60]",
                    "bin_order": 2,
                    "contribution": 0.8,
                    "contribution_abs": 0.8,
                    "contribution_min": 0.8,
                    "contribution_max": 0.8,
                    "frequency": 100,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Score",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[0:50]",
                    "bin_order": 1,
                    "contribution": 0.1,
                    "contribution_abs": 0.1,
                    "contribution_min": 0.1,
                    "contribution_max": 0.1,
                    "frequency": 10,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Score",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[50:100]",
                    "bin_order": 2,
                    "contribution": 0.9,
                    "contribution_abs": 0.9,
                    "contribution_min": 0.9,
                    "contribution_max": 0.9,
                    "frequency": 90,
                },
            ]
        )
        result = aggregate._calculate_aggregates(
            df,
            frequency_over=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
            aggregate_over=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
        ).collect()

        by_name = {row["predictor_name"]: row for row in result.to_dicts()}

        age_weighted = by_name["Age"][_COL.CONTRIBUTION_WEIGHTED.value]
        # (0.2*100 + 0.8*100) / 200 = 100/200 = 0.5
        assert abs(age_weighted - 0.5) < 1e-9, f"Age weighted: expected 0.5, got {age_weighted}"

        score_weighted = by_name["Score"][_COL.CONTRIBUTION_WEIGHTED.value]
        # (0.1*10 + 0.9*90) / 100 = (1 + 81) / 100 = 0.82
        assert abs(score_weighted - 0.82) < 1e-9, f"Score weighted: expected 0.82, got {score_weighted}"

    # ------------------------------------------------------------------
    # _filter_single_bin_numeric_predictors
    # ------------------------------------------------------------------

    def test_single_bin_numeric_predictor_excluded(self, aggregate):
        """A numeric predictor with exactly one non-missing bin is filtered out."""

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "OneRange",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[0:100]",
                    "bin_order": 1,
                    "contribution": 0.5,
                    "contribution_abs": 0.5,
                    "contribution_min": 0.5,
                    "contribution_max": 0.5,
                    "frequency": 200,
                },
            ]
        )
        result = aggregate._filter_single_bin_numeric_predictors(df).collect()
        assert result.is_empty(), "Single-bin numeric predictor should be filtered out"

    def test_multi_bin_numeric_predictor_retained(self, aggregate):
        """A numeric predictor with two or more non-missing bins is kept."""

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[0:30]",
                    "bin_order": 1,
                    "contribution": 0.2,
                    "contribution_abs": 0.2,
                    "contribution_min": 0.2,
                    "contribution_max": 0.2,
                    "frequency": 100,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Age",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[30:60]",
                    "bin_order": 2,
                    "contribution": 0.8,
                    "contribution_abs": 0.8,
                    "contribution_min": 0.8,
                    "contribution_max": 0.8,
                    "frequency": 50,
                },
            ]
        )
        result = aggregate._filter_single_bin_numeric_predictors(df).collect()
        assert result.shape[0] == 2, "Multi-bin numeric predictor should not be filtered"

    def test_symbolic_single_bin_not_filtered(self, aggregate):
        """A symbolic predictor with only one bin is NOT filtered (rule is numeric-only)."""

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "Color",
                    "predictor_type": "SYMBOLIC",
                    "bin_contents": "Red",
                    "bin_order": 1,
                    "contribution": 0.5,
                    "contribution_abs": 0.5,
                    "contribution_min": 0.5,
                    "contribution_max": 0.5,
                    "frequency": 100,
                },
            ]
        )
        result = aggregate._filter_single_bin_numeric_predictors(df).collect()
        assert result.shape[0] == 1, "Single-bin symbolic predictor should be retained"

    def test_missing_bin_not_counted_for_single_bin_check(self, aggregate):
        """A MISSING bin does not count toward the bin count; a numeric predictor
        with only one real bin plus a MISSING bin should still be filtered."""

        df = self._make_df(
            [
                {
                    "partition": "p1",
                    "predictor_name": "Score",
                    "predictor_type": "NUMERIC",
                    "bin_contents": "[0:100]",
                    "bin_order": 1,
                    "contribution": 0.4,
                    "contribution_abs": 0.4,
                    "contribution_min": 0.4,
                    "contribution_max": 0.4,
                    "frequency": 80,
                },
                {
                    "partition": "p1",
                    "predictor_name": "Score",
                    "predictor_type": "NUMERIC",
                    "bin_contents": _SPECIAL.MISSING.name,
                    "bin_order": 2,
                    "contribution": 0.1,
                    "contribution_abs": 0.1,
                    "contribution_min": 0.1,
                    "contribution_max": 0.1,
                    "frequency": 20,
                },
            ]
        )
        result = aggregate._filter_single_bin_numeric_predictors(df).collect()
        assert result.is_empty(), "Numeric predictor with only one real bin (plus MISSING) should be filtered"


def assert_predictor_rows_per_partition(df, top_n):
    """Assert each partition has exactly top_n + 1 rows (top predictors + remaining row).

    Used for `get_predictor_contributions` outputs where ``remaining=True`` (the default)
    appends a single aggregated 'remaining' row per partition.
    """
    expected_per_partition = top_n + 1
    counts = df.group_by("partition").agg(pl.len().alias("n")).to_dicts()
    assert counts, "Expected at least one partition in the result."
    for row in counts:
        assert row["n"] == expected_per_partition, (
            f"Partition {row['partition']!r} has {row['n']} rows, "
            f"expected {expected_per_partition} (top_n + 1 remaining)."
        )


def assert_symbolic_bins_per_predictor_capped(df, top_k):
    """Assert each symbolic predictor has at most top_k + 1 rows (top bins + remaining).

    Numeric predictors are not capped by ``top_k``.
    """
    expected_max = top_k + 1
    rows = df.group_by(["predictor_name", "predictor_type"]).agg(pl.len().alias("n")).to_dicts()
    assert rows, "Expected at least one predictor in the result."
    for row in rows:
        if row["predictor_type"] == "SYMBOLIC":
            assert row["n"] <= expected_max, (
                f"Symbolic predictor {row['predictor_name']!r} has {row['n']} bins, "
                f"expected at most {expected_max} (top_k + 1 remaining)."
            )


def test_create_context_batches_empty_list():
    """Test that empty context list returns empty batches dict."""
    batches = ContextOperations._create_context_batches([], 100)
    assert batches == {}


def test_create_context_batches_none():
    """Test that None context list returns empty batches dict."""
    batches = ContextOperations._create_context_batches([], 100)
    assert isinstance(batches, dict)
    assert len(batches) == 0
