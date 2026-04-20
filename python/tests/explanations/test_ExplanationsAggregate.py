"""Test cases for Aggregate class that handles loading and processing of aggregate data."""

import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from pdstools.explanations import Explanations
from pdstools.explanations.ExplanationsUtils import _COL, _SPECIAL

basePath = Path(__file__).parent.parent.parent.parent


def clean_up(root_dir):
    _root_dir = Path(f"{basePath}/{root_dir}")
    if _root_dir.exists():
        for file in _root_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                # Remove subdirectories recursively
                shutil.rmtree(file)
        _root_dir.rmdir()


@pytest.fixture(scope="class")
def aggregate():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations(
        data_folder=f"{basePath}/data/explanations",
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield explanations.aggregate

    # cleanup .tmp folder
    clean_up(explanations.root_dir)


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
        """Test successful data loading."""
        aggregate.initialized
        aggregate._load_data()

        assert aggregate.initialized is True
        assert aggregate.df_contextual is not None
        assert aggregate.df_overall is not None

    def test_get_df_overall(self, aggregate):
        """Test get_df_overall returns a LazyFrame after loading."""
        df = aggregate.get_df_overall()
        assert isinstance(df, pl.LazyFrame)

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

    def test_load_data_folder_does_not_exist(self, aggregate):
        """Test handling of non-existent aggregates folder."""
        aggregate.initialized = False
        aggregate.data_folderpath = "/non/existent/path"
        with pytest.raises(FileNotFoundError):
            aggregate._load_data()
            assert aggregate.initialized is False

    @patch("polars.scan_parquet")
    def test_load_data_file_not_found_error(self, mock_scan_parquet, aggregate):
        """Test handling of file not found errors."""
        aggregate.data_folderpath = "/non/existent/path"
        mock_scan_parquet.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            aggregate._load_data()
            assert aggregate.initialized is False


class TestAggregatePredictorContributions:
    """Test cases for Aggregate contribution methods."""

    def test_get_predictor_contributions_overall_default_params(self, aggregate):
        """Test get_predictor_contributions with default parameters."""
        df = aggregate.get_predictor_contributions()
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_contributions_overall_custom_params(self, aggregate):
        """Test get_predictor_contributions with custom parameters."""
        df = aggregate.get_predictor_contributions(top_n=3)

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_n(df, 3)

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
        """Test get_predictor_contributions with default parameters."""
        df = aggregate.get_predictor_contributions(context=selected_context)
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_contributions_for_context_custom_params(
        self,
        aggregate,
        selected_context,
    ):
        """Test get_predictor_contributions with custom parameters."""
        df = aggregate.get_predictor_contributions(context=selected_context, top_n=3)

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_n(df, 3)

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
        """Test get_predictor_value_contributions with default parameters."""
        df = aggregate.get_predictor_value_contributions(predictors=predictors)
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_value_contributions_overall_custom_params(
        self,
        aggregate,
        predictors,
    ):
        """Test get_predictor_value_contributions with custom parameters."""
        df = aggregate.get_predictor_value_contributions(predictors=predictors, top_k=3)

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_k(df, 3)

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
        """Test get_predictor_value_contributions with default parameters."""
        df = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            context=selected_context,
        )
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_value_contributions_for_context_custom_params(
        self,
        aggregate,
        predictors,
        selected_context,
    ):
        """Test get_predictor_value_contributions with custom parameters."""
        df = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            context=selected_context,
            top_k=3,
        )

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_k(df, 3)

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
        with pytest.raises(TypeError, match="Unexpected filter kwargs"):
            aggregate.get_predictor_contributions(unknown_param=True)

    def test_get_predictor_value_contributions_unknown_kwarg(self, aggregate, predictors):
        with pytest.raises(TypeError, match="Unexpected filter kwargs"):
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
        # With single-bin numerics included, we should get at least as many rows
        assert df_with_single.shape[0] >= df_default.shape[0]


class TestAggregateFrequencyPct:
    """Test cases for add_frequency_pct_to_df."""

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
            _COL.PARTITON.value: pl.Utf8,
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
            df, group_by=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value]
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
            frequency_over=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
            aggregate_over=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
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
            frequency_over=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
            aggregate_over=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
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
            frequency_over=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
            aggregate_over=[_COL.PARTITON.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
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


def assert_df_has_top_n(df, top_n):
    """Assert that the DataFrame has at least top_n rows."""
    assert df.shape[0] >= top_n, f"DataFrame should have at least {top_n} rows, but has {df.shape[0]} rows."


def assert_df_has_top_k(df, top_k):
    """Assert that the DataFrame has at least top_k rows."""
    predictors_list = df.group_by(["predictor_name", "predictor_type"]).agg(pl.len()).to_dicts()
    for predictor in predictors_list:
        if predictor["predictor_type"] == "SYMBOLIC":
            assert predictor["len"] >= top_k, (
                f"Symbolic predictor {predictor['predictor_name']} should have at least {top_k} rows"
            )
