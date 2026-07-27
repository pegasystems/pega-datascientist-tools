"""Test cases for Aggregates class that handles loading and processing of aggregates data."""

import json
from pathlib import Path

import polars as pl
import pytest
from pdstools.explanations import Explanations
from pdstools.explanations.ContextOperations import ContextOperations
from pdstools.explanations._constants import MISSING, REMAINING, TOTAL_FREQUENCY
from pdstools.explanations.Schema import AGGREGATE_SCHEMA

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


@pytest.fixture(scope="class")
def aggregates():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations.from_aggregates(
        data_folder=DATA_DIR,
        model_name="AdaptiveBoostCT",
    )
    yield explanations.aggregates


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
    """Test cases for the lazily-read ``contextual`` / ``overall`` frames."""

    def test_frames_are_lazy_and_cached(self, aggregates):
        """Frames stay lazy and are computed once, not on every access."""
        # A LazyFrame exposes the schema without reading the row groups.
        assert aggregates.explanations.overall.collect_schema().names() == list(AGGREGATE_SCHEMA)
        assert aggregates.explanations.overall is aggregates.explanations.overall
        assert aggregates.explanations.contextual is aggregates.explanations.contextual

    def test_load_data_success(self, aggregates):
        """Test successful data loading produces the expected fixture shape."""
        overall = aggregates.explanations.overall.collect()
        contextual = aggregates.explanations.contextual.collect()
        expected_cols = {
            "context_partition",
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

    def test_overall_predictors(self, aggregates):
        """The overall frame exposes every predictor in the fixture."""
        df = aggregates.explanations.overall.collect()
        assert df.height == 1072
        assert sorted(df["predictor_name"].unique().to_list()) == [
            "Age",
            "CustomerName",
            "EyeColor",
            "NumX",
            "Occupation",
            "pyName",
        ]

    def test_zero_contribution_rows_filtered(self, aggregates):
        """Test that rows with zero contribution are filtered out during loading."""
        df = aggregates.explanations.overall.collect()
        assert (df["contribution"] != 0.0).all()

    def test_single_bin_numeric_predictors_filtered(self, aggregates):
        """Test that numeric predictors with only one non-missing bin are filtered out."""
        df = aggregates.explanations.overall.collect()
        numeric_df = df.filter((pl.col("predictor_type") == "NUMERIC") & (pl.col("bin_contents") != "MISSING"))
        bin_counts = numeric_df.group_by(["context_partition", "predictor_name"]).agg(
            pl.col("bin_order").n_unique().alias("bin_count")
        )
        assert (bin_counts["bin_count"] > 1).all()

    def test_single_bin_numeric_interval_not_null(self, aggregates):
        """Single-bin numeric predictors have a valid interval after the COALESCE fix.

        When ``include_numeric_single_bin=True``, the SQL COALESCE ensures
        ``bin_contents`` renders as ``[min:max]`` (not ``null`` or empty).
        """
        # predictor_value_contributions returns bin_contents; use all predictors
        all_predictors = (
            aggregates.explanations.overall.select("predictor_name").unique().collect()["predictor_name"].to_list()
        )
        df = aggregates.predictor_value_contributions(predictors=all_predictors, include_numeric_single_bin=True)
        numeric_rows = df.filter((pl.col("predictor_type") == "NUMERIC") & (pl.col("bin_contents") != "MISSING"))

        if numeric_rows.is_empty():
            pytest.skip("No numeric predictor bins in test data")

        assert numeric_rows["bin_contents"].null_count() == 0, (
            "Numeric predictor bins should not have null bin_contents after COALESCE fix"
        )
        for val in numeric_rows["bin_contents"].to_list():
            assert val != "", f"bin_contents should not be empty, got: {val!r}"

    # test_missing_folder_raises_on_access and test_empty_folder_raises_on_access
    # are covered by TestFromAggregates in test_Explanations.py; duplicates deleted.


class TestContextOperations:
    """Coverage for unique-context batching and file creation."""

    def test_create_context_batches_keys(self, aggregates):
        contexts = [f"ctx-{idx}" for idx in range(200)]
        batches = aggregates.context_operations._create_context_batches(
            contexts, aggregates.context_operations.file_batch_limit
        )
        assert list(batches) == ["0", "1"]

    def test_create_context_batches_sizes(self, aggregates):
        contexts = [f"ctx-{idx}" for idx in range(230)]
        batches = aggregates.context_operations._create_context_batches(
            contexts, aggregates.context_operations.file_batch_limit
        )
        assert sum(len(batch) for batch in batches.values()) == len(contexts)
        assert all(len(batch) <= aggregates.context_operations.file_batch_limit for batch in batches.values())

    def test_create_context_batches_single_batch(self, aggregates):
        contexts = [f"ctx-{idx}" for idx in range(42)]
        batches = aggregates.context_operations._create_context_batches(
            contexts, aggregates.context_operations.file_batch_limit
        )
        assert list(batches) == ["0"]
        assert len(batches["0"]) == 42

    def test_create_context_batches_custom_batch_size(self, aggregates):
        contexts = [f"ctx-{idx}" for idx in range(150)]
        batches = aggregates.context_operations._create_context_batches(contexts, 50)
        assert list(batches) == ["0", "1", "2"]
        assert all(len(batch) <= 50 for batch in batches.values())

    @staticmethod
    def _redirect_to_tmp(aggregates, tmp_path, monkeypatch):
        """Point the aggregates at a copy of the sample data inside tmp_path."""
        for name in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
            (tmp_path / name).write_bytes((DATA_DIR / name).read_bytes())
        monkeypatch.setattr(aggregates.explanations, "data_folderpath", tmp_path)
        return tmp_path / "unique_contexts.json"

    def test_create_unique_contexts_file_creates_json(self, aggregates, tmp_path, monkeypatch):
        output = self._redirect_to_tmp(aggregates, tmp_path, monkeypatch)

        contexts = aggregates.context_operations.create_unique_contexts_file()

        assert output.exists()
        persisted = json.loads(output.read_text())
        assert persisted == contexts
        assert list(persisted) == ["0"]
        assert all(isinstance(key, str) for key in persisted)

    def test_create_unique_contexts_file_is_deterministic(self, aggregates, tmp_path, monkeypatch):
        """Repeated calls produce the same mapping, in the same order."""
        output = self._redirect_to_tmp(aggregates, tmp_path, monkeypatch)

        first = aggregates.context_operations.create_unique_contexts_file()
        second = aggregates.context_operations.create_unique_contexts_file()

        assert second == first
        assert json.loads(output.read_text()) == first

    def test_create_unique_contexts_file_overwrites_stale_mapping(self, aggregates, tmp_path, monkeypatch):
        """The JSON is an interchange artifact, not a cache: a stale file is replaced."""
        output = self._redirect_to_tmp(aggregates, tmp_path, monkeypatch)
        output.write_text(json.dumps({"99": ["a-context-that-no-longer-exists"]}))

        contexts = aggregates.context_operations.create_unique_contexts_file()

        assert list(contexts) == ["0"]
        assert json.loads(output.read_text()) == contexts

    def test_batch_limit_change_takes_effect(self, aggregates, tmp_path, monkeypatch):
        """A changed PDSTOOLS_FILE_BATCH_LIMIT must not be masked by an existing file."""
        output = self._redirect_to_tmp(aggregates, tmp_path, monkeypatch)
        co = aggregates.context_operations

        co.create_unique_contexts_file()
        monkeypatch.setattr(co, "file_batch_limit", 5)
        contexts = co.create_unique_contexts_file()

        # 20 unique contexts in the sample data, 5 per batch.
        assert [len(batch) for batch in contexts.values()] == [5, 5, 5, 5]
        assert json.loads(output.read_text()) == contexts

    def test_create_unique_contexts_file_returns_dict(self, aggregates, tmp_path, monkeypatch):
        output = self._redirect_to_tmp(aggregates, tmp_path, monkeypatch)

        contexts = aggregates.context_operations.create_unique_contexts_file()

        assert contexts == json.loads(output.read_text())

    def test_create_batch_parquet_files_creates_files(self, aggregates, tmp_path, monkeypatch):
        self._redirect_to_tmp(aggregates, tmp_path, monkeypatch)
        contexts = aggregates.context_operations.create_unique_contexts_file()

        aggregates.context_operations.create_batch_parquet_files(contexts)

        expected_files = [tmp_path / "batches" / f"BATCH_{key}.parquet" for key in contexts]
        assert all(path.exists() for path in expected_files)

    def test_create_batch_parquet_files_row_counts(self, aggregates, tmp_path, monkeypatch):
        self._redirect_to_tmp(aggregates, tmp_path, monkeypatch)
        contexts = aggregates.context_operations.create_unique_contexts_file()

        aggregates.context_operations.create_batch_parquet_files(contexts)
        contextual = aggregates.explanations.contextual.collect()

        for batch_key, batch_contexts in contexts.items():
            batch_df = pl.read_parquet(tmp_path / "batches" / f"BATCH_{batch_key}.parquet")
            expected = contextual.filter(pl.col("context_partition").is_in(batch_contexts))
            assert batch_df.height == expected.height
            assert set(batch_df["context_partition"].unique()) == set(batch_contexts)

    def test_no_file_writes_on_load(self, tmp_path):
        data_dir = tmp_path / "aggregated_data"
        data_dir.mkdir()
        (data_dir / "BY_CONTEXT.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())
        (data_dir / "OVERVIEW.parquet").write_bytes((DATA_DIR / "OVERVIEW.parquet").read_bytes())

        aggregates = Explanations.from_aggregates(data_folder=data_dir).aggregates
        before = sorted(path.name for path in data_dir.iterdir())
        aggregates.explanations.overall.collect()
        aggregates.explanations.contextual.collect()
        after = sorted(path.name for path in data_dir.iterdir())

        assert after == before

    def test_no_file_writes_on_init(self, tmp_path):
        data_dir = tmp_path / "aggregated_data"
        data_dir.mkdir()
        (data_dir / "BY_CONTEXT.parquet").write_bytes((DATA_DIR / "BY_CONTEXT.parquet").read_bytes())
        (data_dir / "OVERVIEW.parquet").write_bytes((DATA_DIR / "OVERVIEW.parquet").read_bytes())

        before = sorted(path.name for path in data_dir.iterdir())
        Explanations.from_aggregates(data_folder=data_dir)
        after = sorted(path.name for path in data_dir.iterdir())

        assert after == before


class TestAggregateAndContextOperationHelpers:
    """Coverage for helper paths in Aggregates and ContextOperations."""

    def test_unique_contexts_returns_contexts(self, aggregates):
        contexts = aggregates.unique_contexts()
        assert len(contexts) == 20
        assert {context["pyName"] for context in contexts} == {f"P{i}" for i in range(1, 21)}
        assert {tuple(context) for context in contexts} == {
            ("pyChannel", "pyDirection", "pyGroup", "pyIssue", "pyName"),
        }

    def test_internal_predictor_contributions_filters_predictors(self, aggregates, selected_context):
        df = aggregates._predictor_contributions(
            contexts=[selected_context],
            predictors=["Age"],
            remaining=False,
        )
        assert set(df["predictor_name"].unique().to_list()) == {"Age"}

    def test_get_base_df_defaults_to_overall(self, aggregates):
        """Without a context filter, the base frame is the overall frame."""
        assert aggregates._get_base_df() is aggregates.explanations.overall

    def test_context_operations_context_keys(self, aggregates):
        keys = aggregates.context_operations.context_keys
        assert keys
        assert all(key.startswith("py") for key in keys)

    def test_context_operations_get_df_default_and_with_partition(self, aggregates):
        df_default = aggregates.context_operations.get_df()
        assert "context_partition" not in df_default.columns

        df_with_partition = aggregates.context_operations.get_df(with_partition_col=True)
        assert "context_partition" in df_with_partition.columns

    def test_context_operations_get_list_and_context_string(self, aggregates, selected_context):
        contexts = aggregates.context_operations.get_list([selected_context], with_partition_col=False)
        assert len(contexts) == 1
        assert contexts[0]["pyChannel"] == selected_context["pyChannel"]

        context_str = ContextOperations.get_context_info_str(selected_context, sep="|")
        assert "PegaBatch" in context_str
        assert "|" in context_str


class TestAggregatePredictorContributions:
    """Test cases for Aggregates contribution methods."""

    def test_predictor_contributions_overall_default_params(self, aggregates):
        """Default top_n=20 returns one row per predictor (6 in fixture)."""
        df = aggregates.predictor_contributions()
        assert df.height == 6
        assert {"predictor_name", "predictor_type", "contribution", "context_partition"}.issubset(df.columns)
        assert df["context_partition"].n_unique() == 1
        assert sorted(df["predictor_name"].unique().to_list()) == [
            "Age",
            "CustomerName",
            "EyeColor",
            "NumX",
            "Occupation",
            "pyName",
        ]

    def test_predictor_contributions_overall_custom_params(self, aggregates):
        """top_n=3 returns 3 top predictors plus 1 'remaining' row per partition."""
        df = aggregates.predictor_contributions(top_n=3)
        assert_predictor_rows_per_partition(df, top_n=3)

    def test_predictor_contributions_overall_invalid_contribution_type(
        self,
        aggregates,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregates.predictor_contributions(
                sort_by="invalid_type",
            )

    def test_predictor_contributions_overall_invalid_top_n(self, aggregates):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_n value"):
            aggregates.predictor_contributions(top_n=-1)

    def test_predictor_contributions_for_context_default_params(
        self,
        aggregates,
        selected_context,
    ):
        """Context-scoped query returns the same 6 predictor rows for that partition."""
        df = aggregates.predictor_contributions(context=selected_context)
        assert df.height == 6
        assert df["context_partition"].n_unique() == 1

    def test_predictor_contributions_for_context_custom_params(
        self,
        aggregates,
        selected_context,
    ):
        """Context-scoped top_n=3 returns 3 top predictors + 1 'remaining' row."""
        df = aggregates.predictor_contributions(context=selected_context, top_n=3)
        assert_predictor_rows_per_partition(df, top_n=3)

    def test_predictor_contributions_for_context_invalid_contribution_type(
        self,
        aggregates,
        selected_context,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregates.predictor_contributions(
                context=selected_context,
                sort_by="invalid_type",
            )

    def test_predictor_contributions_for_context_invalid_top_n(
        self,
        aggregates,
        selected_context,
    ):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_n value"):
            aggregates.predictor_contributions(context=selected_context, top_n=-1)


class TestAggregatePredictorValueContributions:
    """Test cases for Aggregates predictor value contributions."""

    def test_predictor_value_contributions_overall_default_params(
        self,
        aggregates,
        predictors,
    ):
        """Default top_k returns all bins for the requested predictors (19 in fixture)."""
        df = aggregates.predictor_value_contributions(predictors=predictors)
        assert df.height == 19
        assert {"bin_contents", "bin_order", "predictor_name", "contribution"}.issubset(df.columns)
        assert sorted(df["predictor_name"].unique().to_list()) == ["Age", "EyeColor"]

    def test_predictor_value_contributions_overall_custom_params(
        self,
        aggregates,
        predictors,
    ):
        """top_k=3 returns at most 3 symbolic bins per predictor, plus the forced
        MISSING bin and the 'remaining' rollup for each (9 rows total).

        Was 8 before the ``missing`` flag was repaired: ``_get_missing_predictor_values_df``
        filtered on ``predictor_name`` instead of ``bin_contents``, so the forced
        MISSING bin was never actually added.
        """
        df = aggregates.predictor_value_contributions(predictors=predictors, top_k=3)
        assert df.height == 9
        assert sorted(df.filter(pl.col("bin_contents") == MISSING)["predictor_name"].to_list()) == [
            "Age",
            "EyeColor",
        ]
        assert_symbolic_bins_per_predictor_capped(df, top_k=3)

    def test_predictor_value_contributions_overall_invalid_contribution_type(
        self,
        aggregates,
        predictors,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregates.predictor_value_contributions(
                predictors=predictors,
                sort_by="invalid_type",
            )

    def test_predictor_value_contributions_overall_invalid_top_k(
        self,
        aggregates,
        predictors,
    ):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_k value"):
            aggregates.predictor_value_contributions(predictors=predictors, top_k=-1)

    def test_predictor_value_contributions_for_context_default_params(
        self,
        aggregates,
        predictors,
        selected_context,
    ):
        """Context-scoped value contributions return all bins (19 rows in fixture)."""
        df = aggregates.predictor_value_contributions(
            predictors=predictors,
            context=selected_context,
        )
        assert df.height == 19
        assert sorted(df["predictor_name"].unique().to_list()) == ["Age", "EyeColor"]

    def test_predictor_value_contributions_for_context_custom_params(
        self,
        aggregates,
        predictors,
        selected_context,
    ):
        """Context-scoped top_k=3 returns at most 3 symbolic bins per predictor, plus
        the forced MISSING bin and the 'remaining' rollup for each (10 rows total).

        Was 8 before the ``missing`` flag was repaired; see the overall variant.
        """
        df = aggregates.predictor_value_contributions(
            predictors=predictors,
            context=selected_context,
            top_k=3,
        )
        assert df.height == 10
        assert sorted(df.filter(pl.col("bin_contents") == MISSING)["predictor_name"].to_list()) == [
            "Age",
            "EyeColor",
        ]
        assert_symbolic_bins_per_predictor_capped(df, top_k=3)

    def test_predictor_value_contributions_for_context_invalid_contribution_type(
        self,
        aggregates,
        predictors,
        selected_context,
    ):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregates.predictor_value_contributions(
                predictors=predictors,
                context=selected_context,
                sort_by="invalid_type",
            )

    def test_predictor_value_contributions_for_context_invalid_top_k(
        self,
        aggregates,
        predictors,
        selected_context,
    ):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_k value"):
            aggregates.predictor_value_contributions(
                predictors=predictors,
                context=selected_context,
                top_k=-1,
            )


class TestFilterKwargsValidation:
    """Test that unknown filter kwargs raise TypeError."""

    def test_predictor_contributions_unknown_kwarg(self, aggregates):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            aggregates.predictor_contributions(unknown_param=True)

    def test_predictor_value_contributions_unknown_kwarg(self, aggregates, predictors):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            aggregates.predictor_value_contributions(predictors=predictors, unknown_param=True)


class TestFilterKwargsDefaults:
    """Test that filter kwargs are optional and defaults are applied correctly."""

    def test_predictor_contributions_no_kwargs_uses_defaults(self, aggregates):
        """Calling with no filter kwargs should apply defaults (sort_by=contribution_abs, descending=True)."""
        df_no_kwargs = aggregates.predictor_contributions()
        df_explicit = aggregates.predictor_contributions(
            sort_by="contribution_abs", descending=True, missing=True, remaining=True, include_numeric_single_bin=False
        )
        assert df_no_kwargs.equals(df_explicit)

    def test_predictor_contributions_with_kwargs_overrides_default(self, aggregates):
        """Passing filter kwargs should override the defaults."""
        df_default = aggregates.predictor_contributions()
        df_no_remaining = aggregates.predictor_contributions(remaining=False)
        # Without remaining row, result should differ from the default
        assert not df_default.equals(df_no_remaining)

    def test_predictor_value_contributions_no_kwargs_uses_defaults(self, aggregates, predictors):
        """Calling with no filter kwargs should apply defaults."""
        df_no_kwargs = aggregates.predictor_value_contributions(predictors=predictors)
        df_explicit = aggregates.predictor_value_contributions(
            predictors=predictors,
            sort_by="contribution_abs",
            descending=True,
            missing=True,
            remaining=True,
            include_numeric_single_bin=False,
        )
        assert df_no_kwargs.equals(df_explicit)

    def test_predictor_value_contributions_with_kwargs_overrides_default(self, aggregates, predictors):
        """Passing filter kwargs should override the defaults."""
        df_default = aggregates.predictor_value_contributions(predictors=predictors)
        df_no_remaining = aggregates.predictor_value_contributions(predictors=predictors, remaining=False)
        assert not df_default.equals(df_no_remaining)

    def test_predictor_contributions_include_numeric_single_bin_default(self, aggregates):
        """Default (False) should exclude single-bin numeric predictors."""
        df_default = aggregates.predictor_contributions()
        df_explicit_false = aggregates.predictor_contributions(include_numeric_single_bin=False)
        assert df_default.equals(df_explicit_false)

    def test_predictor_contributions_include_numeric_single_bin_true(self, aggregates):
        """Passing include_numeric_single_bin=True may include extra predictors."""
        df_default = aggregates.predictor_contributions()
        df_with_single = aggregates.predictor_contributions(include_numeric_single_bin=True)
        # With single-bin numerics included, we should get at least as many unique predictors
        default_predictors = set(df_default["predictor_name"].to_list())
        with_single_predictors = set(df_with_single["predictor_name"].to_list())
        assert default_predictors <= with_single_predictors

    def test_predictor_value_contributions_include_numeric_single_bin_default(self, aggregates, predictors):
        """Default (False) should exclude single-bin numeric predictors."""
        df_default = aggregates.predictor_value_contributions(predictors=predictors)
        df_explicit_false = aggregates.predictor_value_contributions(
            predictors=predictors, include_numeric_single_bin=False
        )
        assert df_default.equals(df_explicit_false)

    def test_predictor_value_contributions_include_numeric_single_bin_true(self, aggregates, predictors):
        """Passing include_numeric_single_bin=True may include extra predictor values."""
        df_default = aggregates.predictor_value_contributions(predictors=predictors)
        df_with_single = aggregates.predictor_value_contributions(
            predictors=predictors, include_numeric_single_bin=True
        )
        # In this fixture there are no single-bin numerics, so the two should be identical.
        assert df_with_single.shape[0] == df_default.shape[0]


class TestAggregateFrequencyPct:
    """Test cases for _add_frequency_pct and _add_context_frequency_pct."""

    def test__add_frequency_pct(self, aggregates):
        """Test that frequency_pct column is added correctly."""
        df = aggregates.explanations.overall
        result = aggregates._add_frequency_pct(df, group_by=["context_partition"]).collect()
        assert "frequency_pct" in result.columns
        assert result["frequency_pct"].dtype == pl.Float64

    def test_frequency_pct_values_in_range(self, aggregates):
        """Test that frequency_pct values are between 0 and 100."""
        df = aggregates.explanations.overall
        result = aggregates._add_frequency_pct(df, group_by=["context_partition"]).collect()
        assert (result["frequency_pct"] >= 0.0).all()
        assert (result["frequency_pct"] <= 100.0).all()

    def test_add_context_frequency_pct_exact_values(self, aggregates):
        """Verify context frequency as a share of the overall model.

        Uses a small context DataFrame with known frequencies and asserts
        exact expected values of ``context_freq / overall_freq * 100``.
        """

        overall_df = aggregates.explanations.overall.collect()
        join_cols = ["predictor_name", "predictor_type"]
        overall_totals = overall_df.group_by(join_cols).agg(pl.sum("frequency").alias("expected_overall_total"))

        first_two = overall_totals.head(2)
        context_rows = []
        for row in first_two.to_dicts():
            context_rows.append(
                {
                    "predictor_name": row["predictor_name"],
                    "predictor_type": row["predictor_type"],
                    "frequency": 50,
                }
            )

        context_df = pl.DataFrame(context_rows)
        result = aggregates._add_context_frequency_pct(context_df, join_on=join_cols)

        assert "frequency_pct" in result.columns
        for row in result.to_dicts():
            name = row["predictor_name"]
            ptype = row["predictor_type"]
            overall_total = overall_totals.filter(
                (pl.col("predictor_name") == name) & (pl.col("predictor_type") == ptype)
            )["expected_overall_total"][0]
            expected_pct = round(50 / overall_total * 100, 4)
            assert row["frequency_pct"] == expected_pct, (
                f"frequency_pct for {name}/{ptype}: expected {expected_pct}, got {row['frequency_pct']}"
            )

    def test_add_context_frequency_pct_zero_overall(self, aggregates):
        """When overall frequency is zero, frequency_pct should be 0.0."""

        context_df = pl.DataFrame(
            {
                "predictor_name": ["NonExistentPredictor"],
                "predictor_type": ["NUMERIC"],
                "frequency": [100],
            }
        )
        join_cols = ["predictor_name", "predictor_type"]
        result = aggregates._add_context_frequency_pct(context_df, join_on=join_cols)

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
        """Build a LazyFrame from a list of dicts matching the Aggregates schema."""

        schema = {
            "context_partition": pl.Utf8,
            "predictor_name": pl.Utf8,
            "predictor_type": pl.Utf8,
            "bin_contents": pl.Utf8,
            "bin_order": pl.Int64,
            "contribution": pl.Float64,
            "contribution_abs": pl.Float64,
            "contribution_min": pl.Float64,
            "contribution_max": pl.Float64,
            "frequency": pl.Int64,
        }
        return pl.DataFrame(rows, schema=schema).lazy()

    # ------------------------------------------------------------------
    # _add_total_frequency_to_df
    # ------------------------------------------------------------------

    def test_total_frequency_per_predictor(self, aggregates):
        """total_frequency equals the sum of all bin frequencies for the group."""

        df = self._make_df(
            [
                {
                    "context_partition": "p1",
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
                    "context_partition": "p1",
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
        result = aggregates._add_total_frequency_to_df(
            df, group_by=["context_partition", "predictor_name", "predictor_type"]
        ).collect()

        assert (result[TOTAL_FREQUENCY] == 150).all()

    # ------------------------------------------------------------------
    # _get_weighted_aggregates (formula: sum(c*f) / total_f)
    # ------------------------------------------------------------------

    def test_weighted_average_formula_correctness(self, aggregates):
        """contribution_weighted = sum(contribution * frequency) / total_frequency.

        With bin A (contribution=0.2, frequency=100) and bin B (contribution=0.8,
        frequency=50), total=150:
          correct  = (0.2*100 + 0.8*50) / 150 = 60/150 = 0.4
          wrong    = mean(0.2*100, 0.8*50) / 150 = 30/150 = 0.2
        """

        df = self._make_df(
            [
                {
                    "context_partition": "p1",
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
                    "context_partition": "p1",
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
        result = aggregates._calculate_aggregates(
            df,
            frequency_over=["context_partition", "predictor_name", "predictor_type"],
            aggregate_over=["context_partition", "predictor_name", "predictor_type"],
        ).collect()

        assert result.shape[0] == 1
        weighted = result["contribution_weighted"][0]
        assert abs(weighted - 0.4) < 1e-9, f"Expected 0.4, got {weighted}"

    def test_weighted_average_equal_frequencies_matches_mean(self, aggregates):
        """When all bins have equal frequency, weighted avg equals simple mean."""

        df = self._make_df(
            [
                {
                    "context_partition": "p1",
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
                    "context_partition": "p1",
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
        result = aggregates._calculate_aggregates(
            df,
            frequency_over=["context_partition", "predictor_name", "predictor_type"],
            aggregate_over=["context_partition", "predictor_name", "predictor_type"],
        ).collect()

        weighted = result["contribution_weighted"][0]
        mean_val = result["contribution"][0]
        assert abs(weighted - mean_val) < 1e-9

    # ------------------------------------------------------------------
    # frequency_over scoped per predictor (not per partition)
    # ------------------------------------------------------------------

    def test_weighted_average_scoped_per_predictor(self, aggregates):
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
                    "context_partition": "p1",
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
                    "context_partition": "p1",
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
                    "context_partition": "p1",
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
                    "context_partition": "p1",
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
        result = aggregates._calculate_aggregates(
            df,
            frequency_over=["context_partition", "predictor_name", "predictor_type"],
            aggregate_over=["context_partition", "predictor_name", "predictor_type"],
        ).collect()

        by_name = {row["predictor_name"]: row for row in result.to_dicts()}

        age_weighted = by_name["Age"]["contribution_weighted"]
        # (0.2*100 + 0.8*100) / 200 = 100/200 = 0.5
        assert abs(age_weighted - 0.5) < 1e-9, f"Age weighted: expected 0.5, got {age_weighted}"

        score_weighted = by_name["Score"]["contribution_weighted"]
        # (0.1*10 + 0.9*90) / 100 = (1 + 81) / 100 = 0.82
        assert abs(score_weighted - 0.82) < 1e-9, f"Score weighted: expected 0.82, got {score_weighted}"

    # ------------------------------------------------------------------
    # _filter_single_bin_numeric_predictors
    # ------------------------------------------------------------------

    def test_single_bin_numeric_predictor_excluded(self, aggregates):
        """A numeric predictor with exactly one non-missing bin is filtered out."""

        df = self._make_df(
            [
                {
                    "context_partition": "p1",
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
        result = aggregates._filter_single_bin_numeric_predictors(df).collect()
        assert result.is_empty(), "Single-bin numeric predictor should be filtered out"

    def test_multi_bin_numeric_predictor_retained(self, aggregates):
        """A numeric predictor with two or more non-missing bins is kept."""

        df = self._make_df(
            [
                {
                    "context_partition": "p1",
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
                    "context_partition": "p1",
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
        result = aggregates._filter_single_bin_numeric_predictors(df).collect()
        assert result.shape[0] == 2, "Multi-bin numeric predictor should not be filtered"

    def test_symbolic_single_bin_not_filtered(self, aggregates):
        """A symbolic predictor with only one bin is NOT filtered (rule is numeric-only)."""

        df = self._make_df(
            [
                {
                    "context_partition": "p1",
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
        result = aggregates._filter_single_bin_numeric_predictors(df).collect()
        assert result.shape[0] == 1, "Single-bin symbolic predictor should be retained"

    def test_missing_bin_not_counted_for_single_bin_check(self, aggregates):
        """A MISSING bin does not count toward the bin count; a numeric predictor
        with only one real bin plus a MISSING bin should still be filtered."""

        df = self._make_df(
            [
                {
                    "context_partition": "p1",
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
                    "context_partition": "p1",
                    "predictor_name": "Score",
                    "predictor_type": "NUMERIC",
                    "bin_contents": MISSING,
                    "bin_order": 2,
                    "contribution": 0.1,
                    "contribution_abs": 0.1,
                    "contribution_min": 0.1,
                    "contribution_max": 0.1,
                    "frequency": 20,
                },
            ]
        )
        result = aggregates._filter_single_bin_numeric_predictors(df).collect()
        assert result.is_empty(), "Numeric predictor with only one real bin (plus MISSING) should be filtered"


def assert_predictor_rows_per_partition(df, top_n):
    """Assert each partition has exactly top_n + 1 rows (top predictors + remaining row).

    Used for `predictor_contributions` outputs where ``remaining=True`` (the default)
    appends a single aggregated 'remaining' row per partition.
    """
    expected_per_partition = top_n + 1
    counts = df.group_by("context_partition").agg(pl.len().alias("n")).to_dicts()
    assert counts, "Expected at least one partition in the result."
    for row in counts:
        assert row["n"] == expected_per_partition, (
            f"Partition {row['context_partition']!r} has {row['n']} rows, "
            f"expected {expected_per_partition} (top_n + 1 remaining)."
        )


def assert_symbolic_bins_per_predictor_capped(df, top_k):
    """Assert each symbolic predictor has at most ``top_k`` ordinary bins.

    The 'remaining' rollup and the force-included MISSING bin are appended after
    the top-k selection, so they are excluded from the count rather than folded
    into the limit.

    Numeric predictors are not capped by ``top_k``.
    """
    special = [REMAINING, MISSING]
    rows = (
        df.filter(~pl.col("bin_contents").is_in(special))
        .group_by(["predictor_name", "predictor_type"])
        .agg(pl.len().alias("n"))
        .to_dicts()
    )
    assert rows, "Expected at least one predictor in the result."
    for row in rows:
        if row["predictor_type"] == "SYMBOLIC":
            assert row["n"] <= top_k, (
                f"Symbolic predictor {row['predictor_name']!r} has {row['n']} ordinary bins, expected at most {top_k}."
            )


def test_create_context_batches_empty_list():
    """Test that empty context list returns empty batches dict."""
    batches = ContextOperations._create_context_batches([], 100)
    assert batches == {}


def test_create_context_batches_none():
    """Test that None context list returns empty batches dict."""
    batches = ContextOperations._create_context_batches(None, 100)
    assert isinstance(batches, dict)
    assert len(batches) == 0


def test_missing_flag_changes_predictor_contributions(aggregates):
    """``missing=False`` must exclude MISSING bins from predictor-level contributions.

    Regression test: a dropped ``pl.col(...)`` turned the exclusion filter into a
    constant-true predicate, so both flag values produced identical numbers.
    """
    with_missing = aggregates.predictor_contributions(missing=True)
    without_missing = aggregates.predictor_contributions(missing=False)

    age_with = with_missing.filter(pl.col("predictor_name") == "Age")["contribution"].item()
    age_without = without_missing.filter(pl.col("predictor_name") == "Age")["contribution"].item()
    assert age_with == pytest.approx(-0.011185, abs=1e-6)
    assert age_without == pytest.approx(-0.011055, abs=1e-6)


def test_missing_flag_changes_predictor_value_contributions(aggregates, predictors):
    """``missing`` must control whether MISSING bins appear at value level.

    Regression test: the value-level path had no ``missing`` filter at all, so
    MISSING bins were always returned regardless of the flag.
    """
    with_missing = aggregates.predictor_value_contributions(predictors=predictors, missing=True)
    without_missing = aggregates.predictor_value_contributions(predictors=predictors, missing=False)

    assert with_missing.height == 19
    assert with_missing.filter(pl.col("bin_contents") == MISSING).height == 2
    assert without_missing.height == 17
    assert without_missing.filter(pl.col("bin_contents") == MISSING).height == 0


@pytest.mark.parametrize("top_n", [0, -1, True])
def test_predictor_contributions_rejects_invalid_top_n(aggregates, top_n):
    """``top_n`` must be a positive integer.

    Regression test: the check used truthiness, so ``0`` slipped through while the
    perfectly valid ``1`` was rejected.
    """
    with pytest.raises(ValueError):
        aggregates.predictor_contributions(top_n=top_n)


def test_predictor_contributions_accepts_top_n_of_one(aggregates):
    """``top_n=1`` is valid and returns exactly one predictor plus the rollup."""
    df = aggregates.predictor_contributions(top_n=1)
    assert df["predictor_name"].to_list() == ["pyName", REMAINING]


class TestSchema:
    """The aggregated parquet files are narrowed and cast on read."""

    def test_scan_applies_expected_dtypes(self, aggregates):
        schema = aggregates.explanations.overall.collect_schema()
        assert dict(schema) == AGGREGATE_SCHEMA

    def test_missing_column_raises(self, tmp_path):
        from pdstools.explanations.Schema import apply_schema

        lf = pl.LazyFrame({"context_partition": ["0"], "predictor_name": ["Age"]})
        with pytest.raises(ValueError, match="missing expected column"):
            apply_schema(lf)
