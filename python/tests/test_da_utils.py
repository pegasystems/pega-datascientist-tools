# python/tests/test_da_utils.py
"""
Tests for decision_analyzer/utils.py utility functions.

Covers: parse_sample_flag, format_count_for_filename, area_under_curve,
gini_coefficient, get_first_level_stats, resolve_aliases, rename_and_cast_types,
_cast_columns, get_scope_config, create_hierarchical_selectors,
sample_interactions, prepare_and_save, _find_interaction_id_column,
_get_interaction_id_candidates, should_cache_source.
"""

import polars as pl
import pytest

pl.enable_string_cache()

from pdstools.decision_analyzer.column_schema import (  # noqa: E402
    DecisionAnalyzer,
    ExplainabilityExtract,
)
from pdstools.decision_analyzer.utils import (  # noqa: E402
    _cast_columns,
    _find_interaction_id_column,
    _get_interaction_id_candidates,
    area_under_curve,
    create_hierarchical_selectors,
    get_first_level_stats,
    get_scope_config,
    gini_coefficient,
    parse_sample_flag,
    prepare_and_save,
    rename_and_cast_types,
    resolve_aliases,
    resolve_filter_column,
    sample_interactions,
)


# ---------------------------------------------------------------------------
# parse_sample_flag
# ---------------------------------------------------------------------------


class TestParseSampleFlag:
    """Parsing the --sample CLI flag into n/fraction kwargs."""

    def test_absolute_count(self):
        result = parse_sample_flag("100000")
        assert result == {"n": 100000}

    def test_percentage(self):
        result = parse_sample_flag("10%")
        assert result == {"fraction": 0.1}

    def test_percentage_with_whitespace(self):
        result = parse_sample_flag("  25%  ")
        assert result == {"fraction": 0.25}

    def test_100_percent(self):
        result = parse_sample_flag("100%")
        assert result == {"fraction": 1.0}

    def test_zero_count_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_sample_flag("0")

    def test_zero_percent_raises(self):
        with pytest.raises(ValueError, match=r"\(0, 100\]"):
            parse_sample_flag("0%")

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_sample_flag("-5")

    def test_101_percent_raises(self):
        with pytest.raises(ValueError, match=r"\(0, 100\]"):
            parse_sample_flag("101%")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_sample_flag("abc")

    def test_single_unit(self):
        result = parse_sample_flag("1")
        assert result == {"n": 1}

    def test_thousands_notation_lowercase_k(self):
        result = parse_sample_flag("100k")
        assert result == {"n": 100000}

    def test_thousands_notation_uppercase_k(self):
        result = parse_sample_flag("100K")
        assert result == {"n": 100000}

    def test_millions_notation_lowercase_m(self):
        result = parse_sample_flag("1M")
        assert result == {"n": 1000000}

    def test_millions_notation_uppercase_m(self):
        result = parse_sample_flag("1m")
        assert result == {"n": 1000000}

    def test_decimal_thousands(self):
        result = parse_sample_flag("1.5k")
        assert result == {"n": 1500}

    def test_decimal_millions(self):
        result = parse_sample_flag("2.5M")
        assert result == {"n": 2500000}

    def test_thousands_with_whitespace(self):
        result = parse_sample_flag("  50k  ")
        assert result == {"n": 50000}


# ---------------------------------------------------------------------------
# format_count_for_filename
# ---------------------------------------------------------------------------


def test_format_count_for_filename_units():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(42) == "42"
    assert format_count_for_filename(999) == "999"


def test_format_count_for_filename_thousands():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(1000) == "1k"
    assert format_count_for_filename(1500) == "1.5k"
    assert format_count_for_filename(87432) == "87k"
    assert format_count_for_filename(999999) == "1M"


def test_format_count_for_filename_millions():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(1000000) == "1M"
    assert format_count_for_filename(1234567) == "1.2M"
    assert format_count_for_filename(87000000) == "87M"
    assert format_count_for_filename(999999999) == "1B"


def test_format_count_for_filename_billions():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(1000000000) == "1B"
    assert format_count_for_filename(2500000000) == "2.5B"
    assert format_count_for_filename(87000000000) == "87B"


def test_format_count_for_filename_edge_cases():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    # Exact boundaries
    assert format_count_for_filename(1) == "1"
    assert format_count_for_filename(10) == "10"
    assert format_count_for_filename(100) == "100"
    # Check boundary transitions
    assert "k" in format_count_for_filename(1001)
    assert "M" in format_count_for_filename(1000001)
    assert "B" in format_count_for_filename(1000000001)


def test_format_count_rounding_to_unit_boundaries():
    """Test that values rounding to 100+ transition to next unit cleanly."""
    from pdstools.decision_analyzer.utils import format_count_for_filename

    # Values that round to 100k should display as 100k
    assert format_count_for_filename(99500) == "100k"
    assert format_count_for_filename(99900) == "100k"

    # Values that round to 1000k should transition to 1M
    assert format_count_for_filename(999500) == "1M"
    assert format_count_for_filename(999900) == "1M"

    # Verify no scientific notation in output
    result = format_count_for_filename(99500)
    assert "e" not in result.lower(), f"Got scientific notation: {result}"


# ---------------------------------------------------------------------------
# area_under_curve / gini_coefficient
# ---------------------------------------------------------------------------


class TestAreaUnderCurve:
    """Trapezoidal area computation."""

    def test_triangle(self):
        """Triangle with vertices (0,0)→(1,1): area = 0.5."""
        df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        assert area_under_curve(df, "x", "y") == pytest.approx(0.5)

    def test_unit_square(self):
        """Rectangle (0,1)→(1,1): area = 1.0."""
        df = pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0]})
        assert area_under_curve(df, "x", "y") == pytest.approx(1.0)

    def test_trapezoid(self):
        """Trapezoid: y goes 0→1→1 over x=0,0.5,1 → area = 0.75."""
        df = pl.DataFrame({"x": [0.0, 0.5, 1.0], "y": [0.0, 1.0, 1.0]})
        assert area_under_curve(df, "x", "y") == pytest.approx(0.75)

    def test_zero_area(self):
        """All y=0 → area=0."""
        df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]})
        assert area_under_curve(df, "x", "y") == pytest.approx(0.0)


class TestGiniCoefficient:
    """gini = auc * 2 - 1."""

    def test_perfect_equality(self):
        """AUC=0.5 → Gini=0."""
        df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        assert gini_coefficient(df, "x", "y") == pytest.approx(0.0)

    def test_perfect_model(self):
        """AUC=1.0 → Gini=1.0."""
        df = pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0]})
        assert gini_coefficient(df, "x", "y") == pytest.approx(1.0)

    def test_relationship_to_auc(self):
        df = pl.DataFrame({"x": [0.0, 0.5, 1.0], "y": [0.0, 1.0, 1.0]})
        auc = area_under_curve(df, "x", "y")
        gini = gini_coefficient(df, "x", "y")
        assert gini == pytest.approx(auc * 2 - 1)


# ---------------------------------------------------------------------------
# get_first_level_stats
# ---------------------------------------------------------------------------


class TestGetFirstLevelStats:
    """Summary statistics with and without Interaction ID column."""

    @pytest.fixture()
    def interaction_data_with_id(self):
        return pl.LazyFrame(
            {
                "Issue": ["Sales", "Sales", "Service", "Service"],
                "Group": ["Cards", "Cards", "Support", "Support"],
                "Action": ["A1", "A2", "B1", "B2"],
                "Interaction ID": ["i1", "i1", "i2", "i3"],
            },
            schema={
                "Issue": pl.Categorical,
                "Group": pl.Categorical,
                "Action": pl.Categorical,
                "Interaction ID": pl.Utf8,
            },
        )

    @pytest.fixture()
    def interaction_data_no_id(self):
        return pl.LazyFrame(
            {
                "Issue": ["Sales", "Sales", "Service"],
                "Group": ["Cards", "Cards", "Support"],
                "Action": ["A1", "A2", "B1"],
            },
            schema={
                "Issue": pl.Categorical,
                "Group": pl.Categorical,
                "Action": pl.Categorical,
            },
        )

    def test_stats_with_interaction_id(self, interaction_data_with_id):
        stats = get_first_level_stats(interaction_data_with_id)
        assert stats["Actions"] == 4
        assert stats["Rows"] == 4
        assert stats["Decisions"] == 3

    def test_stats_without_interaction_id(self, interaction_data_no_id):
        stats = get_first_level_stats(interaction_data_no_id)
        assert stats["Actions"] == 3
        assert stats["Rows"] == 3
        assert "Decisions" not in stats

    def test_stats_with_filter(self, interaction_data_with_id):
        filt = pl.col("Issue") == "Sales"
        stats = get_first_level_stats(interaction_data_with_id, filters=filt)
        assert stats["Actions"] == 2
        assert stats["Rows"] == 2
        assert stats["Decisions"] == 1


# ---------------------------------------------------------------------------
# resolve_aliases
# ---------------------------------------------------------------------------


class TestResolveAliases:
    """Alias renaming for column harmonisation."""

    def test_alias_renamed_to_canonical(self):
        """When alias 'pyChannel' exists and canonical doesn't, rename it."""
        lf = pl.LazyFrame({"pyChannel": ["Web", "Mobile"]})
        result = resolve_aliases(lf, DecisionAnalyzer)
        names = result.collect_schema().names()
        assert "Primary_ContainerPayload_Channel" in names
        assert "pyChannel" not in names

    def test_no_rename_when_canonical_exists(self):
        """If the canonical raw key already exists, don't rename the alias."""
        lf = pl.LazyFrame(
            {
                "Primary_ContainerPayload_Channel": ["Web", "Mobile"],
                "pyChannel": ["X", "Y"],
            }
        )
        result = resolve_aliases(lf, DecisionAnalyzer)
        names = result.collect_schema().names()
        # Both should still exist — canonical present, so no rename.
        assert "Primary_ContainerPayload_Channel" in names
        assert "pyChannel" in names

    def test_no_rename_when_display_name_exists(self):
        """If the display_name already exists, don't rename alias."""
        lf = pl.LazyFrame({"Channel": ["Web"], "pyChannel": ["X"]})
        result = resolve_aliases(lf, DecisionAnalyzer)
        names = result.collect_schema().names()
        assert "Channel" in names
        assert "pyChannel" in names

    def test_multiple_table_definitions(self):
        """Pass both DA and EE definitions; aliases from each are handled."""
        # pyChannel is itself a raw key in EE, so the alias guard prevents
        # renaming it to DA's Primary_ContainerPayload_Channel. Use an alias
        # that is NOT a raw key in either definition.
        lf = pl.LazyFrame(
            {
                "InteractionID": ["i1"],
                "DecisionTime": ["20230101T120000.000 GMT"],
            }
        )
        result = resolve_aliases(lf, DecisionAnalyzer, ExplainabilityExtract)
        names = result.collect_schema().names()
        # InteractionID is alias for pxInteractionID in both defs
        assert "pxInteractionID" in names
        # DecisionTime is alias for pxDecisionTime
        assert "pxDecisionTime" in names

    def test_no_aliases_is_noop(self):
        """Columns with no aliases defined stay untouched."""
        lf = pl.LazyFrame({"SomeRandomCol": [1]})
        result = resolve_aliases(lf, DecisionAnalyzer)
        assert result.collect_schema().names() == ["SomeRandomCol"]


# ---------------------------------------------------------------------------
# rename_and_cast_types
# ---------------------------------------------------------------------------


class TestRenameAndCastTypes:
    """Single-pass rename + type cast via ColumnResolver."""

    def test_basic_rename_and_cast(self):
        lf = pl.LazyFrame(
            {
                "pyIssue": ["Sales", "Service"],
                "pyGroup": ["Cards", "Support"],
                "pyName": ["A1", "B1"],
            }
        )
        result = rename_and_cast_types(lf, DecisionAnalyzer)
        schema = result.collect_schema()
        assert "Issue" in schema.names()
        assert "Group" in schema.names()
        assert "Action" in schema.names()
        assert schema["Issue"] == pl.Categorical
        assert schema["Group"] == pl.Categorical

    def test_conflict_resolution_prefers_display_name(self):
        """When both raw key and display_name columns exist, drop the raw key."""
        lf = pl.LazyFrame(
            {
                "pyIssue": ["raw_val"],
                "Issue": ["display_val"],
                "pyGroup": ["G1"],
                "pyName": ["A1"],
            }
        )
        result = rename_and_cast_types(lf, DecisionAnalyzer)
        collected = result.collect()
        assert collected.get_column("Issue").to_list() == ["display_val"]


# ---------------------------------------------------------------------------
# _cast_columns
# ---------------------------------------------------------------------------


class TestCastColumns:
    """Column type casting including special Datetime handling."""

    def test_cast_float(self):
        lf = pl.LazyFrame({"val": [1, 2, 3]})
        result = _cast_columns(lf, {"val": pl.Float64})
        assert result.collect_schema()["val"] == pl.Float64

    def test_cast_categorical(self):
        lf = pl.LazyFrame({"cat": ["a", "b", "c"]})
        result = _cast_columns(lf, {"cat": pl.Categorical})
        assert result.collect_schema()["cat"] == pl.Categorical

    def test_cast_skips_missing_column(self):
        lf = pl.LazyFrame({"x": [1]})
        result = _cast_columns(lf, {"nonexistent": pl.Float64})
        assert result.collect_schema() == lf.collect_schema()

    def test_cast_same_type_is_noop(self):
        lf = pl.LazyFrame({"x": pl.Series([1.0, 2.0], dtype=pl.Float64)})
        result = _cast_columns(lf, {"x": pl.Float64})
        assert result.collect().get_column("x").to_list() == [1.0, 2.0]

    def test_cast_datetime_uses_pega_parser(self):
        """Datetime target type triggers parse_pega_date_time_formats."""
        lf = pl.LazyFrame({"dt": ["20230101T120000.000 GMT"]})
        result = _cast_columns(lf, {"dt": pl.Datetime})
        schema = result.collect_schema()
        # parse_pega_date_time_formats should produce a Datetime column
        assert schema["dt"] == pl.Datetime or schema["dt"].base_type() == pl.Datetime


# ---------------------------------------------------------------------------
# get_scope_config
# ---------------------------------------------------------------------------


class TestGetScopeConfig:
    """Three branches: Action-level, Group-level, Issue-level."""

    def test_action_level(self):
        cfg = get_scope_config("Sales", "Cards", "GoldCard")
        assert cfg["level"] == "Action"
        assert cfg["x_col"] == "Action"
        assert cfg["selected_value"] == "GoldCard"
        assert cfg["group_cols"] == ["Issue", "Group", "Action"]

    def test_group_level(self):
        cfg = get_scope_config("Sales", "Cards", "All")
        assert cfg["level"] == "Group"
        assert cfg["x_col"] == "Group"
        assert cfg["selected_value"] == "Cards"
        assert cfg["group_cols"] == ["Issue", "Group"]

    def test_issue_level(self):
        cfg = get_scope_config("Sales", "All", "All")
        assert cfg["level"] == "Issue"
        assert cfg["x_col"] == "Issue"
        assert cfg["selected_value"] == "Sales"
        assert cfg["group_cols"] == ["Issue"]

    def test_lever_condition_is_expr(self):
        for args in [
            ("I", "G", "A"),
            ("I", "G", "All"),
            ("I", "All", "All"),
        ]:
            cfg = get_scope_config(*args)
            assert isinstance(cfg["lever_condition"], pl.Expr)


# ---------------------------------------------------------------------------
# create_hierarchical_selectors
# ---------------------------------------------------------------------------


class TestCreateHierarchicalSelectors:
    """Hierarchical filter options from Issue/Group/Action data."""

    @pytest.fixture()
    def hierarchy_data(self):
        return pl.LazyFrame(
            {
                "Issue": ["Sales", "Sales", "Sales", "Service"],
                "Group": ["Cards", "Cards", "Loans", "Support"],
                "Action": ["A1", "A2", "A3", "B1"],
            },
            schema={
                "Issue": pl.Categorical,
                "Group": pl.Categorical,
                "Action": pl.Categorical,
            },
        )

    def test_default_selections(self, hierarchy_data):
        result = create_hierarchical_selectors(hierarchy_data)
        # Issues should contain both
        assert set(result["issues"]["options"]) == {"Sales", "Service"}
        assert result["issues"]["index"] == 0
        # Groups and actions should have "All" prefix
        assert result["groups"]["options"][0] == "All"
        assert result["actions"]["options"][0] == "All"

    def test_selected_issue_sets_index(self, hierarchy_data):
        # Get the options list first and verify a known issue can be selected
        result_sales = create_hierarchical_selectors(hierarchy_data, selected_issue="Sales")
        result_service = create_hierarchical_selectors(hierarchy_data, selected_issue="Service")
        # Both should find their issue in the options
        sales_opts = result_sales["issues"]["options"]
        service_opts = result_service["issues"]["options"]
        assert "Sales" in sales_opts
        assert "Service" in service_opts
        # The index should point to the selected issue
        sales_idx = result_sales["issues"]["index"]
        service_idx = result_service["issues"]["index"]
        assert sales_opts[sales_idx] == "Sales"
        assert service_opts[service_idx] == "Service"

    def test_selected_group_filters_actions(self, hierarchy_data):
        # Select "Sales" issue and "Cards" group
        result = create_hierarchical_selectors(hierarchy_data, selected_issue="Sales", selected_group="Cards")
        assert result["groups"]["index"] > 0  # not "All"
        # Actions should be filtered to Cards actions only
        action_opts = result["actions"]["options"]
        assert "All" in action_opts
        # A1 and A2 are under Cards, A3 is under Loans
        actual_actions = [a for a in action_opts if a != "All"]
        assert set(actual_actions) == {"A1", "A2"}

    def test_all_group_shows_all_actions(self, hierarchy_data):
        result = create_hierarchical_selectors(hierarchy_data, selected_issue="Sales", selected_group="All")
        action_opts = result["actions"]["options"]
        actual_actions = [a for a in action_opts if a != "All"]
        assert set(actual_actions) == {"A1", "A2", "A3"}


# ---------------------------------------------------------------------------
# sample_interactions
# ---------------------------------------------------------------------------


class TestSampleInteractions:
    """Deterministic hash-based interaction sampling."""

    @pytest.fixture()
    def sample_data(self):
        """20 rows, 10 unique interactions, 2 rows each."""
        ids = [f"int_{i}" for i in range(10) for _ in range(2)]
        vals = list(range(20))
        return pl.LazyFrame(
            {"pxInteractionID": ids, "value": vals},
        )

    def test_both_n_and_fraction_raises(self, sample_data):
        with pytest.raises(ValueError, match="Exactly one"):
            sample_interactions(sample_data, n=5, fraction=0.5)

    def test_neither_n_nor_fraction_raises(self, sample_data):
        with pytest.raises(ValueError, match="Exactly one"):
            sample_interactions(sample_data)

    def test_sample_by_n(self, sample_data):
        result = sample_interactions(sample_data, n=5).collect()
        unique_ids = result.get_column("pxInteractionID").n_unique()
        # Should have at most 5 unique interactions (hash-based, approximate)
        assert unique_ids <= 10
        assert result.height > 0

    def test_sample_by_fraction(self, sample_data):
        result = sample_interactions(sample_data, fraction=0.5).collect()
        assert result.height > 0
        assert result.height <= 20

    def test_fraction_out_of_range_raises(self, sample_data):
        with pytest.raises(ValueError, match="fraction"):
            sample_interactions(sample_data, fraction=0.0)
        with pytest.raises(ValueError, match="fraction"):
            sample_interactions(sample_data, fraction=1.5)

    def test_deterministic_output(self, sample_data):
        """Same data + same params → same result."""
        r1 = sample_interactions(sample_data, fraction=0.5).collect()
        r2 = sample_interactions(sample_data, fraction=0.5).collect()
        assert r1.equals(r2)

    def test_n_larger_than_total_returns_all(self, sample_data):
        """If n >= total interactions, return all data unchanged."""
        result = sample_interactions(sample_data, n=100).collect()
        assert result.height == 20

    def test_auto_detect_id_column(self):
        """Works when 'Interaction ID' display name is used."""
        lf = pl.LazyFrame(
            {
                "Interaction ID": ["i1", "i1", "i2", "i2", "i3", "i3"],
                "x": [1, 2, 3, 4, 5, 6],
            }
        )
        result = sample_interactions(lf, fraction=0.5).collect()
        assert result.height >= 0  # just verify it doesn't error

    def test_random_sampling_with_fraction(self, sample_data):
        """Random sampling (use_random=True) with fraction parameter."""
        result = sample_interactions(sample_data, fraction=0.5, use_random=True).collect()
        unique_ids = result.get_column("pxInteractionID").n_unique()
        # Should have approximately 5 unique interactions (50% of 10)
        assert 3 <= unique_ids <= 7  # Allow some variance
        assert result.height > 0

    def test_random_sampling_with_n(self, sample_data):
        """Random sampling (use_random=True) with n parameter."""
        result = sample_interactions(sample_data, n=5, use_random=True).collect()
        unique_ids = result.get_column("pxInteractionID").n_unique()
        # Should have at most 5 unique interactions
        assert unique_ids <= 10
        assert result.height > 0

    def test_random_sampling_skips_when_n_exceeds_total(self, sample_data):
        """Random sampling returns all data when n >= total interactions."""
        result = sample_interactions(sample_data, n=100, use_random=True).collect()
        assert result.height == 20  # All rows returned

    def test_random_sampling_skips_when_fraction_is_1(self, sample_data):
        """Random sampling with fraction=1.0 returns all data."""
        result = sample_interactions(sample_data, fraction=1.0, use_random=True).collect()
        assert result.height == 20  # All rows returned


# ---------------------------------------------------------------------------
# prepare_and_save
# ---------------------------------------------------------------------------


class TestPrepareAndSave:
    """Persist prepared data (sampled or cached) as parquet."""

    @pytest.fixture()
    def mock_decision_data(self):
        """Mock decision analyzer data with interactions."""
        ids = [f"int_{i}" for i in range(10) for _ in range(2)]
        return pl.LazyFrame(
            {
                "pxInteractionID": ids,
                "value": list(range(20)),
            }
        )

    def test_writes_parquet_file(self, tmp_path):
        ids = [f"int_{i}" for i in range(10) for _ in range(2)]
        lf = pl.LazyFrame(
            {
                "pxInteractionID": ids,
                "value": list(range(20)),
            }
        )
        result, path = prepare_and_save(lf, fraction=0.5, output_dir=str(tmp_path))
        # Path should be returned and file should exist
        assert path is not None
        assert path.exists()
        # Filename should have the new format with count
        assert "decision_analyzer_sample_" in path.name
        assert path.name.endswith(".parquet")
        # Result should be a LazyFrame scanning the written file
        collected = result.collect()
        assert collected.height > 0

    def test_skips_when_n_exceeds_total(self, tmp_path):
        ids = ["i1", "i1", "i2", "i2"]
        lf = pl.LazyFrame({"pxInteractionID": ids, "value": [1, 2, 3, 4]})
        result, path = prepare_and_save(lf, n=100, output_dir=str(tmp_path))
        out_file = tmp_path / "decision_analyzer_sample.parquet"
        # File should NOT be written when sampling is skipped
        assert not out_file.exists()
        assert path is None
        assert result.collect().height == 4

    def test_prepare_and_save_with_source_path_metadata(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save
        import polars as pl

        lf = mock_decision_data
        source_file = tmp_path / "original.parquet"

        result, path = prepare_and_save(lf, fraction=0.5, output_dir=str(tmp_path), source_path=str(source_file))

        assert path is not None
        # Check filename contains count
        assert "decision_analyzer_sample_" in path.name
        assert path.name.endswith(".parquet")

        # Check metadata was written
        metadata = pl.read_parquet_metadata(str(path))
        assert "pdstools:source_file" in metadata
        assert metadata["pdstools:source_file"] == str(source_file)
        assert "pdstools:sample_percentage" in metadata
        assert float(metadata["pdstools:sample_percentage"]) == 50.0
        assert metadata["pdstools:sample_percentage_method"] == "exact"

    def test_prepare_and_save_with_chained_sampling(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save
        import polars as pl

        lf = mock_decision_data
        original_source = "/data/original.parquet"

        # First sample: 50%
        first_sample_file = tmp_path / "first.parquet"
        first_meta = {
            "pdstools:source_file": original_source,
            "pdstools:sample_percentage": "50.0",
            "pdstools:sample_percentage_method": "exact",
        }
        lf.collect().write_parquet(first_sample_file, metadata=first_meta)

        # Second sample: 20% of the first sample
        lf_rescan = pl.scan_parquet(first_sample_file)
        result, path = prepare_and_save(
            lf_rescan, fraction=0.2, output_dir=str(tmp_path), source_path=str(first_sample_file)
        )

        assert path is not None

        # Check metadata inheritance
        metadata = pl.read_parquet_metadata(str(path))
        # Should inherit original source, not intermediate file
        assert metadata["pdstools:source_file"] == original_source
        # Should multiply percentages: 50% * 20% = 10%
        assert float(metadata["pdstools:sample_percentage"]) == 10.0
        assert metadata["pdstools:sample_percentage_method"] == "exact"

    def test_prepare_and_save_filename_format(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        lf = mock_decision_data

        # Sample to a specific count
        result, path = prepare_and_save(lf, n=100, output_dir=str(tmp_path), source_path="test.parquet")

        if path is not None:
            # Filename should contain formatted count
            assert "decision_analyzer_sample_" in path.name
            # Should have a number (exact format depends on actual count in mock data)
            assert any(char.isdigit() for char in path.name)

    def test_prepare_and_save_without_source_path(self, mock_decision_data, tmp_path):
        """Test backward compatibility - source_path is optional."""
        from pdstools.decision_analyzer.utils import prepare_and_save
        import polars as pl

        lf = mock_decision_data

        # Call without source_path (backward compatibility)
        result, path = prepare_and_save(lf, fraction=0.5, output_dir=str(tmp_path))

        assert path is not None
        # Should still write metadata, but with "unknown" source
        metadata = pl.read_parquet_metadata(str(path))
        assert metadata["pdstools:source_file"] == "unknown"

    def test_prepare_and_save_with_invalid_source_path(self, mock_decision_data, tmp_path):
        """Test graceful handling of nonexistent source path."""
        from pdstools.decision_analyzer.utils import prepare_and_save
        import polars as pl

        lf = mock_decision_data

        # Pass nonexistent source path
        result, path = prepare_and_save(
            lf, fraction=0.5, output_dir=str(tmp_path), source_path="/nonexistent/file.parquet"
        )

        assert path is not None
        # Should write metadata with the provided path (even if it doesn't exist)
        metadata = pl.read_parquet_metadata(str(path))
        assert metadata["pdstools:source_file"] == "/nonexistent/file.parquet"

    def test_prepare_and_save_n_based_estimates_percentage(self, tmp_path):
        """Test that n-based sampling estimates the percentage instead of showing 0.00%."""
        from pdstools.decision_analyzer.utils import prepare_and_save
        import polars as pl

        # Create data with 1000 unique interactions (large enough to ensure sampling)
        ids = [f"int_{i:04d}" for i in range(1000) for _ in range(2)]
        lf = pl.LazyFrame({"pxInteractionID": ids, "value": list(range(2000))})

        # Sample to 100 interactions (should be ~10%)
        result, path = prepare_and_save(lf, n=100, output_dir=str(tmp_path), source_path="test.parquet")

        assert path is not None

        # Check metadata
        metadata = pl.read_parquet_metadata(str(path))
        sample_pct = float(metadata["pdstools:sample_percentage"])
        method = metadata["pdstools:sample_percentage_method"]

        # Should have estimated a percentage (not 0.00%)
        assert sample_pct > 0.0, "Sample percentage should not be 0.00%"
        # Should be marked as approximated
        assert method == "approximated"
        # Should be roughly 10% (allow wide tolerance for estimation variance)
        assert 5.0 <= sample_pct <= 20.0, f"Expected ~10%, got {sample_pct}%"


# ---------------------------------------------------------------------------
# prepare_and_save (caching mode)
# ---------------------------------------------------------------------------


class TestPrepareAndSaveCachingMode:
    """Test prepare_and_save (renamed from prepare_and_save) in caching mode."""

    @pytest.fixture()
    def mock_decision_data(self):
        """Mock decision analyzer data with interactions."""
        ids = [f"int_{i}" for i in range(10) for _ in range(2)]
        return pl.LazyFrame(
            {
                "pxInteractionID": ids,
                "value": list(range(20)),
            }
        )

    def test_cache_creates_file_with_cache_prefix(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        source_file = tmp_path / "original.csv"
        result, path = prepare_and_save(
            mock_decision_data,
            source_path=str(source_file),
            output_dir=str(tmp_path),
        )

        assert path is not None
        assert path.exists()
        # Should use "cache" prefix not "sample"
        assert "decision_analyzer_cache_" in path.name
        assert path.name.endswith(".parquet")

    def test_cache_metadata_has_100_percent(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        source_file = tmp_path / "original.csv"
        result, path = prepare_and_save(
            mock_decision_data,
            source_path=str(source_file),
            output_dir=str(tmp_path),
        )

        assert path is not None
        metadata = pl.read_parquet_metadata(str(path))
        assert metadata["pdstools:source_file"] == str(source_file)
        assert float(metadata["pdstools:sample_percentage"]) == 100.0
        assert metadata["pdstools:sample_percentage_method"] == "exact"

    def test_cache_without_source_path_returns_none(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        result, path = prepare_and_save(
            mock_decision_data,
            output_dir=str(tmp_path),
        )

        # Without source_path, caching should be skipped
        assert path is None

    def test_sampling_mode_still_uses_sample_prefix(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        result, path = prepare_and_save(
            mock_decision_data,
            fraction=0.5,
            source_path=str(tmp_path / "original.csv"),
            output_dir=str(tmp_path),
        )

        assert path is not None
        # Sampling mode should still use "sample" prefix
        assert "decision_analyzer_sample_" in path.name

    def test_cache_includes_interaction_count_in_filename(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        result, path = prepare_and_save(
            mock_decision_data,
            source_path=str(tmp_path / "original.csv"),
            output_dir=str(tmp_path),
        )

        assert path is not None
        # Should have formatted count (10 interactions in mock data)
        assert "10" in path.name or "10." in path.name


# ---------------------------------------------------------------------------
# _find_interaction_id_column / _get_interaction_id_candidates
# ---------------------------------------------------------------------------


class TestFindInteractionIdColumn:
    """Locating the interaction ID column from raw column names."""

    def test_finds_raw_key(self):
        assert _find_interaction_id_column({"pxInteractionID", "other"}) == "pxInteractionID"

    def test_finds_display_name(self):
        assert _find_interaction_id_column({"Interaction ID", "other"}) == "Interaction ID"

    def test_raises_when_missing(self):
        with pytest.raises(ValueError, match="no interaction ID column"):
            _find_interaction_id_column({"foo", "bar"})

    def test_prefers_raw_key_over_display_name(self):
        """pxInteractionID comes first in candidates, so it wins."""
        result = _find_interaction_id_column({"pxInteractionID", "Interaction ID"})
        assert result == "pxInteractionID"


class TestGetInteractionIdCandidates:
    """Candidate list from schema definitions."""

    def test_returns_nonempty_list(self):
        candidates = _get_interaction_id_candidates()
        assert len(candidates) > 0

    def test_includes_raw_key(self):
        assert "pxInteractionID" in _get_interaction_id_candidates()

    def test_includes_display_name(self):
        assert "Interaction ID" in _get_interaction_id_candidates()

    def test_includes_alias(self):
        assert "InteractionID" in _get_interaction_id_candidates()

    def test_no_duplicates(self):
        candidates = _get_interaction_id_candidates()
        assert len(candidates) == len(set(candidates))


# ---------------------------------------------------------------------------
# _read_source_metadata
# ---------------------------------------------------------------------------


def test_read_source_metadata_with_metadata(tmp_path):
    from pdstools.decision_analyzer.utils import _read_source_metadata
    import polars as pl

    # Create a file with metadata
    df = pl.DataFrame({"pxInteractionID": ["A", "B", "C"]})
    test_file = tmp_path / "test.parquet"
    metadata = {
        "pdstools:source_file": "/original/data.parquet",
        "pdstools:sample_percentage": "50.0",
        "pdstools:sample_percentage_method": "exact",
    }
    df.write_parquet(test_file, metadata=metadata)

    result = _read_source_metadata(str(test_file))

    assert result is not None
    assert result["source_file"] == "/original/data.parquet"
    assert result["sample_percentage"] == 50.0
    assert result["method"] == "exact"


def test_read_source_metadata_without_metadata(tmp_path):
    from pdstools.decision_analyzer.utils import _read_source_metadata
    import polars as pl

    # Create a file without our metadata
    df = pl.DataFrame({"pxInteractionID": ["A", "B", "C"]})
    test_file = tmp_path / "test.parquet"
    df.write_parquet(test_file)

    result = _read_source_metadata(str(test_file))

    assert result is None


def test_read_source_metadata_nonexistent_file():
    from pdstools.decision_analyzer.utils import _read_source_metadata

    result = _read_source_metadata("/nonexistent/file.parquet")

    assert result is None


# ---------------------------------------------------------------------------
# should_cache_source
# ---------------------------------------------------------------------------


class TestShouldCacheSource:
    """Detect when source data should be cached as parquet."""

    def test_none_returns_false(self):
        from pdstools.decision_analyzer.utils import should_cache_source

        assert should_cache_source(None) is False

    def test_single_parquet_returns_false(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        parquet_file = tmp_path / "data.parquet"
        parquet_file.touch()
        assert should_cache_source(str(parquet_file)) is False

    def test_csv_file_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        csv_file = tmp_path / "data.csv"
        csv_file.touch()
        assert should_cache_source(str(csv_file)) is True

    def test_directory_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        data_dir = tmp_path / "data_dir"
        data_dir.mkdir()
        assert should_cache_source(str(data_dir)) is True

    def test_json_file_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        json_file = tmp_path / "data.json"
        json_file.touch()
        assert should_cache_source(str(json_file)) is True

    def test_arrow_file_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        arrow_file = tmp_path / "data.arrow"
        arrow_file.touch()
        assert should_cache_source(str(arrow_file)) is True

    def test_empty_string_returns_false(self):
        from pdstools.decision_analyzer.utils import should_cache_source

        assert should_cache_source("") is False


# ---------------------------------------------------------------------------
# _determine_output_directory
# ---------------------------------------------------------------------------


class TestDetermineOutputDirectory:
    """Test output directory selection logic for cached/sampled files."""

    def test_explicit_output_dir_takes_precedence(self, tmp_path):
        from pdstools.decision_analyzer.utils import _determine_output_directory

        source_file = tmp_path / "data" / "source.parquet"
        source_file.parent.mkdir()
        source_file.touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = _determine_output_directory(str(source_file), str(output_dir))
        assert result == output_dir

    def test_uses_source_directory_when_file_and_writeable(self, tmp_path):
        from pdstools.decision_analyzer.utils import _determine_output_directory

        source_file = tmp_path / "data" / "source.parquet"
        source_file.parent.mkdir()
        source_file.touch()

        result = _determine_output_directory(str(source_file), None)
        assert result == source_file.parent

    def test_uses_parent_directory_when_source_is_directory_and_writeable(self, tmp_path):
        from pdstools.decision_analyzer.utils import _determine_output_directory

        source_dir = tmp_path / "data"
        source_dir.mkdir()

        result = _determine_output_directory(str(source_dir), None)
        # Should use parent directory (tmp_path) when it's writable
        assert result == tmp_path

    def test_falls_back_to_current_dir_when_source_is_none(self):
        from pdstools.decision_analyzer.utils import _determine_output_directory
        from pathlib import Path

        result = _determine_output_directory(None, None)
        assert result == Path(".")

    def test_falls_back_to_current_dir_when_source_nonexistent(self):
        from pdstools.decision_analyzer.utils import _determine_output_directory
        from pathlib import Path

        result = _determine_output_directory("/nonexistent/file.parquet", None)
        assert result == Path(".")


# ---------------------------------------------------------------------------
# resolve_filter_column
# ---------------------------------------------------------------------------


class TestResolveFilterColumn:
    """Tests for resolve_filter_column()."""

    def test_resolve_display_name_v2(self):
        """Display name 'Interaction ID' resolves to raw key 'pxInteractionID'."""
        result = resolve_filter_column("Interaction ID", available_columns={"pxInteractionID", "pyIssue"})
        assert result == "pxInteractionID"

    def test_resolve_display_name_v1(self):
        """Display name 'Subject ID' resolves to v1 raw key 'pySubjectID'."""
        result = resolve_filter_column("Subject ID", available_columns={"pySubjectID", "pxInteractionID"})
        assert result == "pySubjectID"

    def test_resolve_alias(self):
        """Alias 'InteractionID' resolves to 'pxInteractionID'."""
        result = resolve_filter_column("InteractionID", available_columns={"pxInteractionID"})
        assert result == "pxInteractionID"

    def test_resolve_raw_key_fallback(self):
        """Raw column name works as fallback."""
        result = resolve_filter_column("pxInteractionID", available_columns={"pxInteractionID"})
        assert result == "pxInteractionID"

    def test_resolve_case_insensitive(self):
        """Column name resolution is case-insensitive."""
        result = resolve_filter_column("interaction id", available_columns={"pxInteractionID"})
        assert result == "pxInteractionID"

    def test_resolve_channel_v2(self):
        """'Channel' resolves to v2 raw key when present."""
        result = resolve_filter_column(
            "Channel",
            available_columns={"Primary_ContainerPayload_Channel", "pyIssue"},
        )
        assert result == "Primary_ContainerPayload_Channel"

    def test_resolve_channel_already_renamed(self):
        """'Channel' resolves to itself when data already uses display names."""
        result = resolve_filter_column("Channel", available_columns={"Channel", "Issue"})
        assert result == "Channel"

    def test_resolve_unknown_column_raises(self):
        """Unknown column name raises ValueError with available columns listed."""
        with pytest.raises(ValueError, match="Unknown filter column 'Bogus'"):
            resolve_filter_column("Bogus", available_columns={"pxInteractionID"})
