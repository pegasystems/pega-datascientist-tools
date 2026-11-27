import polars as pl
import pytest
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
from pdstools.decision_analyzer.plots import offer_quality_piecharts

# The fixture below intentionally omits some columns that DecisionAnalyzer
# treats as defaults; the resulting "default columns are missing" UserWarning
# is expected and not under test here.
pytestmark = pytest.mark.filterwarnings("ignore:The following default columns are missing:UserWarning")


@pytest.fixture
def sample_data():
    """Create sample decision data with multiple categorical dimensions."""
    return pl.LazyFrame(
        {
            "pyIssue": ["Sales", "Retention", "Service"] * 10,
            "pyGroup": ["Cards", "Loans", "Insurance"] * 10,
            "pyName": ["Action1", "Action2", "Action3"] * 10,
            "pyTreatment": ["Email", "SMS", "Web"] * 10,
            "pyChannel": ["Web", "Email", "Mobile"] * 10,
            "pyDirection": ["Inbound", "Outbound", "Inbound"] * 10,
            "Stage Group": ["Filtering", "Arbitration", "Filtering"] * 10,
            "Stage": ["Stage1", "Stage2", "Stage3"] * 10,
            "Priority": [0.5, 0.6, 0.7] * 10,
            "Propensity": [0.1, 0.2, 0.3] * 10,
            "Interaction ID": list(range(30)),
            "Decision Time": ["2024-01-01"] * 30,
            "pySubjectID": ["Subject1"] * 30,
            "Value": [100.0, 200.0, 300.0] * 10,
            "ContextWeight": [1.0] * 30,
            "Weight": [1.0] * 30,
            "pyPropensity": [0.1, 0.2, 0.3] * 10,
            "ModelControlGroup": ["Control"] * 30,
        }
    )


def test_color_mappings_property_exists(sample_data):
    """Test that color_mappings property exists and returns a dict."""
    da = DecisionAnalyzer(sample_data)
    assert hasattr(da, "color_mappings")
    mappings = da.color_mappings
    assert isinstance(mappings, dict)


@pytest.mark.parametrize("level", ["Issue", "Group", "Action"])
@pytest.mark.parametrize("after", [[0, 1, 1], [0, 0, 0]])
def test_simulation_distributions_keep_all_categories_and_baseline_shares(sample_data, level, after):
    da = DecisionAnalyzer(sample_data)
    keys = ["Issue", "Group", "Action"][: ["Issue", "Group", "Action"].index(level) + 1]
    distribution = (
        pl.DataFrame(
            {
                "Issue": ["Sales", "Retention", "Service"],
                "Group": ["Cards", "Loans", "Insurance"],
                "Action": ["Action1", "Action2", "Action3"],
                "Before": [2, 0, 1],
                "After": after,
            }
        )
        .select(*keys, "Before", "After")
        .with_columns(
            (25.0 * pl.col("Before")).alias("Before share (%)"),
            (25.0 * pl.col("After")).alias("After share (%)"),
        )
    )
    data = da.plot.simulation_distribution(distribution, return_df=True)
    labels = [
        " / ".join(values[: len(keys)])
        for values in [
            ("Sales", "Cards", "Action1"),
            ("Service", "Insurance", "Action3"),
            ("Retention", "Loans", "Action2"),
        ]
    ]
    assert data.select("Label", "Before", "After").rows() == [
        (labels[0], 2, after[0]),
        (labels[1], 1, after[2]),
        (labels[2], 0, after[1]),
    ]
    fig = da.plot.simulation_distribution(distribution)
    assert [trace.name for trace in fig.data] == labels
    assert [list(trace.y) for trace in fig.data] == [["Before", "After"]] * 3
    assert [list(trace.x) for trace in fig.data] == [
        [50.0, 25.0 * after[0]],
        [25.0, 25.0 * after[2]],
        [0.0, 25.0 * after[1]],
    ]
    assert [list(trace.customdata) for trace in fig.data] == [
        [[2], [after[0]]],
        [[1], [after[2]]],
        [[0], [after[1]]],
    ]
    if level != "Action":
        assert [trace.marker.color for trace in fig.data] == [da.color_mappings[level][value] for value in data[level]]
    else:
        assert len({trace.marker.color for trace in fig.data}) == 3
    assert fig.layout.barmode == "stack"
    assert fig.layout.legend.traceorder == "normal"
    assert fig.layout.xaxis.range == (0, 100)


def test_simulation_action_colors_distinguish_baseline_winners_and_survive_scenario_changes(sample_data):
    da = DecisionAnalyzer(sample_data)
    distribution = pl.DataFrame(
        {
            "Issue": ["Sales"] * 12,
            "Group": ["Cards"] * 12,
            "Action": [f"Action{i:02}" for i in range(12)],
            "Before": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 0, 0],
            "After": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 0, 0],
        }
    ).with_columns(
        pl.col("Before").cast(pl.Float64).alias("Before share (%)"),
        pl.col("After").cast(pl.Float64).alias("After share (%)"),
    )
    before = da.plot.simulation_distribution(distribution)
    changed = distribution.with_columns(
        pl.when(pl.col("Action") == "Action11").then(2).otherwise(pl.col("After")).alias("After"),
    ).with_columns(pl.col("After").cast(pl.Float64).alias("After share (%)"))
    after = da.plot.simulation_distribution(changed.reverse())
    before_colors = {trace.name: trace.marker.color for trace in before.data}
    after_colors = {trace.name: trace.marker.color for trace in after.data}

    assert before_colors == after_colors
    assert len({before_colors[f"Sales / Cards / Action{i:02}"] for i in range(10)}) == 10
    assert not set(before_colors.values()) & {"#63666F", "#A7A9B4", "#D0D1DB", "#cccccc"}


def test_color_mappings_includes_all_categorical_columns(sample_data):
    """Test that color_mappings includes all categorical dimensions."""
    da = DecisionAnalyzer(sample_data)
    mappings = da.color_mappings

    # Check that key dimensions present in the processed data are included
    # Stage Group is always present, Stage only in v2 data
    expected_columns = ["Issue", "Group", "Action", "Channel", "Direction", "Stage Group"]
    for col in expected_columns:
        assert col in mappings, f"Missing {col} in color_mappings"
        assert isinstance(mappings[col], dict), f"{col} mapping should be a dict"

    # Optional columns - only check if present in the data
    optional_columns = ["Treatment", "Stage"]
    for col in optional_columns:
        if col in da.decision_data.collect_schema().names():
            assert col in mappings, f"{col} is in data but missing from color_mappings"


def test_color_mappings_assigns_consistent_colors(sample_data):
    """Test that colors are assigned consistently based on sorted unique values."""
    da = DecisionAnalyzer(sample_data)
    mappings = da.color_mappings

    # Issue should map sorted values to colors
    issue_mapping = mappings["Issue"]
    assert "Sales" in issue_mapping
    assert "Retention" in issue_mapping
    assert "Service" in issue_mapping

    # All values should have color strings (hex codes)
    for color in issue_mapping.values():
        assert isinstance(color, str)
        assert color.startswith("#")


def test_color_mappings_is_cached(sample_data):
    """Test that color_mappings is computed only once (cached)."""
    da = DecisionAnalyzer(sample_data)

    # Access twice, should return same object (same id)
    mappings1 = da.color_mappings
    mappings2 = da.color_mappings
    assert mappings1 is mappings2


def test_distribution_treemap_uses_consistent_colors(sample_data):
    """Test that treemap uses color_mappings instead of computing dynamically."""
    da = DecisionAnalyzer(sample_data)

    # Verify color_mappings has Issue dimension
    assert "Issue" in da.color_mappings

    # Create treemap plot with Issue as primary scope
    scope_options = ["Issue"]
    fig = da.plot.distribution_as_treemap(
        da.preaggregated_remaining_view.filter(pl.col("Stage Group") == "Arbitration"),
        stage="Arbitration",
        scope_options=scope_options,
    )

    assert [
        (
            trace.type,
            list(trace.labels),
            list(trace.values),
            list(trace.marker.colors),
        )
        for trace in fig.data
    ] == [
        (
            "treemap",
            ["All Actions Arbitration", "Retention", "Sales", "Service"],
            [30.0, 10.0, 10.0, 10.0],
            [
                "#661D34",
                da.color_mappings["Issue"]["Retention"],
                da.color_mappings["Issue"]["Sales"],
                da.color_mappings["Issue"]["Service"],
            ],
        ),
    ]


@pytest.fixture
def mixed_issue_data():
    return pl.LazyFrame(
        {
            "Interaction ID": ["D1", "D1", "D1", "D2", "D2", "D3"],
            "Issue": ["A", "A", "B", "A", "B", "C"],
            "Group": ["G1", "G1", "G2", "G1", "G2", "G3"],
            "Action": ["A1", "A2", "B1", "A1", "B1", "C1"],
            "Priority": [0.3, 0.2, 0.1, 0.2, 0.1, None],
            "Propensity": [0.3, 0.2, 0.1, 0.2, 0.1, None],
            "Value": [1.0] * 6,
            "Context Weight": [1.0] * 6,
            "Levers": [1.0] * 6,
            "Decision Time": ["2024-01-01"] * 6,
            "Stage Group": ["Arbitration"] * 5 + ["Eligibility"],
            "Stage": ["Rank"] * 5 + ["Eligible"],
            "Stage Order": [3800] * 5 + [1500],
            "Record Type": ["FILTERED_OUT"] * 6,
            "Component Name": ["Ranker"] * 5 + ["Eligibility rule"],
            "Strategy Name": ["Strategy"] * 6,
        }
    )


def test_action_colors_are_bounded_popularity_based_and_filter_stable(mixed_issue_data):
    extra_actions = (
        mixed_issue_data.head(1)
        .collect()
        .clear(12)
        .with_columns(
            pl.Series("Action", [f"Rare{i:02}" for i in range(12)]),
            pl.Series("Interaction ID", [f"R{i}" for i in range(12)]),
            pl.lit("A").alias("Issue"),
            pl.lit("G1").alias("Group"),
            pl.lit("Arbitration").alias("Stage Group"),
            pl.lit("Rank").alias("Stage"),
            pl.lit(3800).alias("Stage Order"),
            pl.lit(0.1).alias("Priority"),
            pl.lit(0.1).alias("Propensity"),
            pl.lit("Strategy").alias("Strategy Name"),
            pl.lit("FILTERED_OUT").alias("Record Type"),
            pl.lit("Ranker").alias("Component Name"),
            pl.lit("2024-01-01").alias("Decision Time"),
        )
    )
    da = DecisionAnalyzer(pl.concat([mixed_issue_data, extra_actions.lazy().cast(mixed_issue_data.collect_schema())]))
    colors = da.color_mappings["Action"]
    colored_actions = {action for action, color in colors.items() if color != "#A7A9B4"}
    assert colored_actions == {"A1", "B1", "A2", "C1", *[f"Rare{i:02}" for i in range(6)]}
    assert len({colors[action] for action in colored_actions}) == 10
    assert colors["Rare11"] == "#A7A9B4"
    pies = da.plot.global_winloss_distribution(level="Action", win_rank=1)
    for trace in pies.data:
        assert list(trace.marker.colors) == [colors[label] for label in trace.labels]
    assert any(colors[label] == "#A7A9B4" for trace in pies.data for label in trace.labels)
    frame = pl.LazyFrame({"Action": ["Rare11", "B1"], "Decisions": [1, 2]})
    fig = da.plot.distribution(frame, "Action", None)
    assert {trace.name: trace.marker.color for trace in fig.data} == {
        "Rare11": "#A7A9B4",
        "B1": colors["B1"],
    }


def test_stage_and_component_colors_follow_the_same_family(mixed_issue_data):
    da = DecisionAnalyzer(mixed_issue_data)
    assert da.color_mappings["Stage"]["Eligible"] == da.color_mappings["Stage Group"]["Eligibility"]
    assert da.color_mappings["Stage"]["Rank"] == da.color_mappings["Stage Group"]["Arbitration"]
    fig = da.plot.component_action_impact()
    assert {trace.name: trace.marker.color for trace in fig.data} == {
        "Eligibility rule": da.color_mappings["Stage Group"]["Eligibility"],
        "Ranker": da.color_mappings["Stage Group"]["Arbitration"],
    }
    stages = list(da.AvailableNBADStages)
    fig = da.plot.filtering_components(stages=stages, top_n=5, AvailableNBADStages=stages)
    assert {trace.name: trace.marker.color for trace in fig.data} == {
        "Eligibility rule": da.color_mappings["Stage Group"]["Eligibility"],
        "Ranker": da.color_mappings["Stage Group"]["Arbitration"],
    }
    da.set_level("Stage")
    fig = da.plot.component_action_impact()
    assert {trace.name: trace.marker.color for trace in fig.data} == {
        "Eligibility rule": da.color_mappings["Stage Group"]["Eligibility"],
        "Ranker": da.color_mappings["Stage Group"]["Arbitration"],
    }


def test_engagement_policy_components_use_specific_filter_families(mixed_issue_data):
    raw = mixed_issue_data.head(3).with_columns(
        pl.lit("EngagementPolicies").alias("Stage Group"),
        pl.Series("Stage", ["AllActionsEligibility", "AllActionsSuitability", "AllActionsApplicability"]),
        pl.Series("Component Name", ["Eligibility rule", "Suitability rule", "Applicability rule"]),
    )
    da = DecisionAnalyzer(raw)
    assert da.color_mappings["Stage"]["Issue/Group: All Actions - Eligibility"] == "#001F5F"
    assert da.color_mappings["Stage"]["Issue/Group: All Actions - Suitability"] == "#F76923"
    assert da.color_mappings["Stage"]["Issue/Group: All Actions - Applicability"] == "#005154"
    fig = da.plot.component_action_impact()
    assert {trace.name: trace.marker.color for trace in fig.data} == {
        "Eligibility rule": "#001F5F",
        "Suitability rule": "#F76923",
        "Applicability rule": "#005154",
    }
    shared = DecisionAnalyzer(raw.with_columns(pl.lit("Shared rule").alias("Component Name")))
    assert shared.color_mappings["Component Name"]["Shared rule"] == "#A7A9B4"


def test_win_loss_colors_do_not_depend_on_order_or_count(mixed_issue_data):
    da = DecisionAnalyzer(mixed_issue_data)
    first = pl.LazyFrame({"Issue": ["A", "B"], "Decisions": [100, 1]})
    second = pl.LazyFrame({"Issue": ["B", "A"], "Decisions": [20, 1]})
    for data in (first, second):
        fig = da.plot.distribution(data, "Issue", None, horizontal=True)
        assert {trace.name: trace.marker.color for trace in fig.data} == {
            "A": da.color_mappings["Issue"]["A"],
            "B": da.color_mappings["Issue"]["B"],
        }
        assert all(trace.marker.opacity == 1 and trace.marker.line.width == 0.75 for trace in fig.data)
        assert sorted(float(trace.x[0]) for trace in fig.data) == sorted(data.collect()["Decisions"].to_list())


@pytest.mark.parametrize(("dimension", "a", "b"), [("Issue", "A", "B"), ("Group", "G1", "G2")])
def test_optionality_color_split_preserves_counts_and_propensity(mixed_issue_data, dimension, a, b):
    da = DecisionAnalyzer(mixed_issue_data)
    plain = da.plot.propensity_vs_optionality(stage="Arbitration")
    colored = da.plot.propensity_vs_optionality(stage="Arbitration", color_by=dimension)
    bar_values = {trace.name: dict(zip(trace.x, trace.y, strict=True)) for trace in colored.data if trace.type == "bar"}
    assert bar_values[a][3] == pytest.approx(100 * 2 / 9)
    assert bar_values[b][3] == pytest.approx(100 / 9)
    assert bar_values[a][2] == pytest.approx(100 / 6)
    assert bar_values[b][2] == pytest.approx(100 / 6)
    assert bar_values["No actions"][0] == pytest.approx(100 / 3)
    assert sum(sum(values.values()) for values in bar_values.values()) == pytest.approx(100)
    assert list(colored.data[-1].x) == list(plain.data[-1].x)
    assert list(colored.data[-1].y) == list(plain.data[-1].y)
    filtered = mixed_issue_data.filter(pl.col("Issue") == "A")
    filtered_fig = da.plot.propensity_vs_optionality(stage="Arbitration", df=filtered, color_by=dimension)
    assert filtered_fig.data[0].marker.color == da.color_mappings[dimension][a]
    assert sum(filtered_fig.data[0].y) == pytest.approx(100)


def test_colored_optionality_preserves_missing_propensity(mixed_issue_data):
    da = DecisionAnalyzer(mixed_issue_data.with_columns(pl.lit(None, dtype=pl.Float64).alias("Propensity")))
    data = da.plot.propensity_vs_optionality(color_by="Issue", return_df=True).collect()
    assert data["AverageBestPropensity"].null_count() == data.filter(pl.col("nOffers") > 0).height
    assert all(trace.type == "bar" for trace in da.plot.propensity_vs_optionality(color_by="Issue").data)


def test_colored_optionality_preserves_null_interaction_ids(mixed_issue_data):
    raw = mixed_issue_data.with_columns(
        pl.when(pl.col("Interaction ID") == "D1").then(None).otherwise(pl.col("Interaction ID")).alias("Interaction ID")
    )
    da = DecisionAnalyzer(raw)
    data = da.plot.propensity_vs_optionality(color_by="Issue", return_df=True).collect()
    assert data["Interactions"].sum() == pytest.approx(3)
    assert data.filter((pl.col("Issue") == "A") & (pl.col("nOffers") == 3))["Interactions"].item() == pytest.approx(
        2 / 3
    )


@pytest.mark.parametrize("horizontal", [False, True])
def test_stacked_distribution_hover_has_aggregated_segment_and_bar_total(mixed_issue_data, horizontal):
    da = DecisionAnalyzer(mixed_issue_data)
    data = pl.LazyFrame(
        {"Issue": ["A", "A", "A", "B", "B"], "Group": ["G1", "G1", "G2", "G1", "G2"], "Decisions": [2, 3, 7, 11, 13]}
    )
    fig = da.plot.distribution(data, "Issue", "Group", horizontal=horizontal)
    values = {}
    for trace in fig.data:
        categories, counts = (trace.y, trace.x) if horizontal else (trace.x, trace.y)
        for category, count, custom in zip(categories, counts, trace.customdata, strict=True):
            values[(category, trace.name)] = (count, custom[0])
        assert "Bar total: %{customdata[0]:,}" in trace.hovertemplate
    assert values == {("A", "G1"): (5, 12), ("A", "G2"): (7, 12), ("B", "G1"): (11, 24), ("B", "G2"): (13, 24)}


@pytest.mark.parametrize("color_by", [None, "Issue", "Group"])
def test_optionality_hover_counts_and_stacked_bucket_totals(mixed_issue_data, color_by):
    da = DecisionAnalyzer(mixed_issue_data)
    fig = da.plot.propensity_vs_optionality(color_by=color_by)
    for trace in fig.data:
        assert "customdata[0]" in trace.hovertemplate
        if trace.type == "scatter" or color_by is None:
            assert [row[0] for row in trace.customdata] == [1] * len(trace.x)
        else:
            for row in trace.customdata:
                assert row[1] == pytest.approx(1)
                assert row[2] == pytest.approx(100 / 3)
            assert "Bucket total" in trace.hovertemplate
            assert "weighted decisions" in trace.hovertemplate
    if color_by is not None:
        a = next(trace for trace in fig.data if trace.name == ("A" if color_by == "Issue" else "G1"))
        n_three = list(a.x).index(3)
        assert a.customdata[n_three][0] == pytest.approx(2 / 3)


def test_win_loss_hover_uses_absolute_counts_without_changing_shares(mixed_issue_data):
    da = DecisionAnalyzer(mixed_issue_data)
    data = da.scoring.get_win_loss_distribution_data("Issue", 1).collect()
    assert {(row["Issue"], row["Status"]): row["Count"] for row in data.iter_rows(named=True)} == {
        ("A", "Wins"): 2,
        ("B", "Wins"): 0,
        ("A", "Losses"): 1,
        ("B", "Losses"): 2,
    }
    for status, trace in zip(("Wins", "Losses"), da.plot.global_winloss_distribution("Issue", 1).data, strict=True):
        assert "%{value:,}" in trace.hovertemplate
        assert dict(zip(trace.labels, trace.values, strict=True)) == (
            {"A": 2, "B": 0} if status == "Wins" else {"A": 1, "B": 2}
        )
        assert data.filter(pl.col("Status") == status)["Percentage"].sum() == pytest.approx(1)


def test_without_actions_hover_has_absolute_interaction_counts(mixed_issue_data):
    da = DecisionAnalyzer(mixed_issue_data)
    trace = da.plot.decisions_without_actions_plot().data[0]
    assert {
        stage: (share, count[0]) for stage, share, count in zip(trace.y, trace.x, trace.customdata, strict=True)
    } == {
        "Eligibility": (pytest.approx(100 / 3), 1),
        "Arbitration": (pytest.approx(200 / 3), 2),
    }
    assert "%{customdata[0]:,}" in trace.hovertemplate


def test_lever_summary_centers_replacement_on_selected_source_values(mixed_issue_data):
    da = DecisionAnalyzer(
        mixed_issue_data.with_columns(pl.when(pl.col("Action") == "A1").then(2.0).otherwise(1.0).alias("Levers"))
    )
    settings = da.aggregates.lever_summary(pl.col("Issue") == "A", percentage=18).row(0, named=True)
    assert settings == {
        "Minimum": 1.0,
        "Median": 2.0,
        "Maximum": 2.0,
        "All mandatory": False,
        "Replacement": pytest.approx(2.36),
        "Lowers existing levers": False,
    }
    distribution, summary = da.scoring.lever_simulation(pl.col("Issue") == "A", settings["Replacement"], by="Issue")
    assert summary.select("Before", "After").row(0) == (2, 2)
    assert distribution.filter(pl.col("Selected"))["After"].item() == 2


@pytest.mark.parametrize("percentage", [51, float("nan")])
def test_lever_summary_rejects_invalid_adjustments(mixed_issue_data, percentage):
    da = DecisionAnalyzer(mixed_issue_data)
    with pytest.raises(ValueError, match="between -50% and \\+50%"):
        da.aggregates.lever_summary(pl.col("Issue") == "A", percentage=percentage)


def test_lever_summary_rejects_empty_scope(mixed_issue_data):
    da = DecisionAnalyzer(mixed_issue_data)
    with pytest.raises(ValueError, match="positive, finite median lever"):
        da.aggregates.lever_summary(pl.col("Issue") == "Absent")


def test_offer_quality_already_colors_stage_specific_propensity_categories(mixed_issue_data):
    early_rows = mixed_issue_data.filter(pl.col("Interaction ID") == "D3")
    raw = pl.concat(
        [
            mixed_issue_data,
            early_rows.with_columns(pl.lit("D1").alias("Interaction ID")),
            early_rows.with_columns(pl.lit("D2").alias("Interaction ID")),
        ]
    )
    da = DecisionAnalyzer(raw)
    counts = da.aggregates.filtered_action_counts(
        ["Stage Group", "Interaction ID"],
        propensity_threshold=0.25,
        priority_threshold=0.0,
    )
    quality = da.aggregates.get_offer_quality(counts, "Interaction ID")
    fig = offer_quality_piecharts(quality, 0.25, da.AvailableNBADStages)
    values = {trace.name: list(trace.values) for trace in fig.data}
    assert values["Eligibility"] == [0, 3, 0, 0]
    assert values["Arbitration"] == [1, 0, 1, 1]
    assert all(list(trace.marker.colors) == ["#219e3f", "#4A90E2", "#fca52e", "#cd001f"] for trace in fig.data)
