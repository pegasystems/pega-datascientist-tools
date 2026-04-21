from datetime import datetime
from pathlib import Path

import plotly.express as px
import polars as pl
import pytest
from pdstools import ADMDatamart
from plotly.graph_objs import Figure

basePath = Path(__file__).parent.parent.parent


@pytest.fixture
def sample():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
    )


@pytest.fixture
def sample2():
    data = pl.DataFrame(
        {
            "SnapshotTime": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 3),
            ],
            "ModelID": [1, 2, 1, 2, 1, 2],
            "Performance": [0.75, 0.80, 0.78, 0.82, 0.77, 0.85],
            "ResponseCount": [100, 150, 120, 160, 110, 170],
            "Positives": [100, 150, 120, 160, 110, 170],
            "Group": ["A", "B", "A", "B", "A", "B"],
        },
    ).lazy()
    return ADMDatamart(model_df=data)


def test_bubble_chart(sample: ADMDatamart):
    df = sample.plot.bubble_chart(return_df=True)

    assert df.select(pl.col("Performance").top_k(1)).collect().item() == 77.4901
    assert round(df.select(pl.col("SuccessRate").top_k(1)).collect().item(), 2) == 0.28
    assert df.select(
        pl.col("ResponseCount").top_k(1),
    ).collect().item() == pytest.approx(8174, abs=1e-6)
    plot = sample.plot.bubble_chart()
    assert plot is not None


def test_bubble_chart_with_metric_limits(sample: ADMDatamart):
    """Test bubble chart with metric limit lines enabled."""
    fig = sample.plot.bubble_chart(show_metric_limits=True)
    assert isinstance(fig, Figure)

    # Should have 4 vertical lines (shapes) for the 4 ModelPerformance thresholds
    shapes = [s for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]
    assert len(shapes) == 4

    x_values = sorted(s.x0 for s in shapes)
    assert x_values == pytest.approx([52.0, 55.0, 80.0, 90.0])

    for shape in shapes:
        assert shape.line.dash == "dash"


def test_bubble_chart_metric_limits_off_by_default(sample: ADMDatamart):
    """Test that metric limit lines are not shown by default."""
    fig = sample.plot.bubble_chart()
    shapes = [s for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]
    assert len(shapes) == 0


def test_add_metric_limit_lines_standalone():
    """Test the helper function directly on a bare figure."""
    from pdstools.adm.Plots import add_metric_limit_lines

    fig = px.scatter(x=[50, 60, 70, 80, 90], y=[1, 2, 3, 4, 5])
    fig = add_metric_limit_lines(fig)

    shapes = [s for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]
    assert len(shapes) == 4

    x_values = sorted(s.x0 for s in shapes)
    assert x_values == pytest.approx([52.0, 55.0, 80.0, 90.0])

    # Hard limits (52, 90) should use red, best practice (55, 80) should use green
    for shape in shapes:
        if shape.x0 == pytest.approx(52.0) or shape.x0 == pytest.approx(90.0):
            assert "255, 69, 0" in shape.line.color  # red rgba
        else:
            assert "0, 128, 0" in shape.line.color  # green rgba


def test_add_metric_limit_lines_horizontal():
    """Test the helper function with horizontal orientation."""
    from pdstools.adm.Plots import add_metric_limit_lines

    fig = px.scatter(x=[1, 2, 3], y=[50, 70, 90])
    fig = add_metric_limit_lines(fig, orientation="horizontal")

    # Horizontal lines have y0 == y1
    shapes = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
    assert len(shapes) == 4

    y_values = sorted(s.y0 for s in shapes)
    assert y_values == pytest.approx([52.0, 55.0, 80.0, 90.0])


def test_over_time_with_metric_limits(sample2: ADMDatamart):
    """Test over_time with metric limit lines for Performance."""
    fig = sample2.plot.over_time(metric="Performance", by="ModelID", show_metric_limits=True)
    assert isinstance(fig, Figure)

    # Should have 4 horizontal lines
    shapes = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
    assert len(shapes) == 4


def test_over_time_metric_limits_only_for_performance(sample2: ADMDatamart):
    """Test that metric limits are not shown for non-Performance metrics."""
    fig = sample2.plot.over_time(metric="ResponseCount", by="ModelID", show_metric_limits=True)
    shapes = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
    assert len(shapes) == 0


def test_over_time(sample2: ADMDatamart):
    fig = sample2.plot.over_time(metric="Performance", by="ModelID")
    assert fig is not None

    fig = sample2.plot.over_time(metric="ResponseCount", by="ModelID")
    assert fig is not None

    performance_changes = (
        sample2.plot.over_time(metric="Performance", cumulative=False, return_df=True)
        .collect()
        .get_column("Performance_change")
        .to_list()
    )
    assert performance_changes == [None, 3.0, -1.0, None, 2.0, 3.0]

    responses_over_time = (
        sample2.plot.over_time(metric="ResponseCount", by="ModelID", return_df=True)
        .collect()
        .get_column("ResponseCount")
        .to_list()
    )
    assert responses_over_time == [100.0, 150.0, 120.0, 160.0, 110.0, 170.0]

    fig_faceted = sample2.plot.over_time(
        metric="Performance",
        by="ModelID",
        facet="Group",
    )
    assert fig_faceted is not None

    with pytest.raises(
        ValueError,
        match="The given query resulted in an empty dataframe",
    ):
        sample2.plot.over_time(query=pl.col("ModelID") == "3")


def test_over_time_multi_by(sample2: ADMDatamart):
    """`by` accepts a list — values are combined into one colour-encoded series."""
    fig = sample2.plot.over_time(metric="ResponseCount", by=["ModelID", "Group"])
    assert isinstance(fig, Figure)

    # Two ModelID/Group combinations -> two traces in the line chart
    trace_names = sorted(t.name for t in fig.data)
    assert trace_names == ["1 / A", "2 / B"]

    df = (
        sample2.plot.over_time(
            metric="ResponseCount",
            by=["ModelID", "Group"],
            return_df=True,
        )
        .collect()
        .sort("ModelID / Group", "SnapshotTime")
    )
    assert df.get_column("ModelID / Group").unique().sort().to_list() == [
        "1 / A",
        "2 / B",
    ]
    assert df.get_column("ResponseCount").to_list() == [
        100.0,
        120.0,
        110.0,
        150.0,
        160.0,
        170.0,
    ]


def test_proposition_success_rates(sample: ADMDatamart):
    df = sample.plot.proposition_success_rates(return_df=True)

    assert round(df.select(pl.col("SuccessRate").top_k(1)).collect().item(), 3) == 0.295
    assert df.select(pl.col("Name").top_k(1)).collect().item() == "VisaGold"

    plot = sample.plot.proposition_success_rates()
    assert plot is not None


def test_score_distribution(sample: ADMDatamart):
    model_id = "277a2c48-8888-5b71-911e-443d52c9b50f"

    df = sample.plot.score_distribution(model_id=model_id, return_df=True)

    required_columns = {"BinIndex", "BinSymbol", "BinResponseCount", "BinPropensity"}
    assert all(col in df.collect_schema().names() for col in required_columns)

    assert df.filter(pl.col("PredictorName") != "Classifier").collect().is_empty()

    collected_df = df.collect()
    bin_indices = collected_df["BinIndex"].to_list()
    assert bin_indices == sorted(bin_indices)

    with pytest.raises(
        ValueError,
        match="There is no data for the provided modelid 'invalid_id'",
    ):
        sample.plot.score_distribution(model_id="invalid_id")

    plot = sample.plot.score_distribution(model_id=model_id)
    assert plot is not None


def test_multiple_score_distributions(sample: ADMDatamart):
    model_ids = (
        sample.aggregates.last(table="combined_data")
        .filter(pl.col("PredictorName") == "Classifier")
        .select(pl.col("ModelID").unique())
        .collect()
    )

    assert not model_ids.is_empty()

    plots = sample.plot.multiple_score_distributions(show_all=False)

    assert len(plots) == len(model_ids)

    assert all(isinstance(plot, Figure) for plot in plots)

    # Verify each plot has the required traces
    for plot in plots:
        assert len(plot.data) == 2
        assert plot.data[0].type == "bar"  # Responses
        assert plot.data[1].type == "scatter"  # Propensity


def test_predictor_binning(sample: ADMDatamart):
    first_row = (
        sample.combined_data.select(["ModelID", "PredictorName"])
        .filter(pl.col("PredictorName") != "Classifier")
        .collect()
        .row(0)
    )
    model_id = first_row[0]
    predictor_name = first_row[1]

    df = sample.plot.predictor_binning(
        model_id=model_id,
        predictor_name=predictor_name,
        return_df=True,
    )

    assert not df.collect().is_empty()

    required_columns = ["BinIndex", "BinPropensity", "BinSymbol", "BinResponseCount"]
    assert all(col in df.collect_schema().names() for col in required_columns)

    collected_df = df.collect()
    bin_indices = collected_df["BinIndex"].to_list()
    assert bin_indices == sorted(bin_indices)

    # Test error handling for invalid inputs
    with pytest.raises(ValueError):
        sample.plot.predictor_binning(
            model_id="non_existent_id",
            predictor_name=predictor_name,
        )
    with pytest.raises(ValueError):
        sample.plot.predictor_binning(
            model_id=model_id,
            predictor_name="non_existent_predictor",
        )

    plot = sample.plot.predictor_binning(
        model_id=model_id,
        predictor_name=predictor_name,
    )
    assert isinstance(plot, Figure)

    assert len(plot.data) == 2
    assert plot.data[0].type == "bar"  # Responses
    assert plot.data[1].type == "scatter"  # Propensity


def test_multiple_predictor_binning(sample: ADMDatamart):
    model_id = sample.combined_data.select("ModelID").collect().row(0)[0]

    expected_predictor_count = (
        sample.combined_data.filter(pl.col("ModelID") == model_id).select("PredictorName").unique().collect().shape[0]
    )

    plots = sample.plot.multiple_predictor_binning(model_id=model_id, show_all=False)

    assert isinstance(plots, list)

    assert len(plots) == expected_predictor_count

    assert all(isinstance(plot, Figure) for plot in plots)

    for plot in plots:
        assert len(plot.data) == 2
        assert plot.data[0].type == "bar"  # Responses
        assert plot.data[1].type == "scatter"  # Propensity

    sample.plot.multiple_predictor_binning(model_id="invalid_id", show_all=False) == []


def test_predictor_performance(sample: ADMDatamart):
    df = sample.plot.predictor_performance(return_df=True)
    assert "PredictorName" in df.collect_schema().names()
    assert "median" in df.collect_schema().names()
    assert round(df.select(pl.col("median").top_k(1)).item(), 2) == 65.17

    plot = sample.plot.predictor_performance()
    assert isinstance(plot, Figure)


def test_predictor_category_performance(sample: ADMDatamart):
    df = sample.plot.predictor_category_performance(return_df=True)
    assert df.shape == (3, 10)
    assert "PredictorCategory" in df.columns
    assert "mean" in df.columns
    assert round(df.select(pl.col("mean").top_k(1)).item(), 2) == 56.98

    plot = sample.plot.predictor_category_performance()
    assert isinstance(plot, Figure)


def test_predictor_contribution(sample: ADMDatamart):
    df = sample.plot.predictor_contribution(return_df=True).collect()
    assert df.shape == (3, 4)
    assert "PredictorCategory" in df.columns
    assert "Contribution" in df.columns
    assert df.select(pl.col("Contribution").sum()).item() == pytest.approx(
        100,
        rel=1e-2,
    )
    assert (
        round(
            df.filter(pl.col("PredictorCategory") == "Customer").select("Contribution").item(),
            2,
        )
        == 39.02
    )

    plot = sample.plot.predictor_contribution()
    assert isinstance(plot, Figure)


def test_predictor_performance_heatmap(sample: ADMDatamart):
    df = sample.plot.predictor_performance_heatmap(return_df=True)

    assert (
        round(
            df.filter(Name="Customer.AnnualIncome").select("AutoUsed60Months").collect().item(),
            5,
        )
        == 0.69054
    )
    assert df.select(pl.col("*").exclude("Name")).collect().to_numpy().max() <= 1.0
    assert "Name" in df.collect().columns
    assert "ChannelAction_Template" in df.collect().columns

    plot = sample.plot.predictor_performance_heatmap()
    assert isinstance(plot, Figure)

    df = sample.plot.predictor_performance_heatmap(
        by=pl.concat_str(
            pl.col("Issue"),
            pl.col("Group"),
            pl.col("Name"),
            separator="/",
        ),
        return_df=True,
    )

    assert "Name" not in df.collect().columns
    assert "Predictor" in df.collect().columns
    assert "Sales/CreditCards/ChannelAction_Template" in df.collect().columns


def test_tree_map(sample: ADMDatamart):
    df = sample.plot.tree_map(return_df=True).collect()
    assert df.shape == (68, 10)
    assert df.sort("Total number of positives").row(0)[-1] == 2.0
    plot = sample.plot.tree_map()
    assert isinstance(plot, Figure)


def test_predictor_count(sample: ADMDatamart):
    df = sample.plot.predictor_count(return_df=True)

    assert df.height > 0  # Should have some data
    assert "Count" in df.collect_schema().names()
    assert "Type" in df.collect_schema().names()
    assert "EntryType" in df.collect_schema().names()

    plot = sample.plot.predictor_count()
    assert isinstance(plot, Figure)


def test_binning_lift(sample: ADMDatamart):
    # Get a random model_id and predictor_name
    random_row = sample.combined_data.select(["ModelID", "PredictorName"]).collect().sample(1)
    model_id = random_row["ModelID"][0]
    predictor_name = random_row["PredictorName"][0]

    df = sample.plot.binning_lift(model_id, predictor_name, return_df=True).collect()

    assert not df.is_empty()
    required_columns = [
        "PredictorName",
        "BinIndex",
        "BinPositives",
        "BinNegatives",
        "BinSymbol",
        "Lift",
        "Direction",
        "BinSymbolAbbreviated",
    ]
    assert all(col in df.columns for col in required_columns)

    assert set(df["Direction"]) <= {"pos", "neg", "pos_shaded", "neg_shaded"}
    plot = sample.plot.binning_lift(model_id, predictor_name)
    assert isinstance(plot, Figure)


def test_partitioned_plot(sample: ADMDatamart):
    def dummy_plot_func(*args, **kwargs):
        return px.scatter(x=[1, 2, 3], y=[1, 2, 3])

    facets = [{"Aspect": "A"}, {"Aspect": "B"}]
    plots = sample.plot.partitioned_plot(dummy_plot_func, facets, show_plots=False)
    assert isinstance(plots, list)
    assert len(plots) == len(facets)
    assert all(isinstance(plot, Figure) for plot in plots)


def test_gains_chart_basic(sample: ADMDatamart):
    """Test basic gains chart creation."""
    fig = sample.plot.gains_chart(value="ResponseCount")
    assert isinstance(fig, Figure)
    assert fig.layout.xaxis.title.text == "% of Population"
    assert fig.layout.yaxis.title.text == "% of Responders"


def test_gains_chart_return_df(sample: ADMDatamart):
    """Test gains chart data return."""
    df = sample.plot.gains_chart(value="ResponseCount", return_df=True)
    assert isinstance(df, pl.LazyFrame)
    schema = df.collect_schema().names()
    assert "cum_x" in schema
    assert "cum_y" in schema

    # Verify data makes sense
    collected = df.collect()
    assert collected.height > 0
    # Cumulative y should end at 1.0 (100%)
    assert collected["cum_y"][-1] == pytest.approx(1.0, abs=0.01)


def test_gains_chart_with_grouping(sample: ADMDatamart):
    """Test gains chart with by parameter."""
    fig = sample.plot.gains_chart(value="ResponseCount", by="Channel")
    assert isinstance(fig, Figure)
    # Should have multiple traces, one per Channel
    assert len(fig.data) > 1


def test_gains_chart_with_index(sample: ADMDatamart):
    """Test gains chart with index parameter."""
    df = sample.plot.gains_chart(value="Positives", index="ResponseCount", return_df=True)
    assert isinstance(df, pl.LazyFrame)
    # Verify index column is used for sorting
    collected = df.collect()
    assert collected.height > 0


def test_gains_chart_with_query(sample: ADMDatamart):
    """Test gains chart with query filter."""
    # Use a filter that keeps some but not all data
    df = sample.plot.gains_chart(value="ResponseCount", query=pl.col("Channel") == "Email", return_df=True)
    assert isinstance(df, pl.LazyFrame)
    # Should still return data for filtered subset
    collected = df.collect()
    assert collected.height > 0


def test_performance_volume_distribution_basic(sample: ADMDatamart):
    """Test basic performance volume distribution chart."""
    fig = sample.plot.performance_volume_distribution()
    assert isinstance(fig, Figure)


def test_performance_volume_distribution_return_df(sample: ADMDatamart):
    """Test performance volume distribution data return."""
    df = sample.plot.performance_volume_distribution(return_df=True)
    assert isinstance(df, pl.LazyFrame)
    schema = df.collect_schema().names()
    assert "PerformanceBinned" in schema
    assert "ResponseCount" in schema

    # Verify binning worked
    collected = df.collect()
    assert collected.height > 0


def test_performance_volume_distribution_with_grouping(sample: ADMDatamart):
    """Test performance volume distribution with by parameter."""
    fig = sample.plot.performance_volume_distribution(by="Channel")
    assert isinstance(fig, Figure)
    # Should have multiple traces for different channels
    assert len(fig.data) > 1


def test_performance_volume_distribution_bin_width(sample: ADMDatamart):
    """Test performance volume distribution with custom bin width."""
    df = sample.plot.performance_volume_distribution(bin_width=5, return_df=True)
    assert isinstance(df, pl.LazyFrame)
    collected = df.collect()
    # Verify binning created expected number of bins
    num_bins = collected["PerformanceBinned"].n_unique()
    assert num_bins == 6  # Performance range 50-80 with bin_width=5 produces 6 bins


def test_performance_volume_distribution_with_query(sample: ADMDatamart):
    """Test performance volume distribution with query filter."""
    df = sample.plot.performance_volume_distribution(query=pl.col("ResponseCount") > 100, return_df=True)
    assert isinstance(df, pl.LazyFrame)
    assert df.collect().height > 0


# ---------------------------------------------------------------------------
# Color-consistency tests for box plots
# ---------------------------------------------------------------------------


def test_predictor_category_color_map_stable(sample: ADMDatamart):
    """predictor_category_color_map returns a non-empty, stable mapping."""
    color_map = sample.predictor_category_color_map
    assert isinstance(color_map, dict)
    assert len(color_map) > 0
    # All values should be valid hex color strings
    for cat, color in color_map.items():
        assert isinstance(color, str), f"Color for {cat!r} is not a string"
        assert color.startswith("#"), f"Color {color!r} for {cat!r} is not a hex color"
    # Calling twice returns the same object (cached_property)
    assert sample.predictor_category_color_map is color_map


def test_predictor_performance_consistent_colors(sample: ADMDatamart):
    """Same predictor category gets the same colour in two different filtered plots."""
    fig_all = sample.plot.predictor_performance()
    assert isinstance(fig_all, Figure)

    # Filter to a subset of models so a different subset of categories may appear
    fig_subset = sample.plot.predictor_performance(active_only=True)
    assert isinstance(fig_subset, Figure)

    # Build {name -> marker.color} from each figure's traces
    def trace_colors(fig: Figure) -> dict[str, str]:
        return {trace.name: trace.marker.color for trace in fig.data}

    colors_all = trace_colors(fig_all)
    colors_subset = trace_colors(fig_subset)

    # Any category present in both plots must have the same color
    shared = set(colors_all) & set(colors_subset)
    assert shared, "No shared categories between full and subset plot — cannot test consistency"
    for cat in shared:
        assert colors_all[cat] == colors_subset[cat], (
            f"Category {cat!r} has color {colors_all[cat]!r} in full plot but {colors_subset[cat]!r} in subset plot"
        )


def test_predictor_category_performance_consistent_colors(sample: ADMDatamart):
    """predictor_category_performance colors match the global color map."""
    fig = sample.plot.predictor_category_performance()
    assert isinstance(fig, Figure)

    color_map = sample.predictor_category_color_map
    for trace in fig.data:
        assert trace.name in color_map, f"Category {trace.name!r} missing from color map"
        assert trace.marker.color == color_map[trace.name], (
            f"Category {trace.name!r}: expected {color_map[trace.name]!r}, got {trace.marker.color!r}"
        )
