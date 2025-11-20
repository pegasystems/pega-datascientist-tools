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
        }
    ).lazy()
    return ADMDatamart(model_df=data)


def test_bubble_chart(sample: ADMDatamart):
    df = sample.plot.bubble_chart(return_df=True)

    assert df.select(pl.col("Performance").top_k(1)).collect().item() == 77.4901
    assert round(df.select(pl.col("SuccessRate").top_k(1)).collect().item(), 2) == 0.28
    assert df.select(
        pl.col("ResponseCount").top_k(1)
    ).collect().item() == pytest.approx(8174, abs=1e-6)
    plot = sample.plot.bubble_chart()
    assert plot is not None


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
        metric="Performance", by="ModelID", facet="Group"
    )
    assert fig_faceted is not None

    with pytest.raises(
        ValueError, match="The given query resulted in no more remaining data."
    ):
        sample2.plot.over_time(query=pl.col("ModelID") == "3")


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
        ValueError, match="There is no data for the provided modelid 'invalid_id'"
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
        model_id=model_id, predictor_name=predictor_name, return_df=True
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
            model_id="non_existent_id", predictor_name=predictor_name
        )
    with pytest.raises(ValueError):
        sample.plot.predictor_binning(
            model_id=model_id, predictor_name="non_existent_predictor"
        )

    plot = sample.plot.predictor_binning(
        model_id=model_id, predictor_name=predictor_name
    )
    assert isinstance(plot, Figure)

    assert len(plot.data) == 2
    assert plot.data[0].type == "bar"  # Responses
    assert plot.data[1].type == "scatter"  # Propensity


def test_multiple_predictor_binning(sample: ADMDatamart):
    model_id = sample.combined_data.select("ModelID").collect().row(0)[0]

    expected_predictor_count = (
        sample.combined_data.filter(pl.col("ModelID") == model_id)
        .select("PredictorName")
        .unique()
        .collect()
        .shape[0]
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
    assert (
        round(df.select(pl.col("median").top_k(1)).item(), 2)
        == 65.17
    )

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
        100, rel=1e-2
    )
    assert (
        round(
            df.filter(pl.col("PredictorCategory") == "Customer")
            .select("Contribution")
            .item(),
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
            df.filter(Name="Customer.AnnualIncome")
            .select("AutoUsed60Months")
            .collect()
            .item(),
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
            pl.col("Issue"), pl.col("Group"), pl.col("Name"), separator="/"
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
    df = sample.plot.predictor_count(
        query=(pl.col("Name") == "AutoNew36Months"), return_df=True
    ).collect()

    assert df.shape == (6, 5)
    assert df.row(5)[4] == 44

    plot = sample.plot.predictor_count()
    assert isinstance(plot, Figure)


def test_binning_lift(sample: ADMDatamart):
    # Get a random model_id and predictor_name
    random_row = (
        sample.combined_data.select(["ModelID", "PredictorName"]).collect().sample(1)
    )
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

    facets = {"A", "B"}
    plots = sample.plot.partitioned_plot(dummy_plot_func, facets, show_plots=False)
    assert isinstance(plots, list)
    assert len(plots) == len(facets)
    assert all(isinstance(plot, Figure) for plot in plots)
