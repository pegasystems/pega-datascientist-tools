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


def test_bubble_chart(sample: ADMDatamart):
    df = sample.plot.bubble_chart(return_df=True)

    assert df.select(pl.col("Performance").top_k(1)).collect().item() == 77.4901
    assert round(df.select(pl.col("SuccessRate").top_k(1)).collect().item(), 2) == 0.28
    assert df.select(
        pl.col("ResponseCount").top_k(1)
    ).collect().item() == pytest.approx(8174, abs=1e-6)
    plot = sample.plot.bubble_chart()
    assert plot is not None


def test_over_time(sample: ADMDatamart):
    df = sample.plot.over_time(return_df=True).collect()
    assert df.shape == (70, 3)
    assert round(df.sort("ModelID").row(0)[2], 2) == 55.46
    plot = sample.plot.over_time()
    assert plot is not None


def test_proposition_success_rates(sample: ADMDatamart):
    df = sample.plot.proposition_success_rates(return_df=True)

    assert round(df.select(pl.col("SuccessRate").top_k(1)).collect().item(), 3) == 0.295
    assert df.select(pl.col("Name").top_k(1)).collect().item() == "VisaGold"

    plot = sample.plot.proposition_success_rates()
    assert plot is not None


def test_score_distribution(sample: ADMDatamart):
    model_id = sample.combined_data.select("ModelID").collect().row(0)[0]
    df = sample.plot.score_distribution(model_id=model_id, return_df=True)

    assert df.select(pl.col("PredictorName").top_k(1)).collect().item() == "Classifier"
    plot = sample.plot.score_distribution(model_id=model_id)
    assert plot is not None


def test_multiple_score_distributions(sample: ADMDatamart):
    plot = sample.plot.multiple_score_distributions(show_all=False)
    assert len(plot) == 20
    assert all(plot is not None for plot in plot)


def test_predictor_binning(sample: ADMDatamart):
    random_row = (
        sample.combined_data.select(["ModelID", "PredictorName"]).collect().sample(1)
    )
    model_id = random_row["ModelID"][0]
    predictor_name = random_row["PredictorName"][0]
    df = sample.plot.predictor_binning(
        model_id=model_id, predictor_name=predictor_name, return_df=True
    )
    assert not df.collect().is_empty()
    required_columns = ["BinIndex", "BinPropensity", "BinSymbol", "BinResponseCount"]
    assert all(col in df.collect_schema().names() for col in required_columns)
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
    assert plot is not None


def test_multiple_predictor_binning(sample: ADMDatamart):
    model_id = sample.combined_data.select("ModelID").collect().row(0)[0]
    plots = sample.plot.multiple_predictor_binning(model_id=model_id, show_all=False)
    assert isinstance(plots, list)
    # Same number of plots as number of predictors
    assert (
        len(plots)
        == sample.combined_data.filter(pl.col("ModelID") == model_id)
        .select("PredictorName")
        .unique()
        .collect()
        .shape[0]
    )
    assert all(isinstance(plot, Figure) for plot in plots)


def test_predictor_performance(sample: ADMDatamart):
    df = sample.plot.predictor_performance(return_df=True)
    assert "PredictorName" in df.collect_schema().names()
    assert "PredictorPerformance" in df.collect_schema().names()
    assert (
        round(df.select(pl.col("PredictorPerformance").top_k(1)).collect().item(), 2)
        == 0.86
    )

    plot = sample.plot.predictor_performance()
    assert isinstance(plot, Figure)


def test_predictor_category_performance(sample: ADMDatamart):
    df = sample.plot.predictor_category_performance(return_df=True).collect()
    assert df.shape == (60, 3)
    assert "PredictorCategory" in df.columns
    assert "PredictorPerformance" in df.columns
    assert round(df.select(pl.col("PredictorPerformance").top_k(1)).item(), 2) == 0.62

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
            df.filter(pl.col("Name") == "AutoUsed60Months")
            .select("Customer.AnnualIncome")
            .collect()
            .item(),
            2,
        )
        == 0.69
    )
    assert df.select(pl.col("*").exclude("Name")).collect().to_numpy().max() <= 1.0

    plot = sample.plot.predictor_performance_heatmap()
    assert isinstance(plot, Figure)


def test_models_by_positives(sample: ADMDatamart):
    df = sample.plot.models_by_positives(return_df=True)
    assert df.shape == (31, 5)
    assert (
        round(
            df.filter(pl.col("PositivesBin") == "(0, 10]")
            .filter(pl.col("Channel") == "Email")
            .select("cumModels")
            .item(),
            2,
        )
        == 0.23
    )
    assert df.select(pl.col("cumModels").max()).item() <= 1.0

    plot = sample.plot.models_by_positives()
    assert isinstance(plot, Figure)


def test_tree_map(sample: ADMDatamart):
    df = sample.plot.tree_map(return_df=True).collect()
    assert df.shape == (68, 10)
    assert df.sort("Total number of positives").row(0)[-1] == 2.0
    plot = sample.plot.tree_map()
    assert isinstance(plot, Figure)


def test_predictor_count(sample: ADMDatamart):
    df = sample.plot.predictor_count(return_df=True).collect()
    assert df.shape == (78, 5)
    assert df.row(0)[4] == 44

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


def test_propensity_distribution(sample: ADMDatamart):
    df = sample.plot.propensity_distribution(return_df=True).collect()
    assert df.shape == (2612, 4)
    expected_columns = ["BinPropensity", "Channel", "Direction", "BinResponseCount"]
    assert df.columns == expected_columns
    assert df["BinPropensity"].min() >= 0
    assert df["BinPropensity"].max() <= 1

    plot = sample.plot.predictor_count()
    assert isinstance(plot, Figure)
