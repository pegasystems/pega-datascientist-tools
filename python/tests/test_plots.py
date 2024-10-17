import datetime
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
    assert round(df.sort("ModelID").row(0)[2], 2) == 0.55
    plot = sample.plot.over_time()
    assert plot is not None


def test_proposition_success_rates(sample: ADMDatamart):
    df = sample.plot.proposition_success_rates(return_df=True)

    assert round(df.select(pl.col("SuccessRate").top_k(1)).collect().item(), 3) == 0.295
    assert df.select(pl.col("Name").top_k(1)).collect().item() == "VisaGold"

    plot = sample.plot.proposition_success_rates()
    assert plot is not None


def test_score_distribution(sample: ADMDatamart):
    model_id = "bd70a915-697a-5d43-ab2c-53b0557c85a0"
    df = sample.plot.score_distribution(model_id=model_id, return_df=True)

    assert df.select(pl.col("BinIndex").top_k(1)).collect().item() == 1
    assert (
        round(df.select(pl.col("BinPropensity").top_k(1)).collect().item(), 2) == 0.01
    )

    # Test if the plot works
    plot = sample.plot.score_distribution(model_id=model_id)
    assert plot is not None


def test_multiple_score_distributions(sample: ADMDatamart):
    plot = sample.plot.multiple_score_distributions(show_all=False)
    assert len(plot) == 20
    assert all(plot is not None for plot in plot)


def test_predictor_binning(sample: ADMDatamart):
    model_id = "bd70a915-697a-5d43-ab2c-53b0557c85a0"
    predictor_name = "Customer.HealthMatter"
    df = sample.plot.predictor_binning(
        model_id=model_id, predictor_name=predictor_name, return_df=True
    )

    assert df.select(pl.col("BinIndex").top_k(1)).collect().item() == 1
    assert (
        round(df.select(pl.col("BinPropensity").top_k(1)).collect().item(), 5)
        == 0.00206
    )

    plot = sample.plot.predictor_binning(
        model_id=model_id, predictor_name=predictor_name
    )
    assert plot is not None


def test_multiple_predictor_binning(sample: ADMDatamart):
    model_id = "bd70a915-697a-5d43-ab2c-53b0557c85a0"
    plots = sample.plot.multiple_predictor_binning(model_id=model_id, show_all=False)
    assert isinstance(plots, list)
    assert len(plots) == 90
    assert all(isinstance(plot, Figure) for plot in plots)


def test_predictor_performance(sample: ADMDatamart):
    df = sample.plot.predictor_performance(return_df=True)
    assert "PredictorName" in df.columns
    assert "PerformanceBin" in df.columns
    assert (
        round(df.select(pl.col("PerformanceBin").top_k(1)).collect().item(), 2) == 0.86
    )

    plot = sample.plot.predictor_performance()
    assert isinstance(plot, Figure)


def test_predictor_category_performance(sample: ADMDatamart):
    df = sample.plot.predictor_category_performance(return_df=True).collect()
    assert df.shape == (60, 3)
    assert "PredictorCategory" in df.columns
    assert "PerformanceBin" in df.columns
    assert round(df.select(pl.col("PerformanceBin").top_k(1)).item(), 2) == 0.62

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
    model_id = "bd70a915-697a-5d43-ab2c-53b0557c85a0"
    predictor_name = "Customer.HealthMatter"
    df = sample.plot.binning_lift(model_id, predictor_name, return_df=True).collect()
    assert df.shape == (1, 8)
    assert df.filter(pl.col("BinIndex") == 1).select("Lift").item() == 0.0

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
