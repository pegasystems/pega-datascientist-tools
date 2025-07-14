"""
Additional tests for the Plots module to increase test coverage
"""

import plotly.express as px
import polars as pl
import pytest
from pdstools import ADMDatamart
from plotly.graph_objs import Figure


@pytest.fixture
def sample_datamart(monkeypatch):
    """Create a simple datamart for testing with mocked validation"""
    # Mock the _validate_predictor_data method to bypass schema validation
    monkeypatch.setattr(ADMDatamart, "_validate_predictor_data", lambda self, df: df)
    monkeypatch.setattr(ADMDatamart, "_validate_model_data", lambda self, df, **kwargs: df)
    
    model_data = pl.DataFrame({
        "ModelID": ["model1", "model2", "model3", "model4"],
        "Name": ["Action1", "Action2", "Action1", "Action3"],
        "Channel": ["Web", "Mobile", "Web", "Email"],
        "Direction": ["Inbound", "Inbound", "Inbound", "Outbound"],
        "Performance": [0.75, 0.80, 0.72, 0.65],
        "SuccessRate": [0.25, 0.30, 0.22, 0.18],
        "ResponseCount": [1000, 1200, 800, 500],
        "Positives": [250, 360, 176, 90],
        "SnapshotTime": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]
    }).lazy()
    
    predictor_data = pl.DataFrame({
        "ModelID": ["model1", "model1", "model2", "model2"],
        "PredictorName": ["Age", "Income", "Age", "Income"],
        "BinIndex": [1, 2, 1, 2],
        "BinSymbol": ["<30", ">=30", "<30", ">=30"],
        "BinPositives": [100, 150, 180, 180],
        "BinNegatives": [400, 350, 420, 420],
        "BinResponseCount": [500, 500, 600, 600],
        "Lift": [0.5, 0.8, 0.6, 0.9],
        "EntryType": ["Active", "Active", "Active", "Active"],
        "Type": ["Numeric", "Numeric", "Numeric", "Numeric"],
        "Performance": [0.7, 0.65, 0.72, 0.68],
        "SnapshotTime": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"]
    }).lazy()
    
    # Create a datamart with the test data
    dm = ADMDatamart(model_df=model_data, predictor_df=predictor_data)
    
    # Mock the combined_data property
    combined_data = predictor_data.with_columns(
        pl.lit("Classifier").alias("PredictorName"),
        pl.lit(0.5).alias("BinPropensity")
    )
    dm.combined_data = combined_data
    
    # Set context keys
    dm.context_keys = ["Name", "Channel", "Direction"]
    
    return dm


def test_action_overlap(sample_datamart):
    """Test the action_overlap method"""
    # Test with default parameters
    fig = sample_datamart.plot.action_overlap()
    assert isinstance(fig, Figure)
    
    # Test with return_df=True
    df = sample_datamart.plot.action_overlap(return_df=True)
    assert isinstance(df, pl.DataFrame)
    assert "Channel" in df.columns
    assert df.shape[0] == 3  # 3 channels
    
    # Test with custom group_col
    fig = sample_datamart.plot.action_overlap(group_col="Direction")
    assert isinstance(fig, Figure)
    
    # Test with list of group columns
    fig = sample_datamart.plot.action_overlap(group_col=["Channel", "Direction"])
    assert isinstance(fig, Figure)
    
    # Test with expression
    fig = sample_datamart.plot.action_overlap(
        group_col=pl.concat_str(pl.col("Channel"), pl.col("Direction"), separator="/")
    )
    assert isinstance(fig, Figure)


def test_partitioned_plot(sample_datamart):
    """Test the partitioned_plot method"""
    # Define a simple plot function for testing
    def dummy_plot_func(return_df=False, query=None):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        if return_df:
            return df.lazy()
        return px.scatter(df, x="x", y="y")
    
    # Test with a set of facets
    facets = {"Web", "Mobile", "Email"}
    plots = sample_datamart.plot.partitioned_plot(
        dummy_plot_func, facets, partition_col="Channel", show_plots=False
    )
    
    # Check that we got the expected number of plots
    assert len(plots) == len(facets)
    
    # Check that all plots are Figures
    assert all(isinstance(plot, Figure) for plot in plots)


def test_binning_lift_enhanced(sample_datamart):
    """Enhanced test for the binning_lift method"""
    # Get a model_id and predictor_name from the sample data
    model_id = "model1"
    predictor_name = "Age"
    
    # Test with default parameters
    fig = sample_datamart.plot.binning_lift(model_id, predictor_name)
    assert isinstance(fig, Figure)
    
    # Check that the figure has the expected traces
    assert len(fig.data) > 0
    
    # Test with return_df=True
    df = sample_datamart.plot.binning_lift(model_id, predictor_name, return_df=True)
    assert isinstance(df, pl.LazyFrame)
    
    # Check that the dataframe has the expected columns
    collected_df = df.collect()
    assert "PredictorName" in collected_df.columns
    assert "BinIndex" in collected_df.columns
    assert "BinSymbol" in collected_df.columns
    assert "Lift" in collected_df.columns
    assert "Direction" in collected_df.columns
    
    # Test with query
    fig = sample_datamart.plot.binning_lift(
        model_id, predictor_name, query=pl.col("BinIndex") == 1
    )
    assert isinstance(fig, Figure)
    
    # Skip validation tests since our mock datamart doesn't validate model_ids or predictor_names


def test_requires_decorator(monkeypatch):
    """Test the requires decorator by calling methods with missing data"""
    # Mock the _validate_predictor_data method to bypass schema validation
    monkeypatch.setattr(ADMDatamart, "_validate_predictor_data", lambda self, df: df)
    monkeypatch.setattr(ADMDatamart, "_validate_model_data", lambda self, df, **kwargs: df)
    
    # Create a datamart with missing data
    empty_datamart = ADMDatamart(model_df=None, predictor_df=None)
    
    # Test a method that requires model_data
    with pytest.raises(ValueError, match="Missing data: model_data"):
        empty_datamart.plot.bubble_chart()
    
    # Test a method that requires combined_data
    with pytest.raises(ValueError, match="Missing data: combined_data"):
        empty_datamart.plot.predictor_performance()
    
    # Test another method that requires combined_data
    with pytest.raises(ValueError, match="Missing data: combined_data"):
        empty_datamart.plot.score_distribution(model_id="model1")
