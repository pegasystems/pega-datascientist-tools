"""
Testing the functionality of the Prediction class
"""

import datetime
import os
from unittest.mock import patch, MagicMock

import polars as pl
import pytest
from pdstools import Prediction
from pdstools.utils import cdh_utils

mock_prediction_data = pl.DataFrame(
    {
        "pySnapShotTime": cdh_utils.to_prpc_date_time(datetime.datetime(2040, 4, 1))[
            0:15
        ],  # Polars doesn't like time zones like GMT+0200
        "pyModelId": ["DATA-DECISION-REQUEST-CUSTOMER!MYCUSTOMPREDICTION"] * 4
        + ["DATA-DECISION-REQUEST-CUSTOMER!PredictActionPropensity"] * 4
        + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTMOBILEPROPENSITY"] * 4
        + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"] * 4,
        "pyModelType": "PREDICTION",
        "pySnapshotType": (["Daily"] * 3 + [None]) * 4,
        "pyDataUsage": ["Control", "Test", "NBA", ""] * 4,
        "pyPositives": [100, 400, 500, 1000, 200, 800, 1000, 2000] * 2,
        "pyNegatives": [1000, 2000, 3000, 6000, 3000, 6000, 9000, 18000] * 2,
        "pyCount": [1100, 2400, 3500, 7000, 3200, 6800, 10000, 20000] * 2,
        "pyValue": ([0.65] * 4 + [0.70] * 4) * 2,
    }
).lazy()


@pytest.fixture
def preds_singleday():
    """Fixture to serve as class to call functions from."""
    return Prediction(mock_prediction_data)


@pytest.fixture
def preds_fewdays():
    return Prediction(
        pl.concat(
            [
                mock_prediction_data.with_columns(
                    pySnapShotTime=pl.lit(cdh_utils.to_prpc_date_time(datetime.datetime(2040, 5, 1))[0:15])
                ),
                mock_prediction_data.with_columns(
                    pySnapShotTime=pl.lit(
                        cdh_utils.to_prpc_date_time(
                            datetime.datetime(2040, 5, 16)
                        )[0:15]
                    )
                ),
            ],
            how="vertical",
        )
    )


def test_available(preds_singleday):
    print(preds_singleday.predictions.collect())
    assert preds_singleday.is_available
    assert preds_singleday.is_valid


def test_summary_by_channel_cols(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary.columns == [
        "Prediction",
        "Channel",
        "Direction",
        "usesNBAD",
        "isMultiChannel",
        'DateRange Min',
        'DateRange Max',
        'Duration',
        "Performance",
        "Positives",
        "Negatives",
        "Responses",
        "Positives_Test",
        "Positives_Control",
        "Positives_NBA",
        "Negatives_Test",
        "Negatives_Control",
        "Negatives_NBA",
        "usesImpactAnalyzer",
        "ControlPercentage",
        "TestPercentage",
        "CTR",
        "CTR_Test",
        "CTR_Control",
        "CTR_NBA",
        "ChannelDirectionGroup",
        "isValid",
        "Lift",
    ]


def test_summary_by_channel_channels(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary.select(pl.len()).item() == 4


def test_summary_by_channel_validity(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary["isValid"].to_list() == [True, True, True, True]

def test_summary_by_channel_trend(preds_singleday):
    summary = preds_singleday.summary_by_channel(by_period="1d").collect()
    assert summary.select(pl.len()).item() == 4


def test_summary_by_channel_trend2(preds_fewdays):
    summary = preds_fewdays.summary_by_channel(by_period="1d").collect()
    assert summary.select(pl.len()).item() == 8



def test_summary_by_channel_ia(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary["usesImpactAnalyzer"].to_list() == [True, True, True, True]

    preds_singleday = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (
                pl.col("pyModelId")
                == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"
            )
        )
    )
    # only Web still has the NBA indicator
    assert preds_singleday.summary_by_channel().collect()["usesImpactAnalyzer"].to_list() == [
        False,
        False,
        False,
        True,
    ]


def test_summary_by_channel_lift(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert [round(x, 5) for x in summary["Lift"].to_list()] == [0.83333, 0.88235] * 2


def test_summary_by_channel_controlpct(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert [round(x, 5) for x in summary["ControlPercentage"].to_list()] == [
        15.71429,
        16.0,
    ] * 2
    assert [round(x, 5) for x in summary["TestPercentage"].to_list()] == [
        34.28571,
        34.0,
    ] * 2


def test_summary_by_channel_range(preds_fewdays):
    summary = preds_fewdays.summary_by_channel().collect()
    assert summary['DateRange Min'].to_list() == [datetime.date(2040,5,1)]*4
    assert summary['Positives'].to_list() == [2000,4000,2000,4000]

    summary = preds_fewdays.summary_by_channel(start_date=datetime.date(2040, 5, 15)).collect()
    assert summary['DateRange Min'].to_list() == [datetime.date(2040,5,16)]*4
    assert summary['Positives'].to_list() == [1000,2000,1000,2000]


def test_summary_by_channel_channeldirectiongroup(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()

    assert summary["isMultiChannel"].to_list() == [False, True, False, False]
    assert summary["usesNBAD"].to_list() == [False, True, True, True]
    assert summary["ChannelDirectionGroup"].to_list() == [
        "Other",
        "Other",
        "Mobile/Inbound",
        "Web/Inbound",
    ]


def test_overall_summary_cols(preds_singleday):
    summary = preds_singleday.overall_summary().collect()
    assert summary.columns == [
        'DateRange Min',
        'DateRange Max',
        'Duration',
        "Number of Valid Channels",
        "Overall Lift",
        "Performance",
        "Positives Inbound",
        "Positives Outbound",
        "Responses Inbound",
        "Responses Outbound",
        "Channel with Minimum Negative Lift",
        "Minimum Negative Lift",
        "usesImpactAnalyzer",
        "ControlPercentage",
        "TestPercentage",
        "usesNBAD",
    ]
    assert len(summary) == 1


def test_overall_summary_n_valid_channels(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Number of Valid Channels"].item() == 3


def test_overall_summary_overall_lift(preds_singleday):
    # print(test.overall_summary().collect())
    # print(test.summary_by_channel().collect())
    assert round(preds_singleday.overall_summary().collect()["Overall Lift"].item(), 5) == 0.86217


def test_overall_summary_positives(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Positives Inbound"].item() == 3000
    assert preds_singleday.overall_summary().collect()["Positives Outbound"].item() == 0 # some channels unknown/multi-channel


def test_overall_summary_responsecount(preds_singleday):
    print (preds_singleday.summary_by_channel().select(['Channel','Direction','Responses']).collect())
    assert preds_singleday.overall_summary().collect()["Responses Inbound"].item() == 27000
    assert preds_singleday.overall_summary().collect()["Responses Outbound"].item() == 0 

def test_overall_summary_channel_min_lift(preds_singleday):
    assert (
        preds_singleday.overall_summary().collect()["Channel with Minimum Negative Lift"].item()
        is None
    )

def test_overall_summary_by_period(preds_fewdays):
    summ = preds_fewdays.overall_summary(by_period="1d").collect()
    assert summ.height == 2

def test_overall_summary_min_lift(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Minimum Negative Lift"].item() is None


def test_overall_summary_controlpct(preds_singleday):
    assert (
        round(preds_singleday.overall_summary().collect()["ControlPercentage"].item(), 5)
        == 15.88235
    )
    assert (
        round(preds_singleday.overall_summary().collect()["TestPercentage"].item(), 5) == 34.11765
    )


def test_overall_summary_ia(preds_singleday):
    assert preds_singleday.overall_summary().collect().select(pl.col("usesImpactAnalyzer")).item()

    preds_singleday = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (
                pl.col("pyModelId")
                == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"
            )
        )
    )
    assert preds_singleday.overall_summary().collect()["usesImpactAnalyzer"].to_list() == [True]


def test_plots():
    prediction = Prediction.from_mock_data()

    assert prediction.plot.performance_trend() is not None
    assert prediction.plot.lift_trend() is not None
    assert prediction.plot.responsecount_trend() is not None
    assert prediction.plot.ctr_trend() is not None

    assert prediction.plot.performance_trend("1w") is not None
    assert isinstance(prediction.plot.lift_trend("2d", return_df=True), pl.LazyFrame)
    assert prediction.plot.responsecount_trend("1m") is not None
    assert prediction.plot.ctr_trend("5d") is not None

    assert isinstance(prediction.plot.performance_trend(return_df=True), pl.LazyFrame)
    assert isinstance(prediction.plot.lift_trend(return_df=True), pl.LazyFrame)
    assert isinstance(prediction.plot.responsecount_trend(return_df=True), pl.LazyFrame)
    assert isinstance(prediction.plot.ctr_trend(return_df=True), pl.LazyFrame)


# New tests to improve coverage

def test_from_mock_data():
    """Test the from_mock_data class method with different day parameters."""
    # Test with default days
    pred_default = Prediction.from_mock_data()
    assert pred_default.is_available
    assert pred_default.is_valid
    
    # Test with custom days
    pred_custom = Prediction.from_mock_data(days=30)
    assert pred_custom.is_available
    assert pred_custom.is_valid
    
    # Verify the number of days in the data
    unique_dates = pred_custom.predictions.select(
        pl.col("SnapshotTime").unique()
    ).collect()
    assert len(unique_dates) == 30


def test_from_ds_export():
    """Test the from_ds_export class method."""
    # Create a simple mock for testing
    mock_data = mock_prediction_data
    
    # Create a mock for the read_ds_export function
    # We need to patch where it's imported in the Prediction module
    with patch('pdstools.prediction.Prediction.read_ds_export', return_value=mock_data) as mock_read:
        # Create a test instance with our mock
        pred = Prediction.from_ds_export('test_predictions.zip')
        
        # Verify the function was called with the correct arguments
        mock_read.assert_called_once()
        args, kwargs = mock_read.call_args
        assert args[0] == 'test_predictions.zip'
        assert args[1] == '.'
        
        # Reset the mock for the next test
        mock_read.reset_mock()
        
        # Test with base_path
        pred = Prediction.from_ds_export('test_predictions.zip', '/path/to/data')
        mock_read.assert_called_once()
        args, kwargs = mock_read.call_args
        assert args[0] == 'test_predictions.zip'
        assert args[1] == '/path/to/data'
        
        # Reset the mock for the next test
        mock_read.reset_mock()
        
        # Test with query
        # For this test, we need to patch the __init__ method to check the query parameter
        with patch.object(Prediction, '__init__', return_value=None) as mock_init:
            Prediction.from_ds_export('test_predictions.zip', query={"Class": ["TEST"]})
            mock_init.assert_called_once()
            assert mock_init.call_args[1].get('query') == {"Class": ["TEST"]}


def test_from_pdc():
    """Test the from_pdc class method."""
    # Create mock PDC data with all required columns
    pdc_data = pl.DataFrame({
        "ModelClass": ["DATA-DECISION-REQUEST-CUSTOMER"] * 12,
        "ModelName": ["MYCUSTOMPREDICTION"] * 4 + ["PREDICTMOBILEPROPENSITY"] * 4 + ["PREDICTWEBPROPENSITY"] * 4,
        "ModelID": ["ID1"] * 12,  # Added missing required column
        "ModelType": ["Prediction_Test", "Prediction_Control", "Prediction_NBA", "Prediction"] * 3,
        "Name": ["auc"] * 12,
        "SnapshotTime": [datetime.datetime(2040, 4, 1)] * 12,
        "Performance": [65.0] * 4 + [70.0] * 8,
        "Positives": [400, 100, 500, 1000, 800, 200, 1000, 2000] * 1 + [400, 100, 500, 1000],
        "Negatives": [2000, 1000, 3000, 6000, 6000, 3000, 9000, 18000] * 1 + [2000, 1000, 3000, 6000],
        "ResponseCount": [2400, 1100, 3500, 7000, 6800, 3200, 10000, 20000] * 1 + [2400, 1100, 3500, 7000],
        "ADMModelType": [""] * 12,
        "TotalPositives": [0] * 12,
        "TotalResponses": [0] * 12,
    }).lazy()
    
    # We need to patch the _read_pdc function to avoid actual processing
    with patch('pdstools.utils.cdh_utils._read_pdc', return_value=pdc_data) as mock_read_pdc:
        # Test with return_df=True
        result = Prediction.from_pdc(pdc_data, return_df=True)
        assert isinstance(result, pl.LazyFrame)
        
        # For testing initialization and query parameters, we need to patch the __init__ method
        with patch.object(Prediction, '__init__', return_value=None) as mock_init:
            # Test normal initialization
            Prediction.from_pdc(pdc_data)
            mock_init.assert_called_once()
            
            # Reset the mock for the next test
            mock_init.reset_mock()
            
            # Test with query
            Prediction.from_pdc(pdc_data, query={"ModelName": ["PREDICTWEBPROPENSITY"]})
            mock_init.assert_called_once()
            assert mock_init.call_args[1].get('query') == {"ModelName": ["PREDICTWEBPROPENSITY"]}


def test_prediction_plots_internal_method(preds_singleday):
    """Test the internal _prediction_trend method of PredictionPlots."""
    # Call the internal method directly
    plt, plot_df = preds_singleday.plot._prediction_trend(
        period="1d",
        query=None,
        metric="Performance",
        title="Test Plot"
    )
    
    # Verify the plot was created
    assert plt is not None
    
    # Verify the dataframe has expected columns
    assert isinstance(plot_df, pl.LazyFrame)
    collected_df = plot_df.collect()
    assert "Performance" in collected_df.columns
    assert "Date" in collected_df.columns
    assert "Prediction" in collected_df.columns


def test_prediction_validity_expr(preds_singleday):
    """Test the prediction_validity_expr class attribute."""
    # Get the expression
    expr = Prediction.prediction_validity_expr
    
    # Test the expression on the predictions property which has the correct column names
    result = preds_singleday.predictions.filter(expr).collect()
    
    # Verify it filters as expected
    assert len(result) > 0
    
    # Create a prediction with invalid data (with zeros)
    invalid_data = mock_prediction_data.with_columns(
        pyPositives=pl.lit(0)
    )
    invalid_pred = Prediction(invalid_data)
    
    # Verify it correctly identifies invalid data
    invalid_result = invalid_pred.predictions.filter(expr).collect()
    assert len(invalid_result) == 0


def test_init_with_temporal_snapshot_time():
    """Test initialization with already parsed temporal snapshot time."""
    # Create data with datetime column
    data = mock_prediction_data.with_columns(
        pySnapShotTime=pl.lit(datetime.datetime(2040, 4, 1)).cast(pl.Datetime)
    )
    
    # Initialize prediction
    pred = Prediction(data)
    
    # Verify it was processed correctly
    assert pred.is_available
    assert pred.is_valid
    
    # Check the SnapshotTime column is a date
    schema = pred.predictions.collect_schema()
    assert schema["SnapshotTime"].is_temporal()


def test_lazy_namespace_initialization():
    """Test the LazyNamespace initialization in PredictionPlots."""
    pred = Prediction.from_mock_data()
    
    # Access the plot namespace to trigger initialization
    assert pred.plot is not None
    assert hasattr(pred.plot, 'prediction')
    assert pred.plot.prediction is pred
    
    # Verify the dependencies attribute
    assert hasattr(pred.plot, 'dependencies')
    assert 'plotly' in pred.plot.dependencies
