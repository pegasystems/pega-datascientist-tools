"""Testing the functionality of the Prediction class"""

import datetime
import shutil
from unittest.mock import patch

import polars as pl
import pytest
from pdstools import Prediction
from pdstools.pega_io.File import read_ds_export
from pdstools.prediction.Plots import PredictionPlots
from pdstools.utils import cdh_utils
from plotly.graph_objs import Figure

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
    },
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
                    pySnapShotTime=pl.lit(
                        cdh_utils.to_prpc_date_time(datetime.datetime(2040, 5, 1))[0:15],
                    ),
                ),
                mock_prediction_data.with_columns(
                    pySnapShotTime=pl.lit(
                        cdh_utils.to_prpc_date_time(datetime.datetime(2040, 5, 16))[0:15],
                    ),
                ),
            ],
            how="vertical",
        ),
    )


def test_available(preds_singleday):
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
        "DateRange Min",
        "DateRange Max",
        "Duration",
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
    summary = preds_singleday.summary_by_channel(every="1d").collect()
    assert summary.select(pl.len()).item() == 4


def test_summary_by_channel_trend2(preds_fewdays):
    summary = preds_fewdays.summary_by_channel(every="1d").collect()
    assert summary.select(pl.len()).item() == 8


def test_summary_by_channel_ia(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary["usesImpactAnalyzer"].to_list() == [True, True, True, True]

    preds_singleday = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (pl.col("pyModelId") == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"),
        ),
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
    assert summary["DateRange Min"].to_list() == [datetime.date(2040, 5, 1)] * 4
    assert summary["Positives"].to_list() == [2000, 4000, 2000, 4000]

    summary = preds_fewdays.summary_by_channel(
        start_date=datetime.date(2040, 5, 15),
    ).collect()
    assert summary["DateRange Min"].to_list() == [datetime.date(2040, 5, 16)] * 4
    assert summary["Positives"].to_list() == [1000, 2000, 1000, 2000]


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
        "DateRange Min",
        "DateRange Max",
        "Duration",
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

    pred_data = pl.DataFrame(
        {
            "pySnapShotTime": cdh_utils.to_prpc_date_time(
                datetime.datetime(2040, 4, 1),
            )[0:15],  # Polars doesn't like time zones like GMT+0200
            "pyModelId": ["DATA-DECISION-REQUEST-CUSTOMER!MYCUSTOMPREDICTION"] * 4
            + ["DATA-DECISION-REQUEST-CUSTOMER!PredictActionPropensity"] * 4
            + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTMOBILEPROPENSITY"] * 4
            + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"] * 4,
            "pyModelType": "PREDICTION",
            "pySnapshotType": (["Daily"] * 3 + [None]) * 4,
            "pyDataUsage": ["Control", "Test", "NBA", ""] * 4,
            "pyPositives": [0, 0, 0, 0, 100, 200, 300, 400] * 2,
            "pyNegatives": [1000, 2000, 3000, 6000, 3000, 6000, 9000, 18000] * 2,
            "pyCount": [1100, 2400, 3500, 7000, 3200, 6800, 10000, 20000] * 2,
            "pyValue": ([0.65] * 4 + [0.70] * 4) * 2,
        },
    ).lazy()
    p = Prediction(df=pred_data)
    assert p.overall_summary().collect()["Number of Valid Channels"].item() == 1

    pred_data = pl.DataFrame(
        {
            "pySnapShotTime": cdh_utils.to_prpc_date_time(
                datetime.datetime(2040, 4, 1),
            )[0:15],  # Polars doesn't like time zones like GMT+0200
            "pyModelId": ["DATA-DECISION-REQUEST-CUSTOMER!MYCUSTOMPREDICTION"] * 4
            + ["DATA-DECISION-REQUEST-CUSTOMER!PredictActionPropensity"] * 4
            + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTMOBILEPROPENSITY"] * 4
            + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"] * 4,
            "pyModelType": "PREDICTION",
            "pySnapshotType": (["Daily"] * 3 + [None]) * 4,
            "pyDataUsage": ["Control", "Test", "NBA", ""] * 4,
            "pyPositives": [0, 0, 0, 0, 0, 0, 0, 0] * 2,
            "pyNegatives": [1000, 2000, 3000, 6000, 3000, 6000, 9000, 18000] * 2,
            "pyCount": [1100, 2400, 3500, 7000, 3200, 6800, 10000, 20000] * 2,
            "pyValue": ([0.65] * 4 + [0.70] * 4) * 2,
        },
    ).lazy()
    p = Prediction(df=pred_data)
    # BUG-956453 previously this gave an empty dataframe
    assert p.overall_summary().collect()["Number of Valid Channels"].item() == 0


def test_overall_summary_overall_lift(preds_singleday):
    # print(test.overall_summary().collect())
    # print(test.summary_by_channel().collect())
    assert round(preds_singleday.overall_summary().collect()["Overall Lift"].item(), 5) == 0.86217


def test_overall_summary_positives(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Positives Inbound"].item() == 3000
    assert (
        preds_singleday.overall_summary().collect()["Positives Outbound"].item() == 0
    )  # some channels unknown/multi-channel


def test_overall_summary_responsecount(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Responses Inbound"].item() == 27000
    assert preds_singleday.overall_summary().collect()["Responses Outbound"].item() == 0


def test_overall_summary_channel_min_lift(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Channel with Minimum Negative Lift"].item() is None


def test_overall_summary_by_period(preds_fewdays):
    summ = preds_fewdays.overall_summary(every="1d").collect()
    assert summ.height == 2


def test_overall_summary_min_lift(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Minimum Negative Lift"].item() is None


def test_overall_summary_controlpct(preds_singleday):
    assert (
        round(
            preds_singleday.overall_summary().collect()["ControlPercentage"].item(),
            5,
        )
        == 15.88235
    )
    assert round(preds_singleday.overall_summary().collect()["TestPercentage"].item(), 5) == 34.11765


def test_overall_summary_ia(preds_singleday):
    assert preds_singleday.overall_summary().collect().select(pl.col("usesImpactAnalyzer")).item()

    preds_singleday = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (pl.col("pyModelId") == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"),
        ),
    )
    assert preds_singleday.overall_summary().collect()["usesImpactAnalyzer"].to_list() == [True]


def test_plots():
    prediction = Prediction.from_mock_data()  # 70 days * 3 channels = 210 rows

    # Each figure exposes one trace per channel (Mobile, E-mail, Web).
    expected_trace_names = sorted(
        [
            "Mobile (PREDICTMOBILEPROPENSITY)",
            "E-mail (PREDICTOUTBOUNDEMAILPROPENSITY)",
            "Web (PREDICTWEBPROPENSITY)",
        ]
    )

    perf_fig = prediction.plot.performance_trend()
    assert sorted(t.name for t in perf_fig.data) == expected_trace_names
    assert perf_fig.layout.yaxis.title.text == "Performance (AUC)"
    assert perf_fig.layout.xaxis.title.text == "Date"

    lift_fig = prediction.plot.lift_trend()
    assert sorted(t.name for t in lift_fig.data) == expected_trace_names
    assert lift_fig.layout.yaxis.title.text == "Engagement Lift"

    rc_fig = prediction.plot.responsecount_trend()
    assert sorted(t.name for t in rc_fig.data) == expected_trace_names
    assert rc_fig.layout.yaxis.title.text == "Responses"

    ctr_fig = prediction.plot.ctr_trend()
    assert sorted(t.name for t in ctr_fig.data) == expected_trace_names
    assert ctr_fig.layout.yaxis.title.text == "CTR"

    # Period overrides change the row count of the underlying frame.
    perf_w = prediction.plot.performance_trend("1w", return_df=True).collect()
    assert perf_w.shape == (33, 29)

    lift_2d = prediction.plot.lift_trend("2d", return_df=True).collect()
    assert lift_2d.shape == (105, 29)

    rc_1m = prediction.plot.responsecount_trend("1m", return_df=True).collect()
    assert rc_1m.shape == (210, 29)

    ctr_5d = prediction.plot.ctr_trend("5d", return_df=True).collect()
    assert ctr_5d.shape == (45, 29)

    # Default period (1d): 70 days * 3 channels = 210 rows.
    for method in (
        prediction.plot.performance_trend,
        prediction.plot.lift_trend,
        prediction.plot.responsecount_trend,
        prediction.plot.ctr_trend,
    ):
        df = method(return_df=True).collect()
        assert df.shape == (210, 29)
        assert sorted(df["Channel"].unique().to_list()) == ["E-mail", "Mobile", "Web"]


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
        pl.col("SnapshotTime").unique(),
    ).collect()
    assert len(unique_dates) == 30


def test_from_pdc():
    """Test the from_pdc class method."""
    # Create mock PDC data with all required columns
    pdc_data = pl.DataFrame(
        {
            "ModelClass": ["DATA-DECISION-REQUEST-CUSTOMER"] * 12,
            "ModelName": ["MYCUSTOMPREDICTION"] * 4 + ["PREDICTMOBILEPROPENSITY"] * 4 + ["PREDICTWEBPROPENSITY"] * 4,
            "ModelID": ["ID1"] * 12,  # Added missing required column
            "ModelType": [
                "Prediction_Test",
                "Prediction_Control",
                "Prediction_NBA",
                "Prediction",
            ]
            * 3,
            "Name": ["auc"] * 12,
            "SnapshotTime": [datetime.datetime(2040, 4, 1)] * 12,
            "Performance": [65.0] * 4 + [70.0] * 8,
            "Positives": [400, 100, 500, 1000, 800, 200, 1000, 2000] * 1 + [400, 100, 500, 1000],
            "Negatives": [2000, 1000, 3000, 6000, 6000, 3000, 9000, 18000] * 1 + [2000, 1000, 3000, 6000],
            "ResponseCount": [2400, 1100, 3500, 7000, 6800, 3200, 10000, 20000] * 1 + [2400, 1100, 3500, 7000],
            "ADMModelType": [""] * 12,
            "TotalPositives": [0] * 12,
            "TotalResponses": [0] * 12,
        },
    ).lazy()

    # We need to patch the _read_pdc function to avoid actual processing
    with patch("pdstools.utils.cdh_utils._read_pdc", return_value=pdc_data):
        # Test with return_df=True
        result = Prediction.from_pdc(pdc_data, return_df=True)
        assert isinstance(result, pl.LazyFrame)

        # For testing initialization and query parameters, we need to patch the __init__ method
        with patch.object(Prediction, "__init__", return_value=None) as mock_init:
            # Test normal initialization
            Prediction.from_pdc(pdc_data)
            mock_init.assert_called_once()

            # Reset the mock for the next test
            mock_init.reset_mock()

            # Test with query
            Prediction.from_pdc(pdc_data, query={"ModelName": ["PREDICTWEBPROPENSITY"]})
            mock_init.assert_called_once()
            assert mock_init.call_args[1].get("query") == {
                "ModelName": ["PREDICTWEBPROPENSITY"],
            }


def test_prediction_plots_internal_method(preds_singleday):
    """Test the internal _prediction_trend method of PredictionPlots."""
    plt, plot_df = preds_singleday.plot._prediction_trend(
        period="1d",
        query=None,
        metric="Performance",
        title="Test Plot",
    )

    # The figure plots one trace per non-multi-channel known channel:
    # Mobile and Web (Unknown / Multi-channel are filtered out).
    assert isinstance(plt, Figure)
    assert sorted(t.name for t in plt.data) == [
        "Mobile (PREDICTMOBILEPROPENSITY)",
        "Web (PREDICTWEBPROPENSITY)",
    ]
    assert plt.layout.title.text.startswith("Test Plot")

    collected_df = plot_df.collect()
    # Four channels (Unknown, Multi-channel, Mobile, Web), one snapshot day.
    assert collected_df.shape == (4, 29)
    assert collected_df["Performance"].to_list() == pytest.approx([0.65, 0.70, 0.65, 0.70], abs=1e-6)
    assert collected_df["Prediction"].to_list() == [
        "Unknown (MYCUSTOMPREDICTION)",
        "Multi-channel (PREDICTACTIONPROPENSITY)",
        "Mobile (PREDICTMOBILEPROPENSITY)",
        "Web (PREDICTWEBPROPENSITY)",
    ]
    assert collected_df["Date"].to_list() == [datetime.date(2040, 4, 1)] * 4


def test_prediction_validity_expr(preds_singleday):
    """Test the prediction_validity_expr class attribute."""
    expr = Prediction.prediction_validity_expr

    result = preds_singleday.predictions.filter(expr).collect()

    # Fixture: 16 raw rows, 4 of which have empty pyDataUsage and are dropped
    # by validity (3 valid usages * 4 model ids = 12 rows kept).
    assert result.shape == (12, 23)
    assert sorted(result["Positives"].unique().to_list()) == [100, 200, 400, 500, 800, 1000]

    invalid_data = mock_prediction_data.with_columns(pyPositives=pl.lit(0))
    invalid_pred = Prediction(invalid_data)

    invalid_result = invalid_pred.predictions.filter(expr).collect()
    assert invalid_result.shape == (0, 23)


def test_init_with_temporal_snapshot_time():
    """Test initialization with already parsed temporal snapshot time."""
    # Create data with datetime column
    data = mock_prediction_data.with_columns(
        pySnapShotTime=pl.lit(datetime.datetime(2040, 4, 1)).cast(pl.Datetime),
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
    assert isinstance(pred.plot, PredictionPlots)
    assert pred.plot.prediction is pred

    # Verify the dependencies attribute
    assert hasattr(pred.plot, "dependencies")
    assert "plotly" in pred.plot.dependencies


def test_from_processed_data():
    """Test the from_processed_data class method."""
    pred = Prediction.from_mock_data()

    temp_path = "temp_processed_data"
    predictions_cache = pred.save_data(temp_path)

    cached_data = read_ds_export(predictions_cache)
    # read_ds_export can legitimately return None when the file is missing;
    # narrow the type here so the equality check below is well-defined.
    assert isinstance(cached_data, pl.LazyFrame)

    loaded_pred = Prediction.from_processed_data(cached_data)
    assert loaded_pred.is_available
    assert loaded_pred.is_valid

    original_df = pred.predictions.collect()
    loaded_df = loaded_pred.predictions.collect()
    assert original_df.equals(loaded_df)

    shutil.rmtree(temp_path)


def test_performance_range_in_summary_methods(preds_singleday):
    """Performance values are exact, normalised, and within the [0.5, 1.0] AUC range."""
    # summary_by_channel: input pyValue alternates 0.65 (4x) / 0.70 (4x) across
    # 4 channels -> after normalisation each channel's Performance is the raw
    # pyValue / 100 multiplied by Pega's *100 representation, giving back the
    # original 0.65 / 0.70 (with float32 rounding from the source column).
    channel_summary = preds_singleday.summary_by_channel().collect()
    performance_values = channel_summary["Performance"].to_list()
    assert performance_values == pytest.approx(
        [0.65, 0.70, 0.65, 0.70],
        rel=1e-6,
    )

    # overall_summary aggregates with response-count weighting; capture the
    # exact computed value so a regression in the weighting changes it.
    overall_summary = preds_singleday.overall_summary().collect()
    overall_perf = overall_summary["Performance"].item()
    assert overall_perf == pytest.approx(0.6794117478763356, rel=1e-9)

    # Mock data is deterministic (no RNG seed used in from_mock_data); these
    # values were captured from the fixture and pin the per-channel AUC.
    pred_mock = Prediction.from_mock_data(days=30)
    mock_channel_summary = pred_mock.summary_by_channel().collect()
    mock_rows = dict(
        zip(
            mock_channel_summary["Channel"].to_list(),
            mock_channel_summary["Performance"].to_list(),
            strict=True,
        ),
    )
    assert mock_rows == pytest.approx(
        {
            "Mobile": 0.7150071889215354,
            "E-mail": 0.625005832742956,
            "Web": 0.6700169096404074,
        },
        rel=1e-9,
    )

    # Secondary invariant: a non-degenerate AUC must sit in [0.5, 1.0].
    for perf in performance_values + [overall_perf] + list(mock_rows.values()):
        assert 0.5 <= perf <= 1.0


def test_performance_normalization_from_pega_scale():
    """Test that Performance is correctly normalized from Pega's 50-100 scale to 0.5-1.0."""
    # Create data with Performance in Pega's 50-100 scale
    pega_scale_data = pl.DataFrame(
        {
            "pySnapShotTime": cdh_utils.to_prpc_date_time(
                datetime.datetime(2040, 4, 1),
            )[0:15],
            "pyModelId": ["DATA-DECISION-REQUEST-CUSTOMER!TESTPREDICTION"] * 4,
            "pyModelType": "PREDICTION",
            "pySnapshotType": ["Daily"] * 3 + [None],
            "pyDataUsage": ["Control", "Test", "NBA", ""],
            "pyPositives": [100, 400, 500, 1000],
            "pyNegatives": [1000, 2000, 3000, 6000],
            "pyCount": [1100, 2400, 3500, 7000],
            "pyValue": [65.0, 70.0, 75.0, 80.0],  # Pega scale (50-100)
        },
    ).lazy()

    # Initialize Prediction (should auto-normalize)
    pred = Prediction(pega_scale_data)

    # Check that Performance values are normalized
    performance = pred.predictions.select("Performance").collect()["Performance"].to_list()

    # Verify all performance values are normalised to the [0.5, 1.0] AUC range
    # and equal exactly value/100 for each Pega-scale input (65 -> 0.65 etc.).
    expected = [0.65, 0.70, 0.75]
    assert performance == pytest.approx(expected, rel=1e-6)
    for perf in performance:
        assert 0.5 <= perf <= 1.0

    # Specifically verify the normalization (65/100 = 0.65, etc.)
    assert abs(performance[0] - 0.65) < 0.001, f"Expected 0.65, got {performance[0]}"
