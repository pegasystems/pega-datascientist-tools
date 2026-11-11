"""Testing the functionality of the BinAggregator"""

import logging

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.adm.BinAggregator import BinAggregator
from plotly.graph_objects import Figure

NUM_PREDICTOR = "IH.Web.Inbound.Clicked.pxLastOutcomeTime.DaysSince"
SYM_PREDICTOR = "Param.MessageLengthBand"


def test_overall_num_rollup(sample_binaggregator):
    assert isinstance(sample_binaggregator.roll_up(NUM_PREDICTOR), Figure)
    rolled_up_binning = sample_binaggregator.roll_up(
        NUM_PREDICTOR,
        return_df=True,
    )
    assert rolled_up_binning.columns == [
        "PredictorName",
        "BinIndex",
        "BinLowerBound",
        "BinUpperBound",
        "BinSymbol",
        "Lift",
        "BinResponses",
        "BinCoverage",
        "Models",
        "Topic",
    ]
    assert rolled_up_binning["BinLowerBound"].to_list() == pytest.approx(
        [730.0 * i for i in range(10)],
    )
    assert rolled_up_binning["Models"].to_list() == [46] * 10
    assert rolled_up_binning["BinCoverage"].to_list() == pytest.approx([46] * 10)
    assert rolled_up_binning["Lift"].to_list() == pytest.approx(
        [
            -0.019028,
            -0.021609,
            -0.041880,
            -0.044132,
            -0.044132,
            -0.008921,
            0.026291,
            0.026291,
            0.026291,
            0.026291,
        ],
        abs=1e-6,
    )
    assert rolled_up_binning["BinResponses"].to_list() == pytest.approx(
        [
            8892386.217975,
            1781672.606452,
            1313828.408653,
            1261845.720008,
            1261845.720008,
            1211439.134104,
            1161032.548200,
            1161032.548200,
            1161032.548200,
            1161032.548200,
        ],
        1e-8,
    )


def test_overall_sym_rollup(sample_binaggregator):
    rolled_up_binning = sample_binaggregator.roll_up(
        SYM_PREDICTOR,
        symbols="Not Known",
        return_df=True,
    )
    assert rolled_up_binning.columns == [
        "PredictorName",
        "BinIndex",
        "BinLowerBound",
        "BinUpperBound",
        "BinSymbol",
        "Lift",
        "BinResponses",
        "BinCoverage",
        "Models",
        "Topic",
    ]
    assert rolled_up_binning["Models"].to_list() == [27] * 5
    assert rolled_up_binning["BinCoverage"].to_list() == [27, 27, 0, 27, 27]
    assert rolled_up_binning["BinSymbol"].to_list() == [
        "Residual",
        "Short",
        "Not Known",
        "Long",
        "Medium",
    ]
    lift = rolled_up_binning["Lift"].to_list()
    assert lift[0] is None
    assert lift[1:] == pytest.approx([0.023952, 0.0, -0.007091, -0.048932], abs=1e-6)
    assert rolled_up_binning["BinResponses"].to_list() == pytest.approx(
        [0.0, 5711976.0, 0.0, 2705312.0, 3922267.0],
        1e-5,
    )


def test_create_empty_numbinning(sample_binaggregator):
    target = sample_binaggregator.create_empty_numbinning(
        predictor="Dummy",
        n=5,
        minimum=0,
        maximum=100,
    )
    assert target.shape[0] == 5
    assert target["BinLowerBound"].to_list() == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert target["BinUpperBound"].to_list() == [20.0, 40.0, 60.0, 80.0, 100.0]


def test_create_symbol_list(sample_binaggregator):
    symbols = sample_binaggregator.create_symbol_list(
        predictor=SYM_PREDICTOR,
        n_symbols=None,
        musthave_symbols=["Hello"],
    )
    assert symbols == ["Hello", "Short", "Medium", "Long", "Residual"]
    symbols = sample_binaggregator.create_symbol_list(
        predictor=SYM_PREDICTOR,
        n_symbols=3,
        musthave_symbols=["Hello"],
    )
    assert symbols == ["Hello", "Short", "Medium"]


def test_get_source_numbinning(sample_binaggregator):
    binning = sample_binaggregator.get_source_numbinning(
        NUM_PREDICTOR,
        "05085857-c52c-5daa-b51b-0af576a038af",
    )
    assert binning.shape[0] == 7
    assert binning["BinResponses"].sum() == 181827


def test_combine_two_numbinnings(sample_binaggregator):
    target = sample_binaggregator.create_empty_numbinning(
        "Customer.Age",
        4,
        minimum=20,
        maximum=80,
    )
    source = pl.DataFrame(
        {
            "ModelID": [1] * 3,
            "PredictorName": ["Customer.Age"] * 3,
            "BinIndex": [1, 2, 3],
            "BinLowerBound": [10.0, 25.0, 50.0],
            "BinUpperBound": [25.0, 50.0, 75.0],
            "Lift": [0.4, -0.1, 2.0],
            "BinResponses": [100, 1000, 400],
        },
    )
    combined = sample_binaggregator.combine_two_numbinnings(source, target)
    assert combined.shape[0] == 4
    assert combined["Lift"].to_list() == pytest.approx([0.066667, -0.1, 2.0, 2.0], 1e-5)
    assert combined["BinCoverage"].to_list() == pytest.approx([1, 1, 1, 0.66667], 1e-5)


@pytest.mark.skip(
    reason="query argument to binaggregator currently no longer supported",
)
def test_subsetting():
    dm = ADMDatamart.from_mock_data(end="canonical")
    ba = BinAggregator(dm, query=pl.col("Group").cast(pl.Utf8).str.contains("Loan"))
    result = ba.roll_up("Customer.Age", n=6, aggregation="Group", return_df=True)
    assert result.select(pl.col("Group").n_unique()).item() == 2


def test_log(sample_binaggregator):
    binning = sample_binaggregator.roll_up(
        "Customer.NetWealth",
        boundaries=[10000, 20000],
        n=8,
        minimum=10000,
        maximum=50000,
        distribution="log",
        return_df=True,
    )
    assert binning["BinLowerBound"].to_list() == pytest.approx(
        [
            10000.0,
            20000.0,
            22797.045620951936,
            25985.264452188192,
            29619.362959451737,
            33761.69843250776,
            38483.348970335035,
            43865.33310618707,
        ],
        1e-8,
    )


def test_plot_binning_attribution(sample_binaggregator):
    target = sample_binaggregator.create_empty_numbinning(
        "Customer.Age",
        4,
        minimum=20,
        maximum=80,
    )
    source = pl.DataFrame(
        {
            "ModelID": [1] * 3,
            "PredictorName": ["Customer.Age"] * 3,
            "BinIndex": [1, 2, 3],
            "BinLowerBound": [10.0, 25.0, 50.0],
            "BinUpperBound": [25.0, 50.0, 75.0],
            # "BinSymbol" : ["20-25", "25-50", "50-75"],
            "Lift": [0.4, -0.1, 2.0],
            "BinResponses": [100, 1000, 400],
        },
    )

    fig = sample_binaggregator.plot_binning_attribution(source, target)
    assert isinstance(fig, Figure)


def test_plot_binning_lift(sample_binaggregator):
    assert isinstance(sample_binaggregator.roll_up(NUM_PREDICTOR), Figure)

    assert isinstance(
        sample_binaggregator.plot_lift_binning(
            sample_binaggregator.create_empty_numbinning(
                "Customer.Age",
                4,
                minimum=20,
                maximum=80,
            ),
        ),
        Figure,
    )


def test_logging_output(sample_binaggregator: BinAggregator, caplog):
    """Test that operations produce debug logs when logging enabled."""
    with caplog.at_level(logging.DEBUG):
        sample_binaggregator.roll_up(
            [SYM_PREDICTOR, NUM_PREDICTOR],
            n=6,
            aggregation="Group",
            return_df=True,
        )

    # Should have debug messages about topics and predictors
    assert any("Topic:" in record.message for record in caplog.records)
    assert any("Model ID:" in record.message for record in caplog.records)
    # For symbolic predictors
    assert any("Symbols:" in record.message for record in caplog.records)
    # For symbolic pivot table
    assert any("Pivot table:" in record.message for record in caplog.records)


# then test a roll up over Group

# and a roll up over Group and two predictors
