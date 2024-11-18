"""
Testing the functionality of the BinAggregator
"""

import polars as pl
import pytest
from pdstools import datasets
from pdstools.adm.BinAggregator import BinAggregator
from plotly.graph_objects import Figure


@pytest.fixture
def cdhsample_binaggregator():
    sample = datasets.cdh_sample(
        query=((pl.col("Issue").is_in(["Sales"])) & (pl.col("Direction") == "Inbound"))
    )

    return sample.bin_aggregator


def test_overall_num_rollup(cdhsample_binaggregator):
    assert isinstance(cdhsample_binaggregator.roll_up("Customer.AnnualIncome"), Figure)
    rolled_up_binning = cdhsample_binaggregator.roll_up(
        "Customer.AnnualIncome", return_df=True
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
    assert rolled_up_binning["Models"].to_list() == [4] * 10
    assert rolled_up_binning["BinCoverage"].to_list() == pytest.approx(
        [3.994431] + [4] * 7 + [3.900718, 2.204945], 1e-5
    )
    assert rolled_up_binning["Lift"].to_list() == pytest.approx(
        [
            -0.107611,
            -0.196807,
            -0.166791,
            0.731257,
            0.943641,
            0.943641,
            0.943641,
            0.943641,
            0.967659,
            1.111563,
        ],
        1e-5,
    )
    assert rolled_up_binning["BinResponses"].to_list() == pytest.approx(
        [
            2050.537142,
            1984.863844,
            1273.951683,
            343.582104,
            190.330366,
            190.330366,
            190.330366,
            190.330366,
            190.330366,
            85.413398,
        ],
        1e-8,
    )


def test_overall_sym_rollup(cdhsample_binaggregator):
    rolled_up_binning = cdhsample_binaggregator.roll_up(
        "Customer.MaritalStatus", symbols="Not Known", return_df=True
    )
    assert isinstance(rolled_up_binning, pl.DataFrame)
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
    assert rolled_up_binning["Models"].to_list() == [3] * 5
    assert rolled_up_binning["BinCoverage"].to_list() == [3, 3, 0, 3, 3]
    assert rolled_up_binning["BinSymbol"].to_list() == [
        "Unknown",
        "Single",
        "Not Known",
        "Married",
        "No Resp+",
    ]
    assert rolled_up_binning["Lift"].to_list() == pytest.approx(
        [
            0.392478,
            0.074237,
            0.0,
            -0.208895,
            -0.220171,
        ],
        1e-5,
    )
    assert rolled_up_binning["BinResponses"].to_list() == pytest.approx(
        [
            1507.0,
            1757.0,
            0.0,
            1743.0,
            1683.0,
        ],
        1e-5,
    )


def test_create_empty_numbinning(cdhsample_binaggregator):
    target = cdhsample_binaggregator.create_empty_numbinning(
        predictor="Dummy", n=5, minimum=0, maximum=100
    )
    assert target.shape[0] == 5
    assert target["BinLowerBound"].to_list() == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert target["BinUpperBound"].to_list() == [20.0, 40.0, 60.0, 80.0, 100.0]


def test_create_symbol_list(cdhsample_binaggregator):
    symbols = cdhsample_binaggregator.create_symbol_list(
        predictor="Customer.RiskCode", n_symbols=None, musthave_symbols=["Hello"]
    )
    assert symbols == ["Hello", "R1", "R2", "R3", "R4"]
    symbols = cdhsample_binaggregator.create_symbol_list(
        predictor="Customer.RiskCode", n_symbols=3, musthave_symbols=["Hello"]
    )
    assert symbols == ["Hello", "R1", "R2"]


def test_get_source_numbinning(cdhsample_binaggregator):
    binning = cdhsample_binaggregator.get_source_numbinning(
        "Customer.Age", "08ca1302-9fc0-57bf-9031-d4179d400493"
    )
    assert isinstance(binning, pl.DataFrame)
    assert binning.shape[0] == 11
    assert binning["BinResponses"].sum() == 4605


def test_combine_two_numbinnings(cdhsample_binaggregator):
    target = cdhsample_binaggregator.create_empty_numbinning(
        "Customer.Age", 4, minimum=20, maximum=80
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
        }
    )
    combined = cdhsample_binaggregator.combine_two_numbinnings(source, target)
    assert combined.shape[0] == 4
    assert combined["Lift"].to_list() == pytest.approx([0.066667, -0.1, 2.0, 2.0], 1e-5)
    assert combined["BinCoverage"].to_list() == pytest.approx([1, 1, 1, 0.66667], 1e-5)


@pytest.mark.skip(
    reason="query argument to binaggregator currently no longer supported"
)
def test_subsetting():
    dm = datasets.cdh_sample()
    ba = BinAggregator(dm, query=pl.col("Group").cast(pl.Utf8).str.contains("Loan"))
    result = ba.roll_up("Customer.Age", n=6, aggregation="Group", return_df=True)
    assert result.select(pl.col("Group").n_unique()).item() == 2


def test_log(cdhsample_binaggregator):
    binning = cdhsample_binaggregator.roll_up(
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


def test_plot_binning_attribution(cdhsample_binaggregator):
    target = cdhsample_binaggregator.create_empty_numbinning(
        "Customer.Age", 4, minimum=20, maximum=80
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
        }
    )

    fig = cdhsample_binaggregator.plot_binning_attribution(source, target)
    assert isinstance(fig, Figure)


def test_plot_binning_lift(cdhsample_binaggregator):
    assert isinstance(cdhsample_binaggregator.roll_up("Customer.Age"), Figure)

    assert isinstance(
        cdhsample_binaggregator.plot_lift_binning(
            cdhsample_binaggregator.create_empty_numbinning(
                "Customer.Age", 4, minimum=20, maximum=80
            )
        ),
        Figure,
    )


@pytest.mark.skip(reason="no way of currently testing this")
def test_verbose_mode(cdhsample_binaggregator: BinAggregator, capsys):
    cdhsample_binaggregator.roll_up(
        ["Customer.Prefix", "Customer.Age"], n=6, aggregation="Group", verbose=True
    )
    captured = capsys.readouterr()
    assert "Pivot table:" in captured


# then test a roll up over Group

# and a roll up over Group and two predictors
