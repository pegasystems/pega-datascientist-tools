"""
Testing the functionality of the BinAggregator
"""


import sys
import pytest

import pathlib
from plotly.graph_objects import Figure
import polars as pl

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import datasets
from pdstools import BinAggregator


@pytest.fixture
def cdhsample_binaggregator():
    dm = datasets.CDHSample(subset=False)
    return BinAggregator(dm)


def test_overall_num_rollup(cdhsample_binaggregator):
    assert isinstance(cdhsample_binaggregator.roll_up("Customer.Age"), Figure)
    rolled_up_binning = cdhsample_binaggregator.roll_up("Customer.Age", return_df=True)
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
    assert rolled_up_binning["Models"].to_list() == [20] * 10
    assert rolled_up_binning["BinCoverage"].to_list() == pytest.approx(
        [19.8333333] + [20] * 7 + [19.5, 18.6666667], 1e-5
    )
    assert rolled_up_binning["Lift"].to_list() == pytest.approx(
        [
            0.42120621812754727,
            0.0901427039270719,
            -0.14888281428939504,
            -0.20378680869690577,
            -0.013911905947971337,
            0.26038574790414176,
            0.3732155566449165,
            0.33113205,
            0.3396226153846153,
            0.353823625,
        ],
        1e-8,
    )
    assert rolled_up_binning["BinResponses"].to_list() == pytest.approx(
        [
            2407.832060596152,
            3929.821488827346,
            6070.309319634542,
            7844.387419462578,
            6838.913967899006,
            3774.6789587887847,
            1469.6187810832494,
            1001.5007948345176,
            1001.5007948345176,
            906.4364140393087,
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
    assert rolled_up_binning["Models"].to_list() == [15] * 5
    assert rolled_up_binning["BinCoverage"].to_list() == [15, 15, 0, 15, 15]
    assert rolled_up_binning["BinSymbol"].to_list() == [
        "Unknown",
        "Single",
        "Not Known",
        "Married",
        "No Resp+",
    ]
    assert rolled_up_binning["Lift"].to_list() == pytest.approx(
        [0.493990, 0.055068, 0.0, -0.19841, -0.253957], 1e-5
    )
    assert rolled_up_binning["BinResponses"].to_list() == pytest.approx(
        [8092.0, 9581.5, 0.0, 9312.5, 9502.0], 1e-5
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


# then test a roll up over Group

# and a roll up over Group and two predictors

