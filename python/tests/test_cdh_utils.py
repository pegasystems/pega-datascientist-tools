"""
Testing the functionality of the cdh_utils functions
"""

import datetime
import pathlib
import sys

import numpy as np
import polars as pl
import pytest
from pytz import timezone

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import cdh_utils, datasets


def test_safe_range_auc():
    assert cdh_utils.safe_range_auc(0.8) == 0.8
    assert cdh_utils.safe_range_auc(0.4) == 0.6
    assert cdh_utils.safe_range_auc(np.nan) == 0.5


def test_auc_from_probs():
    assert cdh_utils.auc_from_probs([1, 1, 0], [0.6, 0.2, 0.2]) == 0.75
    assert cdh_utils.auc_from_probs([1, 1, 1], [0.6, 0.2, 0.2]) == 0.5
    assert (
        cdh_utils.auc_from_probs(
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [20, 19, 18, 17, 16, 15, 14, 13, 11.5, 11.5, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        )
        == 0.825
    )
    assert (
        abs(
            cdh_utils.auc_from_probs(
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                [
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.333333333333333,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.666666666666667,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                    0.875,
                ],
            )
            - 0.7637363
        )
        < 1e-6
    )


def test_auc_from_bincounts():
    assert cdh_utils.auc_from_bincounts([3, 1, 0], [2, 0, 1]) == 0.75
    positives = [50, 70, 75, 80, 85, 90, 110, 130, 150, 160]
    negatives = [1440, 1350, 1170, 990, 810, 765, 720, 675, 630, 450]
    assert abs(cdh_utils.auc_from_bincounts(positives, negatives) - 0.6871) < 1e-6


def test_aucpr_from_probs():
    assert (
        abs(cdh_utils.aucpr_from_probs([1, 1, 0], [0.6, 0.2, 0.2]) - 0.4166667) < 1e-6
    )
    assert cdh_utils.aucpr_from_probs([1, 1, 1], [0.6, 0.2, 0.2]) == 0.0
    # assert abs(cdh_utils.aucpr_from_probs([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    #    [0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.666666666666667, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875]) - 0.8610687) < 1e-6


def test_aucpr_from_bincounts():
    assert cdh_utils.aucpr_from_bincounts([3, 1, 0], [2, 0, 1]) == 0.625
    assert (
        abs(
            cdh_utils.aucpr_from_bincounts(
                [50, 70, 75, 80, 85, 90, 110, 130, 150, 160],
                [1440, 1350, 1170, 990, 810, 765, 720, 675, 630, 450],
            )
            - 0.1489611
        )
        < 1e-6
    )


def test_auc2gini():
    assert abs(cdh_utils.auc2GINI(0.8232) - 0.6464) < 1e-6


def test_fromPRPCDateTime():
    assert (
        cdh_utils.fromPRPCDateTime("20180316T134127.847 GMT", True)
        == "2018-03-16 13:41:27 GMT"
    )
    assert cdh_utils.fromPRPCDateTime(
        "20180316T134127.847 GMT", False
    ) == datetime.datetime(2018, 3, 16, 13, 41, 27, 847, tzinfo=timezone("GMT"))
    assert (
        cdh_utils.fromPRPCDateTime("20180316T134127.847345", True)
        == "2018-03-16 13:41:27 "
    )
    assert (
        cdh_utils.fromPRPCDateTime("20180316T134127.8", True) == "2018-03-16 13:41:27 "
    )
    assert cdh_utils.fromPRPCDateTime("20180316T134127", True) == "2018-03-16 13:41:27 "


def test_toPRPCDateTime():
    assert (
        cdh_utils.toPRPCDateTime(
            datetime.datetime(2018, 3, 16, 13, 41, 27, 847000, tzinfo=timezone("GMT"))
        )
        == "20180316T134127.847 GMT+0000"
    )
    assert (
        cdh_utils.toPRPCDateTime(
            datetime.datetime(
                2018, 3, 16, 13, 41, 27, 847000, tzinfo=timezone("US/Eastern")
            )
        )
        == "20180316T134127.847 GMT-0456"
    )
    assert (
        cdh_utils.toPRPCDateTime(datetime.datetime(2018, 3, 16, 13, 41, 27, 847000))[
            :-3
        ]
        == "20180316T134127.847 GMT+0000"[:-3]
    )


def test_weighted_average_polars():
    input = pl.DataFrame(
        {
            "SuccessRate": [1, 3, 10, 40],
            "Channel": ["SMS", "SMS", "Web", "Web"],
            "ResponseCount": [3, 1, 10, 5],
        }
    )
    output = (
        input.group_by("Channel")
        .agg(
            cdh_utils.weighted_average_polars("SuccessRate", "ResponseCount").alias(
                "SuccessRate_weighted"
            ),
        )
        .sort("Channel")
    )

    expected_output = pl.DataFrame(
        {"Channel": ["SMS", "Web"], "SuccessRate_weighted": [1.5, 20]}
    ).sort("Channel")

    assert output.equals(expected_output)

    output = (
        input.filter(pl.col("Channel") == "SMS")
        .with_columns(
            cdh_utils.weighted_average_polars(
                vals="SuccessRate", weights="ResponseCount"
            ).alias("weighted_average")
        )
        .sort("SuccessRate")
    )

    expected_output = pl.DataFrame(
        {
            "SuccessRate": [1, 3],
            "Channel": ["SMS", "SMS"],
            "ResponseCount": [3, 1],
            "weighted_average": [1.5, 1.5],
        }
    ).sort("SuccessRate")

    assert output.equals(expected_output)


def test_weighted_performance_polars():
    input = pl.DataFrame(
        {
            "Performance": [0.5, 0.8, 0.75, 0.5],  # 0.6, 0.6
            "Channel": ["SMS", "SMS", "Web", "Web"],
            "ResponseCount": [2, 1, 2, 3],
        }
    )

    output = (
        input.group_by("Channel")
        .agg(cdh_utils.weighted_performance_polars())
        .sort("Channel")
    )

    expected_output = pl.DataFrame(
        {"Channel": ["SMS", "Web"], "Performance": [0.6, 0.6]}
    ).sort("Channel")

    assert output.equals(expected_output)


def test_zRatio():
    input = pl.DataFrame(
        {
            "Predictor_range": [
                "MISSING",
                "<7.08",
                "[7.08, 12.04>",
                "[12.04, 18.04>",
                ">=18.04",
            ],
            "BinPositives": [0, 7, 11, 12, 6],
            "BinNegatives": [5, 2208, 1919, 1082, 352],
        }
    )

    output = (
        input.with_columns(cdh_utils.zRatio())
        .with_columns(pl.col("ZRatio").round(2))
        .sort("Predictor_range")
    )

    Zratios = [-3.05, 1.66, -2.24, 1.76, -0.51]
    expected_output = input.sort("Predictor_range").with_columns(
        pl.Series(name="ZRatio", values=Zratios)
    )

    assert output.equals(expected_output)


def test_lift():
    input = pl.DataFrame(
        {
            "BinPositives": [0, 7, 11, 12, 6],
            "BinNegatives": [5, 2208, 1919, 1082, 352],
        }
    )

    output = input.with_columns(cdh_utils.lift()).with_columns(pl.col("Lift").round(7))

    vals = [0, 0.4917733, 0.8869027, 1.7068860, 2.6080074]
    expected_output = input.with_columns(pl.Series(name="Lift", values=vals))

    assert output.equals(expected_output)


def test_log_odds():
    input = pl.DataFrame(
        {
            "Predictor_range": [
                "MISSING",
                "<7.08",
                "[7.08, 12.04>",
                "[12.04, 18.04>",
                ">=18.04",
            ],
            "Positives": [0, 7, 11, 12, 6],
            "ResponseCount": [5, 2215, 1930, 1094, 358],
        }
    )
    output = input.with_columns(cdh_utils.LogOdds().round(2))

    log_odds_list = [1.65, -0.81, -0.23, 0.43, 0.87]
    expected_output = input.with_columns(
        pl.Series(name="LogOdds", values=log_odds_list)
    )

    assert output.equals(expected_output)


def test_featureImportance():
    input = pl.DataFrame(
        {
            "ModelID": ["001", "001", "002", "002"],
            "PredictorName": ["Age", "Age", "CreditScore", "CreditScore"],
            "BinResponseCount": [20, 30, 12, 10],
            "BinPositives": [2, 7, 11, 6],
        }
    )

    output = input.with_columns(cdh_utils.featureImportance().round(2)).sort(
        "BinPositives"
    )
    importance_list = [-0.12, 0.28, -0.12, 0.28]
    expected_output = input.sort("BinPositives").with_columns(
        pl.Series(name="FeatureImportance", values=importance_list)
    )

    assert output.equals(expected_output)


# Test _capitalize function
def test_capitalize_behavior():
    assert set(cdh_utils._capitalize(["pyTest", "pzTest", "pxTest"])) == {"Test"}
    assert cdh_utils._capitalize(["test"]) == ["Test"]
    assert cdh_utils._capitalize(["RESPONSEcOUNT"]) == ["ResponseCount"]
    assert cdh_utils._capitalize(["responsecounter"]) == ["ResponseCounter"]
    assert cdh_utils._capitalize(["responsenumber"]) == ["Responsenumber"]
    assert cdh_utils._capitalize(["Response Count"]) == ["Response Count"]
    assert cdh_utils._capitalize(["Response_count1"]) == ["Response_Count1"]
    assert cdh_utils._capitalize("Response_count1") == ["Response_Count1"]


def test_PredictorCategorization():
    df = (
        pl.LazyFrame({"PredictorName": ["Customer.Variable", "Variable"]})
        .with_columns(cdh_utils.defaultPredictorCategorization())
        .collect()
    )
    assert df.get_column("PredictorCategory").to_list() == ["Customer", "Primary"]

    df = (
        pl.LazyFrame({"Predictor": ["Customer.Variable", "Variable"]})
        .with_columns(cdh_utils.defaultPredictorCategorization(x="Predictor"))
        .collect()
    )
    assert df.get_column("PredictorCategory").to_list() == ["Customer", "Primary"]


def test_gains_table():
    # Only "y" values
    df = pl.DataFrame({"Age": [10, 20, 30, 20]})
    gains = cdh_utils.gains_table(df, "Age")
    assert gains.get_column("cum_x").to_list() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert gains.get_column("cum_y").to_list() == [0.0, 0.375, 0.625, 0.875, 1.0]

    # With an index of the same values result should be the same
    df = pl.DataFrame({"Age": [10, 20, 30, 20], "PopulationSize": [1, 1, 1, 1]})
    gains = cdh_utils.gains_table(df, value="Age", index="PopulationSize")
    assert gains.get_column("cum_x").to_list() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert gains.get_column("cum_y").to_list() == [0.0, 0.375, 0.625, 0.875, 1.0]

    # When index values different, the sorting will be different
    df = pl.DataFrame({"Age": [10, 20, 30, 20], "PopulationSize": [1, 2, 4, 1]})
    gains = cdh_utils.gains_table(df, "Age", "PopulationSize")
    assert gains.get_column("cum_x").to_list() == [0.0, 0.125, 0.250, 0.500, 1.000]
    assert gains.get_column("cum_y").to_list() == [0.0, 0.250, 0.375, 0.625, 1.000]

    # Numbers taken from the classifier view of the default model report. These are
    # already ordered by CTR
    df = pl.DataFrame(
        {
            "Responses": [
                16.0,
                12.0,
                129.0,
                374.0,
                419.0,
                79.0,
                77.0,
                563.0,
                234.0,
                220.0,
                870.0,
                949.0,
                4.0,
            ],
            "Positives": [
                8.0,
                5.0,
                49.0,
                103.0,
                108.0,
                20.0,
                15.0,
                106.0,
                36.0,
                31.0,
                95.0,
                89.0,
                0.0,
            ],
            "CumPos": [
                0.012000000476837159,
                0.019500000476837157,
                0.09319999694824219,
                0.24809999465942384,
                0.41049999237060547,
                0.44060001373291013,
                0.4631999969482422,
                0.622599983215332,
                0.6766999816894531,
                0.7233000183105469,
                0.8662000274658204,
                1.0,
                1.0,
            ],
            "CumTotal": [
                0.0040999999642372135,
                0.0070999997854232786,
                0.03980000019073486,
                0.13460000038146971,
                0.24079999923706055,
                0.26079999923706054,
                0.28030000686645506,
                0.4229999923706055,
                0.4822999954223633,
                0.5379999923706055,
                0.7584999847412109,
                0.999000015258789,
                1.0,
            ],
        }
    )
    # without responses, x should increase by 1/13 and order is order of positives
    gains = cdh_utils.gains_table(df, "Positives")
    # x should increase by 1/13
    assert gains.get_column("cum_x").round(5).to_list() == [
        0,
        0.07692,
        0.15385,
        0.23077,
        0.30769,
        0.38462,
        0.46154,
        0.53846,
        0.61538,
        0.69231,
        0.76923,
        0.84615,
        0.92308,
        1,
    ]
    expected_gains = [
        round(e, 5)
        for e in [
            0.162406015037594,
            0.321804511278195,
            0.476691729323308,
            0.619548872180451,
            0.753383458646616,
            0.827067669172932,
            0.881203007518797,
            0.92781954887218,
            0.957894736842105,
            0.980451127819549,
            0.992481203007519,
            1,
            1,
        ]
    ]
    assert gains.get_column("cum_y").round(5).to_list() == [0] + expected_gains

    # with responses, the order should be the same as in the classifier data
    gains = cdh_utils.gains_table(df, "Positives", index="Responses")
    assert gains.get_column("cum_x").round(4).to_list() == [0] + [
        round(e, 4) for e in df["CumTotal"].to_list()
    ]
    assert gains.get_column("cum_y").round(4).to_list() == [0] + [
        round(e, 4) for e in df["CumPos"].to_list()
    ]

    # With a grouping field

    df = pl.DataFrame(
        {
            "Positives": [100, 10, 20, 500, 5000, 30, 10000, 15000, 20],
            "Dimension": ["A", "B", "B", "A", "A", "B", "A", "A", "B"],
        }
    )
    gains = cdh_utils.gains_table(df, "Positives", by="Dimension")
    assert gains.get_column("cum_x").round(5).to_list() == [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert gains.get_column("cum_y").round(5).to_list() == [
        0,
        0.49020,
        0.81699,
        0.98039,
        0.99673,
        1,
        0.0,
        0.375,
        0.625,
        0.875,
        1.0,
    ]

    # df = pl.DataFrame(
    #     {"Positives" : [100, 10, 20, 500, 5000, 30, 10000, 15000, 20],
    #      "Count" : [1, 1, 2, 1, 1, 4, 1, 1, 1],
    #      "Dimension" : ['A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'B']}
    # )
    # gains = cdh_utils.gains_table(df, "Positives", index="Count", by = "Dimension")
    # print(gains)
    # assert gains.get_column("cum_x").to_list() == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5, 0.75, 0.875, 1.0]
    # assert gains.get_column("cum_y").round(5).to_list() == [0, 0.49020, 0.81699, 0.98039, 0.99673, 1, 0.0, 0.375, 0.625, 0.875, 1.0]


def test_legend_color_order():
    input_fig = datasets.CDHSample().plotPerformanceSuccessRateBubbleChart()
    output_fig = cdh_utils.legend_color_order(input_fig)

    assert output_fig.data[0].marker.color == "#001F5F"
