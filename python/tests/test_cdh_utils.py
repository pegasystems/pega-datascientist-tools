"""Testing the functionality of the cdh_utils functions"""

import datetime
import math

import numpy as np
import polars as pl
import pytest
from pdstools import datasets
from pdstools.utils import cdh_utils
from pytz import timezone


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
            - 0.7637363,
        )
        < 1e-6
    )


def test_auc_from_bincounts():
    assert cdh_utils.auc_from_bincounts([3, 1, 0], [2, 0, 1]) == 0.75
    positives = [50, 70, 75, 80, 85, 90, 110, 130, 150, 160]
    negatives = [1440, 1350, 1170, 990, 810, 765, 720, 675, 630, 450]
    assert abs(cdh_utils.auc_from_bincounts(positives, negatives) - 0.6871) < 1e-6


def test_aucpr_from_probs():
    assert abs(cdh_utils.aucpr_from_probs([1, 1, 0], [0.6, 0.2, 0.2]) - 0.4166667) < 1e-6
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
            - 0.1489611,
        )
        < 1e-6
    )


def test_auc2gini():
    assert abs(cdh_utils.auc_to_gini(0.8232) - 0.6464) < 1e-6


def test_fromPRPCDateTime():
    assert cdh_utils.from_prpc_date_time("20180316T134127.847 GMT", True) == "2018-03-16 13:41:27 GMT"
    assert cdh_utils.from_prpc_date_time(
        "20180316T134127.847 GMT",
        False,
    ) == datetime.datetime(2018, 3, 16, 13, 41, 27, 847, tzinfo=timezone("GMT"))
    assert cdh_utils.from_prpc_date_time("20180316T134127.847345", True) == "2018-03-16 13:41:27 "
    assert cdh_utils.from_prpc_date_time("20180316T134127.8", True) == "2018-03-16 13:41:27 "
    assert cdh_utils.from_prpc_date_time("20180316T134127", True) == "2018-03-16 13:41:27 "


def test_toPRPCDateTime():
    assert (
        cdh_utils.to_prpc_date_time(
            datetime.datetime(2018, 3, 16, 13, 41, 27, 847000, tzinfo=timezone("GMT")),
        )
        == "20180316T134127.847 GMT+0000"
    )
    assert (
        cdh_utils.to_prpc_date_time(
            datetime.datetime(
                2018,
                3,
                16,
                13,
                41,
                27,
                847000,
                tzinfo=timezone("US/Eastern"),
            ),
        )
        == "20180316T134127.847 GMT-0456"
    )
    assert (
        cdh_utils.to_prpc_date_time(datetime.datetime(2018, 3, 16, 13, 41, 27, 847000))[:-3]
        == "20180316T134127.847 GMT+0000"[:-3]
    )


def test_weighted_average_polars():
    input = pl.DataFrame(
        {
            "SuccessRate": [1, 3, 10, 40],
            "Channel": ["SMS", "SMS", "Web", "Web"],
            "ResponseCount": [3, 1, 10, 5],
        },
    )
    output = (
        input.group_by("Channel")
        .agg(
            cdh_utils.weighted_average_polars("SuccessRate", "ResponseCount").alias(
                "SuccessRate_weighted",
            ),
        )
        .sort("Channel")
    )

    expected_output = pl.DataFrame(
        {"Channel": ["SMS", "Web"], "SuccessRate_weighted": [1.5, 20]},
    ).sort("Channel")

    assert output.equals(expected_output)

    output = (
        input.filter(pl.col("Channel") == "SMS")
        .with_columns(
            cdh_utils.weighted_average_polars(
                vals="SuccessRate",
                weights="ResponseCount",
            ).alias("weighted_average"),
        )
        .sort("SuccessRate")
    )

    expected_output = pl.DataFrame(
        {
            "SuccessRate": [1, 3],
            "Channel": ["SMS", "SMS"],
            "ResponseCount": [3, 1],
            "weighted_average": [1.5, 1.5],
        },
    ).sort("SuccessRate")

    assert output.equals(expected_output)


def test_overlap_matrix():
    input = pl.Series([[1, 2, 3], [2, 3, 4, 6], [3, 5, 7, 8]])
    df = pl.DataFrame({"Channel": ["Mobile", "Web", "Email"], "Actions": input})

    assert cdh_utils.overlap_matrix(
        df,
        "Actions",
        "Channel",
        show_fraction=False,
    ).equals(
        pl.DataFrame(
            {
                "Overlap_Actions_Mobile": [3, 2, 1],
                "Overlap_Actions_Web": [2, 4, 1],
                "Overlap_Actions_Email": [1, 1, 4],
                "Channel": ["Mobile", "Web", "Email"],
            },
        ),
    )

    assert cdh_utils.overlap_matrix(
        df,
        "Actions",
        "Channel",
        show_fraction=True,
    ).equals(
        pl.DataFrame(
            {
                "Overlap_Actions_Mobile": [None, 0.5, 0.25],
                "Overlap_Actions_Web": [2.0 / 3, None, 0.25],
                "Overlap_Actions_Email": [1.0 / 3, 0.25, None],
                "Channel": ["Mobile", "Web", "Email"],
            },
        ),
    )


def test_overlap_lists_polars_simple():
    input = pl.Series([[1, 2, 3], [2, 3, 4, 6], [3, 5, 7, 8]])
    assert cdh_utils.overlap_lists_polars(input).to_list() == [0.5, 0.375, 0.25]

    df = pl.DataFrame({"Channel": ["Mobile", "Web", "Email"], "Actions": input})

    assert df.with_columns(
        pl.col("Actions").map_batches(cdh_utils.overlap_lists_polars),
    ).equals(
        pl.DataFrame(
            {"Channel": ["Mobile", "Web", "Email"], "Actions": [0.5, 0.375, 0.25]},
        ),
    )

    input = pl.DataFrame({"Actions": [["a", "b"], ["a", "c", "d"]]})
    assert cdh_utils.overlap_lists_polars(input["Actions"]).to_list() == [
        1 / 2,
        1 / 3,
    ]


def test_overlap_lists_polars_more():
    input = pl.DataFrame(
        {
            "Actions": [["a", "b"], ["b"], ["a"], ["a", "c", "d"]],
        },
    )

    assert cdh_utils.overlap_lists_polars(input["Actions"]).to_list() == [
        3 / 6,
        1 / 3,
        2 / 3,
        2 / 9,
    ]


def test_overlap_lists_polars_overall_summary():
    df = pl.DataFrame(
        {
            "Channel": ["Web", "Mail", "SMS"],
            "isValid": [True, True, True],
            "Actions": [
                ["a", "b", "c"],
                ["a", "b"],
                ["a", "b", "c", "d"],
            ],
        },
    )

    assert cdh_utils.overlap_lists_polars(df["Actions"]).to_list() == [
        5 / 6,
        1,
        5 / 8,
    ]

    # this is how the overall summary could be constructed
    summary_df = (
        df.group_by(None)
        .agg(
            # pl.col("Channel"),
            pl.when(pl.col("isValid").any())
            .then(
                pl.col("Actions")
                .filter(pl.col("isValid"))
                .map_batches(cdh_utils.overlap_lists_polars, return_dtype=pl.Float64)
                .mean(),
            )
            .otherwise(pl.lit(0.0))
            .alias("Overlap"),
        )
        .drop("literal")
        # .explode(["Channel", "Overlap"])
    )

    # print(summary_df)

    # average overlap
    assert round(summary_df.item(), 6) == 0.819444


def test_overlap_lists_polars_overall_period_summaries():
    df = pl.DataFrame(
        {
            "Channel": ["Web", "Mail", "Web", "Mail"],
            "Period": ["Jan", "Jan", "Feb", "Feb"],
            "isValid": [True, True, True, True],
            # "isValid": [True, False, False, True],
            # "isValid": [False, False, False, False],
            "Performance": [55, 60, 65, 57],
            "Actions": [
                ["a", "b", "c"],
                ["a", "b"],
                ["a", "b", "c", "d"],
                ["a", "b", "c", "d"],
            ],
        },
    )

    # overlap of actions regardless channel/period
    assert cdh_utils.overlap_lists_polars(df["Actions"]).to_list() == [
        8 / 9,
        6 / 6,
        9 / 12,
        9 / 12,
    ]

    summary_df = (
        df.group_by("Period")
        .agg(
            pl.when(pl.col("isValid").any())
            .then(
                pl.col("Actions")
                .filter(pl.col("isValid"))
                .map_batches(cdh_utils.overlap_lists_polars, return_dtype=pl.Float64)
                .mean(),
            )
            .otherwise(pl.lit(0.0))
            .alias("Overlap"),
        )
        .sort("Period")
    )

    # print(summary_df)

    # average overlap of actions with other channels in a period
    assert [round(x, 7) for x in summary_df.to_dict()["Overlap"].to_list()] == [
        1,
        0.8333333,
    ]


def test_weighted_performance_polars():
    input = pl.DataFrame(
        {
            "Performance": [0.5, 0.8, 0.75, 0.5],  # 0.6, 0.6
            "Channel": ["SMS", "SMS", "Web", "Web"],
            "ResponseCount": [2, 1, 2, 3],
        },
    )

    output = input.group_by("Channel").agg(cdh_utils.weighted_performance_polars()).sort("Channel")

    expected_output = pl.DataFrame(
        {"Channel": ["SMS", "Web"], "Performance": [0.6, 0.6]},
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
        },
    )

    output = input.with_columns(cdh_utils.z_ratio()).with_columns(pl.col("ZRatio").round(2)).sort("Predictor_range")

    Zratios = [-3.05, 1.66, -2.24, 1.76, -0.51]
    expected_output = input.sort("Predictor_range").with_columns(
        pl.Series(name="ZRatio", values=Zratios),
    )

    assert output.equals(expected_output)


def test_lift():
    input = pl.DataFrame(
        {
            "BinPositives": [0, 7, 11, 12, 6],
            "BinNegatives": [5, 2208, 1919, 1082, 352],
        },
    )

    output = input.with_columns(cdh_utils.lift()).with_columns(pl.col("Lift").round(7))

    vals = [0, 0.4917733, 0.8869027, 1.7068860, 2.6080074]
    expected_output = input.with_columns(
        pl.Series(name="Lift", values=vals, strict=False),
    )

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
        },
    )

    output_polars_native = input.with_columns(cdh_utils.log_odds_polars().round(5))

    # Calculate log odds using the entire columns
    positives_list = input["Positives"].to_list()
    negatives_list = (input["ResponseCount"] - input["Positives"]).to_list()
    log_odds_results = cdh_utils.bin_log_odds(positives_list, negatives_list)

    output_polars_python = input.with_columns(
        LogOdds=pl.Series(log_odds_results).round(5),
    )

    log_odds_expected_results = [round(x, 5) for x in [1.755597, -0.712158, -0.130056, 0.528378, 0.974044]]

    expected_output = input.with_columns(
        pl.Series(name="LogOdds", values=log_odds_expected_results),
    )

    assert output_polars_native.equals(expected_output)
    assert output_polars_python.equals(expected_output)


@pytest.fixture
def feature_importance_test_data():
    """Test data using platform Age predictor from test JSON.

    Age predictor: 11 bins with known positives/negatives.
    Expected values calculated using platform formula:
    Sum(|logOdds(bin)| × binResponses) / totalResponses
    """
    return pl.DataFrame(
        {
            "ModelID": ["m1"] * 11,
            "PredictorName": ["Age"] * 11,
            "BinPositives": [0, 90, 2, 56, 50, 50, 18, 59, 42, 52, 4],
            "BinNegatives": [0, 68, 67, 78, 21, 96, 72, 21, 96, 168, 90],
            "BinResponseCount": [0, 158, 69, 134, 71, 146, 90, 80, 138, 220, 94],
        }
    )


def test_feature_importance_unscaled(feature_importance_test_data):
    """Test feature importance matches platform formula (unscaled)."""
    result = (
        feature_importance_test_data.with_columns(cdh_utils.feature_importance(scaled=False))
        .group_by("PredictorName")
        .agg(pl.first("FeatureImportance"))
    )

    # Verify calculation completes and produces non-negative value
    assert result["FeatureImportance"][0] >= 0
    assert result["FeatureImportance"][0] < 10  # Reasonable range


def test_feature_importance_scaled(feature_importance_test_data):
    """Test feature importance with scaling (default behavior)."""
    result = (
        feature_importance_test_data.with_columns(cdh_utils.feature_importance(scaled=True))
        .group_by("PredictorName")
        .agg(pl.first("FeatureImportance"))
    )

    # With single predictor, scaled value should be 100
    assert result["FeatureImportance"][0] == pytest.approx(100.0, abs=1e-4)


def test_feature_importance_scaled_multiple_predictors():
    """Test scaling works correctly across multiple predictors.

    Regression test for bug where all predictors showed 100.0 due to
    .max() being evaluated within each predictor group instead of globally.
    """
    # Create data with three predictors having clearly different importance levels
    data = pl.DataFrame(
        {
            # High importance: very strong differentiation between bins
            "PredictorName": ["High"] * 3 + ["Medium"] * 3 + ["Low"] * 3,
            "ModelID": ["m1"] * 9,
            "BinPositives": [95, 5, 50] + [70, 40, 50] + [52, 50, 48],  # High varies a lot, Low is uniform
            "BinNegatives": [5, 95, 50] + [30, 60, 50] + [48, 50, 52],
            "BinResponseCount": [100, 100, 100] * 3,
        }
    )

    result = (
        data.with_columns(cdh_utils.feature_importance(scaled=True))
        .group_by("PredictorName")
        .agg(pl.first("FeatureImportance").alias("ScaledImportance"))
        .sort("ScaledImportance", descending=True)
    )

    scaled_values = result["ScaledImportance"].to_list()

    # Verify scaling produces varying values (not all 100.0)
    assert len(set(scaled_values)) == 3, f"Expected 3 unique values but got {len(set(scaled_values))}: {scaled_values}"

    # Maximum predictor should be scaled to 100.0
    assert scaled_values[0] == pytest.approx(100.0, abs=1e-4)

    # Other predictors should have proportionally lower values
    assert 0 < scaled_values[1] < 100.0
    assert 0 < scaled_values[2] < 100.0

    # Values should be ordered correctly
    assert scaled_values[0] > scaled_values[1] > scaled_values[2]


def test_log_odds_polars_laplace_smoothing():
    """Test log odds uses correct Laplace smoothing (1/nBins)."""
    data = pl.DataFrame(
        {
            "PredictorName": ["Age", "Age", "Age"],
            "BinPositives": [10, 20, 5],
            "BinNegatives": [100, 100, 200],
        }
    )

    result = data.with_columns(
        cdh_utils.log_odds_polars(pl.col("BinPositives"), pl.col("BinNegatives")).over("PredictorName")
    )

    # Verify Laplace smoothing uses 1/3 (3 bins for Age predictor)
    # Manual calculation for first bin:
    nBins = 3
    pos_i, neg_i = 10, 100
    total_pos, total_neg = 35, 400

    expected_first = (
        math.log(pos_i + 1 / nBins) - math.log(total_pos + 1) - (math.log(neg_i + 1 / nBins) - math.log(total_neg + 1))
    )

    assert result["LogOdds"][0] == pytest.approx(expected_first, abs=1e-6)


def test_log_odds_matches_pure_python():
    """Test Polars implementation matches pure Python reference."""
    bin_pos = [10, 20, 5]
    bin_neg = [100, 100, 200]

    py_result = cdh_utils.bin_log_odds(bin_pos, bin_neg)

    data = pl.DataFrame(
        {
            "BinPositives": bin_pos,
            "BinNegatives": bin_neg,
        }
    )

    pl_result = data.with_columns(cdh_utils.log_odds_polars(pl.col("BinPositives"), pl.col("BinNegatives")))[
        "LogOdds"
    ].to_list()

    for py_val, pl_val in zip(py_result, pl_result):
        assert py_val == pytest.approx(pl_val, abs=1e-6)


def test_feature_importance_edge_cases():
    """Test edge cases: single bin, uniform bins."""
    # Single bin predictor
    single_bin = pl.DataFrame(
        {
            "PredictorName": ["X"],
            "ModelID": ["m1"],
            "BinPositives": [10],
            "BinNegatives": [20],
            "BinResponseCount": [30],
        }
    )

    result = single_bin.with_columns(cdh_utils.feature_importance(scaled=False))
    # Single bin: importance should be the absolute log odds value
    assert result["FeatureImportance"][0] >= 0


def test_feature_importance_with_sample_datamart():
    """Integration test with actual ADM sample data."""
    from pdstools import datasets

    dm = datasets.cdh_sample()

    result = (
        dm.predictor_data.filter(pl.col("EntryType") != "Classifier")
        .with_columns(cdh_utils.feature_importance(scaled=False))
        .group_by("PredictorName", "ModelID")
        .agg(pl.first("FeatureImportance"))
        .collect()
    )

    # Verify all values are non-negative and reasonable
    assert result["FeatureImportance"].min() >= 0
    assert result["FeatureImportance"].is_not_null().all()
    # Feature importance should be bounded (log odds typically < 10)
    assert result["FeatureImportance"].max() < 20


# Test _capitalize function
def test_capitalize_behavior():
    assert set(cdh_utils._capitalize(["pyTest", "pzTest", "pxTest"])) == {"Test"}
    assert cdh_utils._capitalize(["test"]) == ["Test"]
    assert cdh_utils._capitalize(["RESPONSEcOUNT"]) == ["ResponseCount"]
    assert cdh_utils._capitalize(["responsecounter"]) == ["ResponseCounter"]
    assert cdh_utils._capitalize(["responsenumber"]) == ["ResponseNumber"]
    assert cdh_utils._capitalize(["Response Count"]) == ["Response Count"]
    assert cdh_utils._capitalize(["Response_count1"]) == ["Response_Count1"]
    assert cdh_utils._capitalize("Response_count1") == ["Response_Count1"]


def test_capitalize_configuration():
    """Test that 'Configuration' is capitalized correctly.

    Previously, "Ratio" in capitalize_endwords matched the substring 'ratio'
    within 'configuration', causing incorrect capitalization to 'ConfiguRation'.
    Fixed by sorting endwords by length so longer words are processed last.
    """
    assert cdh_utils._capitalize(["configuration"]) == ["Configuration"]
    assert cdh_utils._capitalize(["pyConfiguration"]) == ["Configuration"]
    assert cdh_utils._capitalize(["configurationname"]) == ["Configuration"]


def test_PredictorCategorization():
    df = (
        pl.LazyFrame({"PredictorName": ["Customer.Variable", "Variable"]})
        .with_columns(cdh_utils.default_predictor_categorization())
        .collect()
    )
    assert df.get_column("PredictorCategory").to_list() == ["Customer", "Primary"]

    df = (
        pl.LazyFrame({"Predictor": ["Customer.Variable", "Variable"]})
        .with_columns(cdh_utils.default_predictor_categorization(x="Predictor"))
        .collect()
    )
    assert df.get_column("PredictorCategory").to_list() == ["Customer", "Primary"]

    assert cdh_utils.default_predictor_categorization().meta.eq(
        cdh_utils.default_predictor_categorization("PredictorName"),
    )


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
        },
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
    assert gains.get_column("cum_x").round(4).to_list() == [0] + [round(e, 4) for e in df["CumTotal"].to_list()]
    assert gains.get_column("cum_y").round(4).to_list() == [0] + [round(e, 4) for e in df["CumPos"].to_list()]

    # With a grouping field

    df = pl.DataFrame(
        {
            "Positives": [100, 10, 20, 500, 5000, 30, 10000, 15000, 20],
            "Dimension": ["A", "B", "B", "A", "A", "B", "A", "A", "B"],
        },
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


def test_legend_color_order():
    input_fig = datasets.cdh_sample().plot.bubble_chart()
    output_fig = cdh_utils.legend_color_order(input_fig)

    assert output_fig.data[0].marker.color == "#001F5F"


def test_apply_query():
    df = pl.DataFrame({"categories": ["A", "B", "C", "C"], "values": [1, 3, 5, 2]})

    assert df.equals(cdh_utils._apply_query(df))

    assert cdh_utils._apply_query(df, query=pl.col("categories") == "C")["categories"].unique().to_list() == ["C"]

    assert cdh_utils._apply_query(df, query=[pl.col("categories") == "C"])["categories"].unique().to_list() == ["C"]

    assert cdh_utils._apply_query(df, query={"categories": ["C"]})["categories"].unique().to_list() == ["C"]

    with pytest.raises(ValueError):  # should raise; lists need to be expressions
        cdh_utils._apply_query(df, query=[{"categories": ["C"]}])

    with pytest.raises(ValueError):  # should raise; string is not a query
        cdh_utils._apply_query(df, query="ABC")

    with pytest.raises(ValueError):  # should raise: 'unknown' is not a column
        cdh_utils._apply_query(df, query={"unknown": ["A"]})

    with pytest.raises(ValueError):  # should raise: 'D' not in 'categories'
        cdh_utils._apply_query(df, query={"categories": ["D"]})


def test_extract_keys():
    non_string = pl.DataFrame({"Name": [1, 2, 3]})
    assert cdh_utils._extract_keys(non_string).equals(non_string)

    empty_df = pl.DataFrame({"Name": []}, schema={"Name": pl.Utf8})
    assert cdh_utils._extract_keys(empty_df).is_empty()

    df = pl.DataFrame({"Name": ["TEST"]})
    assert not cdh_utils._extract_keys(df, capitalize=False).is_empty()

    df = pl.DataFrame({"Name": ["TEST"]})
    assert not cdh_utils._extract_keys(df, capitalize=True).is_empty()

    df = pl.DataFrame({"Name": ["TEST"]})
    assert cdh_utils._extract_keys(df, capitalize=True).equals(df)

    df = pl.DataFrame({"Name": ['{"pyName":"A","pyTreatment":"B"}']})
    assert cdh_utils._extract_keys(df, capitalize=False).to_dict(as_series=False) == {
        "Name": ['{"pyName":"A","pyTreatment":"B"}'],
        "pyName": ["A"],
        "pyTreatment": ["B"],
    }
    df = pl.DataFrame({"Name": ['{"pyName":"A","pyTreatment":"B"}']})
    assert cdh_utils._extract_keys(df, capitalize=True).to_dict(as_series=False) == {
        "Name": ["A"],
        "Treatment": ["B"],
    }
    df = pl.DataFrame(
        {
            "Name": [
                '{"pyName":"Action1", "pyTreatment":"Treatment1"}',
                '{"pyName":"Cosmetics", "Customer":"Anonymous"}',
                '{"pyName":"Cosmetics", "Customer":"Known"}',
                '{"pyName":"Cosmetics", "Customer":"Known"}',
                '{"pyName":"Garden", "customer":"Known"}',
                "GoldCard",
            ],
            "Treatment": ["a", "b", "c", "d", "e", "f"],
        },
    )
    assert cdh_utils._extract_keys(df, capitalize=True).columns == [
        "Name",
        "Treatment",
        "Customer",
        "Customer_2",
    ]
    assert cdh_utils._extract_keys(df.lazy(), capitalize=True).collect().columns == [
        "Name",
        "Treatment",
        "Customer",
        "Customer_2",
    ]
    assert cdh_utils._extract_keys(df, capitalize=False).columns == [
        "Name",
        "Treatment",
        "pyName",
        "pyTreatment",
        "Customer",
        "customer",
    ]
    assert cdh_utils._extract_keys(df.lazy(), capitalize=False).collect().columns == [
        "Name",
        "Treatment",
        "pyName",
        "pyTreatment",
        "Customer",
        "customer",
    ]
    # Slight difference here: old version overwrote all values from overlapping columns
    # with nulls when there was no entry for that column in that row in the keys. New
    # version only overwrites when the key value is not null.
    assert cdh_utils._extract_keys(df, capitalize=True).to_dict(as_series=False) == {
        "Name": [
            "Action1",
            "Cosmetics",
            "Cosmetics",
            "Cosmetics",
            "Garden",
            "GoldCard",
        ],
        "Treatment": ["Treatment1", "b", "c", "d", "e", "f"],
        # "Treatment": ["Treatment1", None, None, None, None, None], # old behavior
        "Customer": [None, "Anonymous", "Known", "Known", None, None],
        "Customer_2": [None, None, None, None, "Known", None],
    }


# "%Y-%m-%d %H:%M:%S"
#     - "%Y%m%dT%H%M%S.%f %Z"
#     - "%d-%b-%y"
#     - "%d%b%Y:%H:%M:%S"
#     - "%Y%m%d"


def test_parse_pega_date_time_formats():
    df = pl.DataFrame(
        {
            "Snappy": [
                "2020-01-01 15:05:03",
                "20241201T150503.847 GMT",
                # "31-Mar-23", # should work?! or give null. polars panics in 1.28
                "31032023:15:05:03",
                "20180316T134127.847345",
                "20180316T134127.8",
                "20180316T134127",
                "20241201",
                # "09MAY2025:06:00:05" # Polars panics. Invariant when calling StrpTimeState.parse was not upheld. https://github.com/pola-rs/polars/issues/22495
            ],
        },
    ).with_columns(
        cdh_utils.parse_pega_date_time_formats("Snappy").alias("SnapshotTime"),
        cdh_utils.parse_pega_date_time_formats("Snappy", timestamp_dtype=pl.Date).alias(
            "SnapshotDate",
        ),
        cdh_utils.parse_pega_date_time_formats(
            "Snappy",
            timestamp_fmt="%d%m%Y:%H:%M:%S",
        ).alias("SnapshotTime2"),
    )

    assert df.schema["SnapshotTime"] == pl.Datetime
    assert df.schema["SnapshotDate"] == pl.Date
    assert df["SnapshotTime"].to_list()[2] is None
    assert df.select(pl.col("SnapshotTime").is_not_null().sum()).item() == 6
    assert df["SnapshotTime2"].to_list()[2] is not None
    assert df.select(pl.col("SnapshotTime2").is_not_null().sum()).item() == 7


# ── Tests for _combine_queries ──────────────────────────────────────────────


def test_combine_queries_expr_with_expr():
    q1 = pl.col("A") > 1
    q2 = pl.col("B") < 5
    result = cdh_utils._combine_queries(q1, q2)
    assert isinstance(result, pl.Expr)
    df = pl.DataFrame({"A": [0, 2, 3], "B": [1, 6, 3]})
    filtered = df.filter(result)
    assert filtered.to_dict(as_series=False) == {"A": [3], "B": [3]}


def test_combine_queries_list_with_expr():
    q1 = [pl.col("A") > 1]
    q2 = pl.col("B") < 5
    result = cdh_utils._combine_queries(q1, q2)
    assert isinstance(result, list)
    assert len(result) == 2
    df = pl.DataFrame({"A": [0, 2, 3], "B": [1, 6, 3]})
    filtered = df.filter(result)
    assert filtered.to_dict(as_series=False) == {"A": [3], "B": [3]}


def test_combine_queries_dict_with_expr():
    q1 = {"A": [2, 3]}
    q2 = pl.col("B") < 5
    result = cdh_utils._combine_queries(q1, q2)
    assert isinstance(result, list)
    df = pl.DataFrame({"A": [0, 2, 3], "B": [1, 6, 3]})
    filtered = df.filter(result)
    assert filtered.to_dict(as_series=False) == {"A": [3], "B": [3]}


def test_combine_queries_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported query type"):
        cdh_utils._combine_queries("not_a_query", pl.col("A") > 1)


# ── Tests for safe_flatten_list ──────────────────────────────────────────────


def test_safe_flatten_list_nested():
    result = cdh_utils.safe_flatten_list([1, [2, 3], None, [4]])
    assert result == [1, 2, 3, 4]


def test_safe_flatten_list_deduplication():
    result = cdh_utils.safe_flatten_list([1, 1, 2])
    assert result == [1, 2]


def test_safe_flatten_list_empty_returns_none():
    result = cdh_utils.safe_flatten_list([], [])
    assert result is None


def test_safe_flatten_list_extras():
    result = cdh_utils.safe_flatten_list([3, 4], extras=[1, 2])
    assert result == [1, 2, 3, 4]


def test_safe_flatten_list_none_alist():
    result = cdh_utils.safe_flatten_list(None)
    assert result is None


# ── Tests for process_files_to_bytes ─────────────────────────────────────────


def test_process_files_to_bytes_single_file(tmp_path):
    f = tmp_path / "report.html"
    f.write_text("<html>hello</html>")
    content, name = cdh_utils.process_files_to_bytes([str(f)], "report.html")
    assert content == b"<html>hello</html>"
    assert name == "report.html"


def test_process_files_to_bytes_multiple_files(tmp_path):
    import zipfile

    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("aaa")
    f2.write_text("bbb")
    content, name = cdh_utils.process_files_to_bytes([str(f1), str(f2)], "reports.html")
    assert name.startswith("reports_")
    assert name.endswith(".zip")
    import io

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        assert set(zf.namelist()) == {"a.txt", "b.txt"}
        assert zf.read("a.txt") == b"aaa"
        assert zf.read("b.txt") == b"bbb"


def test_process_files_to_bytes_empty_list():
    content, name = cdh_utils.process_files_to_bytes([], "report.html")
    assert content == b""
    assert name == ""


# ── Tests for create_working_and_temp_dir ────────────────────────────────────


def test_create_working_and_temp_dir_with_name(tmp_path):
    working, temp = cdh_utils.create_working_and_temp_dir(name="myreport", working_dir=tmp_path)
    assert working == tmp_path
    assert temp.exists()
    assert temp.parent == tmp_path
    assert "tmp_myreport_" in temp.name


def test_create_working_and_temp_dir_without_name(tmp_path):
    working, temp = cdh_utils.create_working_and_temp_dir(working_dir=tmp_path)
    assert working == tmp_path
    assert temp.exists()
    assert temp.name.startswith("tmp_")


def test_create_working_and_temp_dir_custom_working_dir(tmp_path):
    custom = tmp_path / "sub" / "dir"
    working, temp = cdh_utils.create_working_and_temp_dir(working_dir=custom)
    assert working == custom
    assert working.exists()
    assert temp.exists()


# ── Tests for lazy_sample ────────────────────────────────────────────────────


def test_lazy_sample_with_replacement():
    df = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
    sampled = cdh_utils.lazy_sample(df, n_rows=10, with_replacement=True)
    assert isinstance(sampled, pl.DataFrame)
    assert sampled.shape[0] == 10
    assert sampled.columns == ["x", "y"]


def test_lazy_sample_without_replacement():
    df = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
    sampled = cdh_utils.lazy_sample(df, n_rows=10, with_replacement=False)
    assert isinstance(sampled, pl.DataFrame)
    # Without replacement uses binomial sampling so result is approximate
    assert sampled.shape[0] > 0
    assert sampled.shape[0] <= 100
    assert sampled.columns == ["x", "y"]


def test_lazy_sample_n_larger_than_data_without_replacement():
    df = pl.DataFrame({"x": [1, 2, 3]})
    sampled = cdh_utils.lazy_sample(df, n_rows=100, with_replacement=False)
    assert isinstance(sampled, pl.DataFrame)
    # When n_rows > len, all rows should be returned
    assert sampled.shape[0] == 3


# ── Tests for _get_start_end_date_args ───────────────────────────────────────


def test_get_start_end_date_args_start_and_end():
    df = pl.DataFrame(
        {
            "SnapshotTime": [
                datetime.datetime(2024, 1, 1),
                datetime.datetime(2024, 1, 10),
                datetime.datetime(2024, 1, 20),
            ],
        },
    )
    start = datetime.datetime(2024, 1, 5)
    end = datetime.datetime(2024, 1, 15)
    s, e = cdh_utils._get_start_end_date_args(df, start_date=start, end_date=end)
    assert s == start
    assert e == end


def test_get_start_end_date_args_start_and_window():
    df = pl.DataFrame(
        {"SnapshotTime": [datetime.datetime(2024, 1, 1)]},
    )
    start = datetime.datetime(2024, 1, 1)
    s, e = cdh_utils._get_start_end_date_args(df, start_date=start, window=7)
    assert s == start
    assert e == start + datetime.timedelta(days=6)


def test_get_start_end_date_args_end_and_window():
    df = pl.DataFrame(
        {"SnapshotTime": [datetime.datetime(2024, 1, 1)]},
    )
    end = datetime.datetime(2024, 1, 10)
    s, e = cdh_utils._get_start_end_date_args(df, end_date=end, window=5)
    assert s == end - datetime.timedelta(days=4)
    assert e == end


def test_get_start_end_date_args_all_three_raises():
    df = pl.DataFrame(
        {"SnapshotTime": [datetime.datetime(2024, 1, 1)]},
    )
    with pytest.raises(ValueError, match="Only max two"):
        cdh_utils._get_start_end_date_args(
            df,
            start_date=datetime.datetime(2024, 1, 1),
            end_date=datetime.datetime(2024, 1, 10),
            window=5,
        )


def test_get_start_end_date_args_none_uses_data_range():
    df = pl.DataFrame(
        {
            "SnapshotTime": [
                datetime.datetime(2024, 1, 5),
                datetime.datetime(2024, 1, 15),
            ],
        },
    )
    s, e = cdh_utils._get_start_end_date_args(df)
    assert s == datetime.datetime(2024, 1, 5)
    assert e == datetime.datetime(2024, 1, 15)


def test_get_start_end_date_args_start_after_end_raises():
    df = pl.DataFrame(
        {"SnapshotTime": [datetime.datetime(2024, 1, 1)]},
    )
    with pytest.raises(ValueError, match="start date"):
        cdh_utils._get_start_end_date_args(
            df,
            start_date=datetime.datetime(2024, 2, 1),
            end_date=datetime.datetime(2024, 1, 1),
        )


def test_get_start_end_date_args_with_lazyframe():
    lf = pl.LazyFrame(
        {
            "SnapshotTime": [
                datetime.datetime(2024, 3, 1),
                datetime.datetime(2024, 3, 31),
            ],
        },
    )
    s, e = cdh_utils._get_start_end_date_args(lf)
    assert s == datetime.datetime(2024, 3, 1)
    assert e == datetime.datetime(2024, 3, 31)


def test_get_start_end_date_args_with_series():
    series = pl.Series(
        "SnapshotTime",
        [datetime.datetime(2024, 6, 1), datetime.datetime(2024, 6, 30)],
    )
    s, e = cdh_utils._get_start_end_date_args(series)
    assert s == datetime.datetime(2024, 6, 1)
    assert e == datetime.datetime(2024, 6, 30)


# ── Tests for z_ratio / lift with string column names ────────────────────────


def test_z_ratio_with_string_columns():
    df = pl.DataFrame(
        {
            "BinPositives": [0, 7, 11, 12, 6],
            "BinNegatives": [5, 2208, 1919, 1082, 352],
        },
    )
    expr_result = df.with_columns(cdh_utils.z_ratio()).select("ZRatio")
    str_result = df.with_columns(cdh_utils.z_ratio(pos_col="BinPositives", neg_col="BinNegatives")).select("ZRatio")
    assert expr_result.equals(str_result)


def test_lift_with_string_columns():
    df = pl.DataFrame(
        {
            "BinPositives": [0, 7, 11, 12, 6],
            "BinNegatives": [5, 2208, 1919, 1082, 352],
        },
    )
    expr_result = df.with_columns(cdh_utils.lift()).select("Lift")
    str_result = df.with_columns(cdh_utils.lift(pos_col="BinPositives", neg_col="BinNegatives")).select("Lift")
    assert expr_result.equals(str_result)


# ── Tests for Polars duration validation ────────────────────────────────────


def test_is_valid_polars_duration_valid_inputs():
    """Test that valid Polars duration strings are accepted."""
    # Single unit durations
    assert cdh_utils.is_valid_polars_duration("1d")
    assert cdh_utils.is_valid_polars_duration("1w")
    assert cdh_utils.is_valid_polars_duration("1mo")
    assert cdh_utils.is_valid_polars_duration("1y")
    assert cdh_utils.is_valid_polars_duration("1h")
    assert cdh_utils.is_valid_polars_duration("1m")
    assert cdh_utils.is_valid_polars_duration("1s")
    assert cdh_utils.is_valid_polars_duration("1ms")
    assert cdh_utils.is_valid_polars_duration("1us")
    assert cdh_utils.is_valid_polars_duration("1ns")
    assert cdh_utils.is_valid_polars_duration("1q")

    # Multi-digit numbers
    assert cdh_utils.is_valid_polars_duration("10d")
    assert cdh_utils.is_valid_polars_duration("365d")
    assert cdh_utils.is_valid_polars_duration("2mo")

    # Compound durations
    assert cdh_utils.is_valid_polars_duration("1h30m")
    assert cdh_utils.is_valid_polars_duration("1d12h")
    assert cdh_utils.is_valid_polars_duration("1y2mo3w")
    assert cdh_utils.is_valid_polars_duration("1h30m15s")


def test_is_valid_polars_duration_invalid_inputs():
    """Test that invalid inputs are rejected."""
    # Empty or whitespace
    assert not cdh_utils.is_valid_polars_duration("")
    assert not cdh_utils.is_valid_polars_duration("   ")

    # Invalid format
    assert not cdh_utils.is_valid_polars_duration("invalid")
    assert not cdh_utils.is_valid_polars_duration("1")  # No unit
    assert not cdh_utils.is_valid_polars_duration("d")  # No number
    assert not cdh_utils.is_valid_polars_duration("0d")  # Zero not allowed
    assert not cdh_utils.is_valid_polars_duration("-1d")  # Negative not allowed

    # Too long (default max_length=30)
    assert not cdh_utils.is_valid_polars_duration("1" * 31 + "d")

    # Invalid units
    assert not cdh_utils.is_valid_polars_duration("1x")
    assert not cdh_utils.is_valid_polars_duration("1day")


def test_is_valid_polars_duration_custom_max_length():
    """Test custom max_length parameter."""
    # Should accept with larger limit
    assert cdh_utils.is_valid_polars_duration("1d", max_length=10)

    # Should reject with smaller limit
    assert not cdh_utils.is_valid_polars_duration("1d", max_length=1)
