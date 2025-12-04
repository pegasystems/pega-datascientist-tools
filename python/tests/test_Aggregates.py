"""
Testing the functionality of the ADMDatamart Aggregates functions
"""

import pathlib
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.adm.Aggregates import Aggregates

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def dm_aggregates():
    """Fixture to serve as class to call functions from."""
    return Aggregates(
        ADMDatamart.from_ds_export(
            base_path=f"{basePath}/data",
            model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
            predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
        )
    )


# Consider moving into ADMDatamart class as a "from_scratch" method
def modeldata_from_scratch(**overrides):
    defaults = pl.DataFrame(
        {
            "Positives": 0,
            "ResponseCount": 0,
            "SnapshotTime": "20250101",
            "Configuration": "Web_Click_Through_Rate",
            "ModelID": "ModelA",
            "Performance": 0.5,
            "Name": "A",
            "Treatment": "A1",
            "Channel": "Web",
            "Direction": "Inbound",
        }
    )
    df = defaults
    for col in overrides.keys():
        df = df.with_columns(pl.lit(overrides.get(col)).alias(col))
    if len(overrides.keys()):
        df = df.explode(overrides.keys())
    df = df.with_columns(ModelID=pl.format("ModelID_{}_{}", "Name", "Treatment"))

    return df.lazy()


@pytest.fixture
def dm_minimal():
    events = [
        ("20330101", "A", 100, 1000, "Mobile", "Inbound"),
        ("20330101", "B", 100, 1000, "Web", "Inbound"),
        ("20330101", "C", 100, 1000, "Web", "Inbound"),
        ("20330115", "A", 150, 10500, "Mobile", "Inbound"),
        ("20330115", "B", 100, 1000, "Web", "Inbound"),
        ("20330115", "C", 160, 12000, "Web", "Inbound"),
        ("20330201", "B", 150, 1000, "Mobile", "Inbound"),
        ("20330201", "C", 160, 12000, "Web", "Inbound"),
        ("20330201", "E", 100, 1000, "Web", "Inbound"),
        ("20330228", "B", 151, 1100, "Mobile", "Inbound"),
        ("20330228", "C", 300, 15000, "Web", "Inbound"),
        ("20330228", "E", 100, 1000, "Web", "Inbound"),
        ("20330301", "C", 200, 1000, "Mobile", "Inbound"),
        ("20330301", "F", 100, 1000, "Web", "Inbound"),
        ("20330301", "G", 100, 1000, "Web", "Inbound"),
        ("20330331", "C", 400, 2000, "Mobile", "Inbound"),
        ("20330331", "F", 100, 1000, "Web", "Inbound"),
        ("20330331", "G", 300, 4000, "Web", "Inbound"),
    ]

    data = pl.DataFrame(
        modeldata_from_scratch().collect().to_dicts()[0]
        | {
            v: [e[i] for e in events]
            for i, v in enumerate(
                [
                    "SnapshotTime",
                    "Name",
                    "Positives",
                    "ResponseCount",
                    "Channel",
                    "Direction",
                ]
            )
        }
    ).with_columns(ModelID=pl.format("ModelID_{}", "Name"))

    return ADMDatamart(model_df=data.lazy())


def test_init(dm_aggregates):
    assert dm_aggregates


def test_model_summary(dm_aggregates):
    assert dm_aggregates.model_summary().collect().shape[0] == 68
    assert dm_aggregates.model_summary().collect().shape[1] == 20


def test_predictors_overview(dm_aggregates):
    assert dm_aggregates.predictors_overview().collect().height == 1800
    assert dm_aggregates.predictors_overview().collect().width == 13


def test_predictors_global_overview(dm_aggregates):
    assert dm_aggregates.predictors_global_overview().collect().height == 89
    assert dm_aggregates.predictors_global_overview().collect().width == 9


def test_summary_by_channel(dm_aggregates):
    summary_by_channel = dm_aggregates.summary_by_channel().collect()
    assert summary_by_channel.height == 3
    assert summary_by_channel.width == 23
    assert summary_by_channel["ChannelDirection"].to_list() == [
        "Email/Outbound",
        "SMS/Outbound",
        "Web/Inbound",
    ]
    assert summary_by_channel["ChannelDirectionGroup"].to_list() == [
        "E-mail/Outbound",
        "SMS/Outbound",
        "Web/Inbound",
    ]
    assert summary_by_channel["Actions"].to_list() == [24, 27, 19]
    assert summary_by_channel["Positives"].to_list() == [1181, 6367, 4638]
    assert summary_by_channel["Responses"].to_list() == [48603, 74054, 47177]
    assert summary_by_channel["Used Actions"].to_list() == [24, 27, 19]
    assert summary_by_channel["Treatments"].to_list() == [0] * 3
    assert summary_by_channel["Used Treatments"].to_list() == [0] * 3
    assert summary_by_channel["Duration"].to_list() == [18000] * 3
    assert [round(x, 6) for x in summary_by_channel["OmniChannel"].to_list()] == [
        0.604167,
        0.592593,
        0.763158,
    ]
    assert summary_by_channel["isValid"].to_list() == [True, True, True]

    # Force one channel to be invalid
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Positives=pl.when(pl.col.Channel == "SMS")
        .then(pl.lit(0))
        .otherwise("Positives")
    )
    summary_by_channel = dm_aggregates.summary_by_channel().collect()
    assert summary_by_channel["isValid"].to_list() == [True, False, True]

    # Force only one channel to be valid
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Positives=pl.when(pl.col.Channel != "Web")
        .then(pl.lit(0))
        .otherwise("Positives")
    )
    summary_by_channel = dm_aggregates.summary_by_channel().collect()
    assert summary_by_channel["isValid"].to_list() == [False, False, True]
    assert summary_by_channel["OmniChannel"].to_list() == [
        None,
        None,
        0,
    ]


def test_summary_by_channel_timeslices(dm_minimal):
    s1 = dm_minimal.aggregates.summary_by_channel(
        start_date=datetime(2033, 1, 1), end_date=datetime(2033, 1, 31), debug=True
    ).collect()

    assert s1["Actions"].to_list() == [1, 2]
    assert s1["New Actions"].to_list() == [1, 2]
    assert s1["Used Actions"].to_list() == [1, 2]
    assert s1["isValid"].to_list() == [False, True]

    s2 = dm_minimal.aggregates.summary_by_channel(
        start_date=datetime(2033, 2, 1), end_date=datetime(2033, 2, 28)
    ).collect()
    assert s2["Actions"].to_list() == [1, 2]
    assert s2["New Actions"].to_list() == [0, 1]
    assert s2["Used Actions"].to_list() == [1, 2]
    assert s2["isValid"].to_list() == [False, True]


def test_used_actions():
    events = [
        # A1 and A2 are both used in january
        ("20330101", "A", "A1", 100, 1000, "Mobile", "Inbound"),
        ("20330101", "A", "A2", 500, 1000, "Mobile", "Inbound"),
        ("20330115", "A", "A1", 150, 1500, "Mobile", "Inbound"),
        ("20330115", "A", "A2", 600, 1200, "Mobile", "Inbound"),
        # But neither is used in february
        # See issue 370: if there are treatments, with different counts, that should not lead to flagging as used action
        ("20330201", "A", "A1", 150, 1500, "Mobile", "Inbound"),
        ("20330201", "A", "A2", 600, 1200, "Mobile", "Inbound"),
        ("20330228", "A", "A1", 150, 1500, "Mobile", "Inbound"),
        ("20330228", "A", "A2", 600, 1200, "Mobile", "Inbound"),
        # Two are used in march
        ("20330301", "B", "A1", 150, 1500, "Mobile", "Inbound"),
        ("20330301", "A", "A2", 600, 1200, "Mobile", "Inbound"),
        ("20330328", "B", "A1", 160, 1600, "Mobile", "Inbound"),
        ("20330328", "A", "A2", 700, 1400, "Mobile", "Inbound"),
    ]

    data = pl.DataFrame(
        modeldata_from_scratch().collect().to_dicts()[0]
        | {
            v: [e[i] for e in events]
            for i, v in enumerate(
                [
                    "SnapshotTime",
                    "Name",
                    "Treatment",
                    "Positives",
                    "ResponseCount",
                    "Channel",
                    "Direction",
                ]
            )
        }
    ).with_columns(ModelID=pl.format("ModelID_{}_{}", "Name", "Treatment"))

    dm_used = ADMDatamart(model_df=data.lazy())

    summary = dm_used.aggregates.summary_by_channel().collect()
    assert summary["Used Actions"].to_list() == [2]

    summary = dm_used.aggregates.summary_by_channel(every="1mo").collect()
    assert summary["Used Actions"].to_list() == [1, 0, 2]


def test_custom_channel_mapping(dm_aggregates):
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Channel=pl.when(pl.col.Channel == "SMS")
        .then(pl.lit("MyChannel"))
        .otherwise(pl.col.Channel)
        .cast(pl.Categorical)
    )

    summary_by_channel = dm_aggregates.summary_by_channel(
        custom_channels={"MyChannel": "Web"}
    ).collect()
    assert summary_by_channel.height == 3
    assert summary_by_channel.width == 23
    assert summary_by_channel["ChannelDirection"].to_list() == [
        "Email/Outbound",
        "MyChannel/Outbound",
        "Web/Inbound",
    ]
    assert summary_by_channel["ChannelDirectionGroup"].to_list() == [
        "E-mail/Outbound",
        "Web/Outbound",
        "Web/Inbound",
    ]


def test_aggregate_overall_summary(dm_aggregates):
    overall_summary = dm_aggregates.overall_summary().collect()

    assert overall_summary.height == 1
    assert overall_summary.width == 21
    assert overall_summary["Number of Valid Channels"].item() == 3
    assert overall_summary["Actions"].item() == 35
    assert overall_summary["Treatments"].item() == 0

    assert round(overall_summary["OmniChannel"].item(), 5) == 0.65331

    # Force only one channel to be valid
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Positives=pl.when(pl.col.Channel != "SMS")
        .then(pl.lit(0))
        .otherwise("Positives")
    )

    overall_summary = dm_aggregates.overall_summary().collect()

    # print(overall_summary)

    assert overall_summary["Number of Valid Channels"].item() == 1


def test_aggregate_summary_by_channel_and_time(dm_aggregates):
    summary_by_channel = dm_aggregates.summary_by_channel(every="4h").collect()
    assert summary_by_channel.height == 6
    assert summary_by_channel.width == 23
    assert set(summary_by_channel["Responses"].to_list()) == {
        27322.0,
        13062.0,
        45786.0,
        19557.0,
        28230.0,
        10890.0,
    }
    assert [round(x, 6) for x in summary_by_channel["OmniChannel"].to_list()] == [
        0.604167,
        0.604167,
        0.592593,
        0.592593,
        0.763158,
        0.763158,
    ]


def test_overall_summary_2():
    dm = ADMDatamart(modeldata_from_scratch())
    summ = dm.aggregates.overall_summary().collect()
    assert summ.height == 1
    assert summ["Number of Valid Channels"].item() == 0

    dm = ADMDatamart(
        modeldata_from_scratch(Positives=[0, 100, 200], ResponseCount=[0, 500, 1000])
    )
    summ = dm.aggregates.overall_summary().collect()
    assert summ["Number of Valid Channels"].item() == 1


def test_uses_NBAD():
    dm = ADMDatamart(modeldata_from_scratch(Configuration=["MyConfig"]))
    summ = dm.aggregates.overall_summary().collect()
    assert not summ["usesNBAD"].item()

    dm = ADMDatamart(
        modeldata_from_scratch(Configuration=["MyConfig", "Web_Click_Through_Rate"])
    )
    summ = dm.aggregates.overall_summary().collect()
    assert summ["usesNBAD"].item()


def test_model_technique():
    dm = ADMDatamart(modeldata_from_scratch())
    summ = dm.aggregates.overall_summary().collect()
    assert summ["usesAGB"].item() is None

    dm = ADMDatamart(modeldata_from_scratch(ModelTechnique=["NaiveBayes"]))
    summ = dm.aggregates.overall_summary().collect()
    assert not summ["usesAGB"].item()

    dm = ADMDatamart(
        modeldata_from_scratch(
            ModelTechnique=["NaiveBayes", "GradientBoost", "RandomForest"]
        )
    )
    summ = dm.aggregates.overall_summary().collect()
    assert summ["usesAGB"].item()


def test_omnichannel():
    dm = ADMDatamart(
        modeldata_from_scratch(
            Name=["A", "B", "A", "B"],
            Channel=["Mobile", "Mobile", "Web", "Web"],
            Positives=[0, 200, 0, 200],
            ResponseCount=[0, 1000, 0, 1000],
        )
    )
    summ = dm.aggregates.summary_by_channel().collect()
    assert summ["OmniChannel"].to_list() == [1.0, 1.0]

    dm = ADMDatamart(
        modeldata_from_scratch(
            Name=["A", "A", "A", "B"],
            Channel=["Mobile", "Mobile", "Web", "Web"],
            Positives=[0, 200, 0, 200],
            ResponseCount=[0, 1000, 0, 1000],
        )
    )
    summ = dm.aggregates.summary_by_channel().collect()
    assert summ["OmniChannel"].to_list() == [1.0, 0.5]

    dm = ADMDatamart(
        modeldata_from_scratch(
            Name=["A", "A", "A", "B"],
            Channel=["Mobile", "Mobile", "Web", "Web"],
            Positives=[0, 0, 0, 200],
            ResponseCount=[0, 1000, 0, 1000],
        )
    )
    summ = dm.aggregates.summary_by_channel().collect()
    assert summ["OmniChannel"].to_list() == [None, 0.0]


def test_aggregate_overall_summary_by_time(dm_aggregates):
    overall_summary = dm_aggregates.overall_summary(every="1h").collect()
    assert overall_summary.height == 6
    assert overall_summary.width == 21

    # print(dm_aggregates.summary_by_channel(every="1h",debug=True).collect().select("Channel","Direction","Period","Positives","TotalPositives","isValid").to_pandas())

    assert overall_summary["Number of Valid Channels"].to_list() == [2, 3, 3, 3, 3, 3]
    assert overall_summary["Actions"].to_list() == [34, 35, 34, 31, 33, 34]
    assert overall_summary["Treatments"].to_list() == [0] * 6
    assert [round(x, 6) for x in overall_summary["OmniChannel"].to_list()] == [
        0.728745,
        0.635802,
        0.660903,
        0.559028,
        0.646232,
        0.662062,
    ]


def test_overall_summary_timeslices(dm_minimal):
    s1 = dm_minimal.aggregates.overall_summary(
        start_date=datetime(2033, 1, 1), window=timedelta(weeks=4)
    ).collect()

    # print(
    #     dm_minimal.model_data.select(
    #         "SnapshotTime",
    #         # "Channel",
    #         "Name",
    #         # "Treatment",
    #         "IsUpdated",
    #         "LastUpdate",
    #         "ResponseCount",
    #         "Positives",
    #     )
    #     .sort("SnapshotTime","Name")
    #     .collect()
    #     .to_pandas()
    # )

    assert s1["Actions"].item() == 3  # A, B, C
    assert s1["New Actions"].item() == 3
    assert (
        s1["Used Actions"].item() == 3
    )  # B not updated but newly introduced counts as used
    s2 = dm_minimal.aggregates.overall_summary(
        start_date=datetime(2033, 2, 1), window=timedelta(weeks=4), debug=True
    ).collect()
    assert s2["Actions"].item() == 3
    assert s2["New Actions"].item() == 1  # E is new
    assert (
        s2["Used Actions"].item() == 3
    )  # B and C are used, E is not used but introduced new
    s3 = dm_minimal.aggregates.overall_summary(
        start_date=datetime(2033, 2, 1)
    ).collect()
    assert s3["Actions"].item() == 5  # B, C, E, F, G
    assert s3["New Actions"].item() == 3  # E, F, G are new
    assert s3["Used Actions"].item() == 5  # C, B, G updated, E,F new
    s4 = dm_minimal.aggregates.overall_summary(window=31).collect()
    assert s4["Actions"].item() == 3  # C, F, G
    assert s4["New Actions"].item() == 2  # F, G
    assert s4["Used Actions"].item() == 3  # C, F, G
    assert s4["DateRange Min"].item() == date(2033, 3, 1)

    # with pytest.raises(ValueError):
    #     # Can't use all 3 arguments
    #     dm_minimal.aggregates.overall_summary(start_date=datetime(2031, 1, 1), end_date=datetime(2033, 1, 1), window=31).collect()
    # with pytest.raises(ValueError):
    #     # No data left
    #     dm_minimal.aggregates.overall_summary(start_date=datetime(2031, 1, 1), window=31).collect()
    # with pytest.raises(ValueError):
    #     # No data left
    #     dm_minimal.aggregates.overall_summary(start_date=datetime(2031, 1, 1), end_date=datetime(2001, 1, 1)).collect()


def test_summary_by_configuration(dm_aggregates):
    configuration_summary = dm_aggregates.summary_by_configuration().collect()
    assert "usesAGB" in configuration_summary.columns
    assert "usesNBAD" in configuration_summary.columns


def test_new_actions():
    df = modeldata_from_scratch(
        Name=["A", "A", "B", "B", "E", "E", "C", "C"],
        SnapshotTime=[
            "20250101",
            "20250102",
            "20250101",
            "20250102",
            "20250201",
            "20250202",
            "20250215",
            "20250216",
        ],
        Channel=[
            "Mobile",
            "Mobile",
            "Mobile",
            "Mobile",
            "Web",
            "Web",
            "Web",
            "Web",
        ],
        Positives=[0, 200, 200, 200, 0, 200, 0, 200],
        ResponseCount=[0, 1000, 1000, 1000, 0, 1000, 0, 1000],
    )
    dm = ADMDatamart(df)

    agg = dm.aggregates.overall_summary().collect()
    assert agg["New Actions"].item() == 4

    agg = dm.aggregates.overall_summary(every="1w").collect()
    assert agg["New Actions"].to_list() == [2, 1, 1]

    agg = dm.aggregates.summary_by_channel().collect()
    assert agg["New Actions"].to_list() == [2, 2]


def test_new_actions_issue_455():
    # In January we have Card1, Loan1
    # In February we have Credit1 (new), Loan1 (but not updated)
    # In March we have Card1, Card2 (new), Credit1, Loan2 (new)

    model_data = pl.DataFrame(
        {
            "Issue": "Issue1",
            "Group": "Group1",
            "Configuration": "Configuration1",
            "Channel": "Web",
            "Direction": "Inbound",
            "Performance": 0.5,
            "Name": [
                "Card1",
                "Card1",
                "Loan1",
                "Loan1",
                "Credit1",
                "Credit1",
                "Loan1",
                "Loan1",
                "Card1",
                "Card1",
                "Card2",
                "Card2",
                "Credit1",
                "Credit1",
                "Loan2",
                "Loan2",
            ],
            "Positives": [
                100,
                101,
                100,
                101,
                100,
                101,
                101,
                101,
                102,
                103,
                100,
                101,
                102,
                103,
                100,
                101,
            ],
            "ResponseCount": [
                1000,
                1010,
                1000,
                1010,
                1000,
                1010,
                1010,
                1010,
                1020,
                1030,
                1000,
                1010,
                1020,
                1030,
                1000,
                1010,
            ],
            "ModelID": [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 4, 4, 3, 3, 5, 5],
            "SnapshotTime": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-02-01",
                "2023-02-02",
                "2023-02-03",
                "2023-02-04",
                "2023-03-01",
                "2023-03-02",
                "2023-03-03",
                "2023-03-04",
                "2023-03-05",
                "2023-03-06",
                "2023-03-07",
                "2023-03-08",
            ],
        }
    ).with_columns(
        pl.col("SnapshotTime").str.strptime(pl.Date, "%Y-%m-%d"),
    )

    from pdstools import ADMDatamart

    dm = ADMDatamart(model_data.lazy())

    agg = dm.aggregates.overall_summary(every="1mo").collect()
    assert agg["Actions"].to_list() == [2, 2, 4]
    assert agg["Used Actions"].to_list() == [2, 1, 4]
    assert agg["New Actions"].to_list() == [2, 1, 2]  # [0, 1, 2]

    agg = dm.aggregates.overall_summary().collect()
    assert agg["Actions"].item() == 5
    assert agg["Used Actions"].item() == 5
    assert agg["New Actions"].item() == 5
