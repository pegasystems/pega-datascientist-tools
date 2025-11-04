"""
Testing the functionality of the ImpactAnalyzer class
"""

import pathlib

import polars as pl
import pytest
from pdstools import ImpactAnalyzer

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def simple_ia():
    # using the same PDC data for now, consider to change to something better: artificial data or curated manual data
    sample_path = f"{basePath}/data/ia/CDH_Metrics_ImpactAnalyzer.json"
    return ImpactAnalyzer.from_pdc(sample_path)


def test_from_pdc():
    """Test creating an ImpactAnalyzer instance from JSON data"""
    sample_path = f"{basePath}/data/ia/CDH_Metrics_ImpactAnalyzer.json"
    analyzer = ImpactAnalyzer.from_pdc(sample_path)

    # Verify the instance was created correctly
    assert isinstance(analyzer, ImpactAnalyzer)
    assert isinstance(analyzer.ia_data, pl.LazyFrame)

    # Verify the data was loaded correctly
    collected_data = analyzer.ia_data.collect()
    assert collected_data.height > 0
    assert "ControlGroup" in collected_data.columns

    # EngagementPolicy test is inactive, all others should be there
    assert (
        "NBAHealth_EngagementPolicy"
        not in analyzer.ia_data.select(pl.col("ControlGroup").unique())
        .collect()["ControlGroup"]
        .to_list()
    )
    assert analyzer.ia_data.select(pl.col("ControlGroup").unique().sort()).collect()[
        "ControlGroup"
    ].to_list() == [
        "NBAHealth_LeverPriority",
        "NBAHealth_ModelControl_1",
        "NBAHealth_ModelControl_2",
        "NBAHealth_NBA",
        "NBAHealth_NBAPrioritization",
        "NBAHealth_PropensityPriority",
    ]

    # Verify the number of impressions for a specific control group
    assert (
        analyzer.ia_data.filter(Channel="DirectMail")
        .filter(ControlGroup="NBAHealth_NBA")
        .select(pl.col("Impressions"))
        .collect()
        .item()
        == 9985661
    )

    # Verify all channel totals, as these are re-calculated
    agg = analyzer.summarize_control_groups(by=[]).collect()
    assert agg.filter(ControlGroup="NBAHealth_LeverPriority")["Accepts"].item() == 39189

    # Verify re-calculated Lift numbers
    lift_data = (
        ImpactAnalyzer.from_pdc(sample_path, return_input_df=True)
        .filter(ChannelName="Email")
        .filter(ExperimentName="NBAHealth_NBAPrioritization")
        .collect()["EngagementLift"]
        .item()
    )
    assert (
        lift_data == 0.012584196
    )  # this just asserts the above expression works, it is the raw value from the data file
    experiment_data_recalculated = (
        analyzer.summarize_experiments("Channel")
        .collect()
        .filter(Experiment="NBA vs Random")
        .filter(Channel="Email")
    )
    assert round(experiment_data_recalculated["CTR_Lift"].item(), 6) == 0.012584
    assert (
        round(experiment_data_recalculated["Value_Lift"].item(), 6) == 0.924631
    )  # this is currently copied from the data, not re-calculated

    # This experiment is not active, all values should be null
    assert (
        analyzer.summarize_experiments("Channel")
        .collect()
        .filter(Experiment="NBA vs Only Eligibility Rules")
        .select(pl.col("CTR_Lift").is_null().all())["CTR_Lift"]
        .item()
    )

    # Verify the lift numbers for all channels aggregated
    assert analyzer.summarize_experiments().select(
        pl.col("CTR_Lift").round(6)
    ).collect()["CTR_Lift"].to_list() == [0.009563, 0.002653, None, 0.003784, 0.002215]

    # Value lift isn't exactly the same overall because we can't recalculate it from the current PDC data
    # and the weighted average is perhaps not the same
    assert analyzer.summarize_experiments().select(
        pl.col("Value_Lift").round(1)
    ).collect()["Value_Lift"].to_list() == [
        round(x, 1) if x is not None else x
        for x in [1.032273, 0.002653, None, 0.018921, 0.911865]
    ]

    # However for a specific channel the Value Lift should be exactly the same, as this is just copied from the data
    assert analyzer.summarize_experiments("Channel").filter(Channel="SMS").select(
        pl.col("Value_Lift").round(6)
    ).collect()["Value_Lift"].to_list() == [
        1.312399,
        0.012333,
        None,
        0.032556,
        0.961498,
    ]


def test_summarize_control_groups(simple_ia):
    agg = simple_ia.summarize_control_groups(by=[]).collect()
    assert agg.columns == [
        "ControlGroup",
        "Impressions",
        "Accepts",
        "CTR",
        "ValuePerImpression",
    ]


def test_overall_summary(simple_ia):
    summary = simple_ia.overall_summary().collect()

    assert summary.height == 1
    assert summary.width == 10

    # TODO more tests if we get more non-PDC data or have some artificial data


def test_summary_by_channel(simple_ia):
    summary = simple_ia.summary_by_channel().collect()

    assert summary.width == 11
    assert summary.height == 4

    # TODO more tests if we get more non-PDC data or have some artificial data


def test_plots(simple_ia):
    from plotly.graph_objs import Figure

    assert isinstance(simple_ia.plot.trend(return_df=True), pl.LazyFrame)
    assert isinstance(simple_ia.plot.overview(return_df=True), pl.LazyFrame)

    assert isinstance(simple_ia.plot.trend(), Figure)
    assert isinstance(simple_ia.plot.overview(), Figure)
