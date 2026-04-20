"""Testing the functionality of the ImpactAnalyzer class"""

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
        "EngagementPolicy"
        not in analyzer.ia_data.select(pl.col("ControlGroup").unique()).collect()["ControlGroup"].to_list()
    )
    assert analyzer.ia_data.select(pl.col("ControlGroup").unique().sort()).collect()["ControlGroup"].to_list() == [
        "LeverPriority",
        "ModelControl_1",
        "ModelControl_2",
        "NBA",
        "NBAPrioritization",
        "PropensityPriority",
    ]

    # Verify the number of impressions for a specific control group
    assert (
        analyzer.ia_data.filter(Channel="DirectMail")
        .filter(ControlGroup="NBA")
        .select(pl.col("Impressions"))
        .collect()
        .item()
        == 9985661
    )

    # Verify all channel totals, as these are re-calculated
    agg = analyzer.summarize_control_groups(by=[]).collect()
    assert agg.filter(ControlGroup="LeverPriority")["Accepts"].item() == 39189

    # Verify re-calculated Lift numbers
    lift_data = (
        ImpactAnalyzer.from_pdc(sample_path, return_wide_df=True)
        .filter(ChannelName="Email")
        .filter(ExperimentName="NBAHealth_NBAPrioritization")
        .collect()["EngagementLift"]
        .item()
    )
    assert (
        lift_data == 0.012584196
    )  # this just asserts the above expression works, it is the raw value from the data file
    experiment_data_recalculated = (
        analyzer.summarize_experiments("Channel").collect().filter(Experiment="NBA vs Random").filter(Channel="Email")
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
        pl.col("CTR_Lift").round(6),
    ).collect()["CTR_Lift"].to_list() == [0.009563, 0.002653, None, 0.003784, 0.002215]

    # Value lift isn't exactly the same overall because we can't recalculate it from the current PDC data
    # and the weighted average is perhaps not the same
    assert analyzer.summarize_experiments().select(
        pl.col("Value_Lift").round(1),
    ).collect()["Value_Lift"].to_list() == [
        round(x, 1) if x is not None else x for x in [1.032273, 0.002653, None, 0.018921, 0.911865]
    ]

    # However for a specific channel the Value Lift should be exactly the same, as this is just copied from the data
    assert analyzer.summarize_experiments("Channel").filter(Channel="SMS").select(
        pl.col("Value_Lift").round(6),
    ).collect()["Value_Lift"].to_list() == [
        1.312399,
        0.012333,
        None,
        0.032556,
        0.961498,
    ]


def test_from_pdc_with_custom_reader():
    """Test creating an ImpactAnalyzer instance from JSON data with explicit reader function"""
    import json

    sample_path = f"{basePath}/data/ia/CDH_Metrics_ImpactAnalyzer.json"

    def custom_reader(file_path):
        """Custom reader function for testing"""
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    # Test with custom reader
    analyzer = ImpactAnalyzer.from_pdc(sample_path, reader=custom_reader)

    # Verify the instance was created correctly
    assert isinstance(analyzer, ImpactAnalyzer)
    assert isinstance(analyzer.ia_data, pl.LazyFrame)

    # Verify basic data integrity (should be same as default reader)
    collected_data = analyzer.ia_data.collect()
    assert collected_data.height > 0
    assert "ControlGroup" in collected_data.columns

    # Verify specific data point to ensure reader worked correctly
    assert (
        analyzer.ia_data.filter(Channel="DirectMail")
        .filter(ControlGroup="NBA")
        .select(pl.col("Impressions"))
        .collect()
        .item()
        == 9985661
    )


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
    assert isinstance(simple_ia.plot.control_groups_trend(return_df=True), pl.LazyFrame)

    assert isinstance(simple_ia.plot.trend(), Figure)
    assert isinstance(simple_ia.plot.overview(), Figure)
    assert isinstance(simple_ia.plot.control_groups_trend(), Figure)


def test_plot_layout_and_formatting(simple_ia):
    """Test plot layout improvements and formatting"""
    from plotly.graph_objs import Figure

    # Test overview plot
    overview_fig = simple_ia.plot.overview()
    assert isinstance(overview_fig, Figure)

    # Test trend plot
    trend_fig = simple_ia.plot.trend()
    assert isinstance(trend_fig, Figure)


def test_plot_with_query(simple_ia):
    """Test plot functionality with query parameter"""
    # Test with experiment filter query (using valid column from summarized data)
    query = pl.col("Experiment").str.contains("NBA vs Random")

    # Test overview with query
    overview_data = simple_ia.plot.overview(query=query, return_df=True).collect()
    # Should only contain the filtered experiment
    assert "NBA vs Random" in overview_data["Experiment"].to_list()

    # Test trend with query - using facet parameter to include Channel for filtering
    trend_data = simple_ia.plot.trend(
        facet="Channel",
        query=pl.col("Channel") == "Email",
        return_df=True,
    ).collect()
    assert "Experiment" in trend_data.columns
    assert "CTR_Lift" in trend_data.columns
    if "Channel" in trend_data.columns:
        unique_channels = trend_data["Channel"].unique().to_list()
        assert "Email" in unique_channels

    # Test that plots still render with query
    overview_fig = simple_ia.plot.overview(query=query)
    trend_fig = simple_ia.plot.trend(
        facet="Channel",
        query=pl.col("Channel") == "Email",
    )

    assert overview_fig is not None
    assert trend_fig is not None


def test_plot_experiment_color_map():
    """Test the experiment color mapping function"""
    from pdstools.impactanalyzer.Plots import Plots

    expected_experiments = [
        "Adaptive Models vs Random Propensity",
        "NBA vs No Levers",
        "NBA vs Only Eligibility Rules",
        "NBA vs Propensity Only",
        "NBA vs Random",
    ]

    color_map = Plots._get_experiment_color_map()
    assert color_map == expected_experiments
    assert len(color_map) == 5


def test_plot_with_facet_parameter(simple_ia):
    """Test plot functions with facet parameter"""
    from plotly.graph_objs import Figure

    # Test overview with facet by Channel (facet automatically added to by)
    overview_faceted = simple_ia.plot.overview(facet="Channel")
    assert isinstance(overview_faceted, Figure)

    # Test trend with facet by Channel (facet automatically added to by)
    trend_faceted = simple_ia.plot.trend(facet="Channel")
    assert isinstance(trend_faceted, Figure)

    # Test control groups trend with facet by Channel (facet automatically added to by)
    control_groups_faceted = simple_ia.plot.control_groups_trend(facet="Channel")
    assert isinstance(control_groups_faceted, Figure)

    # Test that plots still work without facet (default behavior)
    overview_no_facet = simple_ia.plot.overview()
    trend_no_facet = simple_ia.plot.trend()
    control_groups_no_facet = simple_ia.plot.control_groups_trend()

    assert isinstance(overview_no_facet, Figure)
    assert isinstance(trend_no_facet, Figure)
    assert isinstance(control_groups_no_facet, Figure)

    # Test that default behavior still works with facet
    trend_with_facet = simple_ia.plot.trend(facet="Channel")
    assert isinstance(trend_with_facet, Figure)

    # Test that facet works correctly
    overview_with_facet = simple_ia.plot.overview(facet="Channel")
    assert isinstance(overview_with_facet, Figure)


def test_plot_facet_edge_cases(simple_ia):
    """Test edge cases for facet parameter handling"""
    # Test with facet provided (overview)
    overview_data = simple_ia.plot.overview(facet="Channel", return_df=True).collect()
    assert "Channel" in overview_data.columns

    # Test with facet provided (trend)
    trend_data = simple_ia.plot.trend(facet="Channel", return_df=True).collect()
    assert "Channel" in trend_data.columns
    assert "SnapshotTime" in trend_data.columns  # default should still be there

    # Test with facet provided (control_groups_trend)
    control_groups_data = simple_ia.plot.control_groups_trend(
        facet="Channel",
        return_df=True,
    ).collect()
    assert "Channel" in control_groups_data.columns

    # Test with facet=None (should work normally)
    overview_no_facet_data = simple_ia.plot.overview(
        facet=None,
        return_df=True,
    ).collect()
    assert "Experiment" in overview_no_facet_data.columns

    # Test facet with trend (should include both facet and default time dimension)
    trend_with_facet = simple_ia.plot.trend(facet="Channel", return_df=True).collect()
    assert "Channel" in trend_with_facet.columns
    assert "SnapshotTime" in trend_with_facet.columns


def test_plot_with_every_parameter(simple_ia):
    """Test plot functions with every parameter for time aggregation"""
    from plotly.graph_objs import Figure

    # Test trend with every parameter only
    trend_with_every = simple_ia.plot.trend(every="1mo")
    assert isinstance(trend_with_every, Figure)

    # Test control groups trend with every parameter only
    control_groups_with_every = simple_ia.plot.control_groups_trend(every="1w")
    assert isinstance(control_groups_with_every, Figure)

    # CRITICAL TEST: Test the combination of facet and every parameters
    # This was failing before the fix with TypeError
    trend_facet_and_every = simple_ia.plot.trend(facet="Channel", every="1mo")
    assert isinstance(trend_facet_and_every, Figure)

    control_groups_facet_and_every = simple_ia.plot.control_groups_trend(
        facet="Channel",
        every="1w",
    )
    assert isinstance(control_groups_facet_and_every, Figure)

    # Test return_df functionality with facet + every combination
    trend_data = simple_ia.plot.trend(
        facet="Channel",
        every="1mo",
        return_df=True,
    ).collect()
    assert "Channel" in trend_data.columns
    assert "SnapshotTime" in trend_data.columns
    assert "Experiment" in trend_data.columns

    control_groups_data = simple_ia.plot.control_groups_trend(
        facet="Channel",
        every="1w",
        return_df=True,
    ).collect()
    assert "Channel" in control_groups_data.columns
    assert "SnapshotTime" in control_groups_data.columns
    assert "ControlGroup" in control_groups_data.columns


def _make_vbd_parquet(outcomes_by_channel: dict[str, list]) -> str:
    """Helper: write minimal VBD parquet for testing outcome label scenarios.

    outcomes_by_channel: {"Web/Inbound": ["Impression", "Accepted"], "Email/Outbound": ["Sent", "Click"]}
    Each channel gets 100 aggregated rows per outcome.
    """
    import tempfile

    rows = []
    for channel_dir, outcomes in outcomes_by_channel.items():
        channel, direction = channel_dir.split("/")
        for outcome in outcomes:
            rows.append(
                {
                    "pyOutcomeTime": "2024-01-01 10:00:00",
                    "pyChannel": channel,
                    "pyDirection": direction,
                    "pxMktValue": None,
                    "pyReason": "Test",
                    "pxMktType": None,
                    "pyApplication": "App1",
                    "pxApplicationVersion": "1.0",
                    "pyIssue": "Sales",
                    "pyGroup": "Cards",
                    "pyName": "GoldCard",
                    "pyTreatment": "Default",
                    "pyOutcome": outcome,
                    "pxAggregateCount": 100,
                    "pyValue": 0.0,
                }
            )

    df = pl.DataFrame(rows).cast({"pxMktValue": pl.String, "pxMktType": pl.String})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.write_parquet(f.name)
        return f.name


def test_from_vbd_default_outcome_labels():
    """Default outcome labels count standard outcomes correctly per channel."""
    path = _make_vbd_parquet(
        {
            "Web/Inbound": ["Impression", "Clicked", "UnknownOutcome"],
        }
    )
    ia = ImpactAnalyzer.from_vbd(path)
    collected = ia.ia_data.collect()
    # Web channel-aware defaults: "Impression" → impressions; "Clicked" → accepts
    assert collected["Impressions"].sum() == 100
    assert collected["Accepts"].sum() == 100


def test_from_vbd_global_outcome_labels_override():
    """Custom global outcome_labels replace the class defaults entirely."""
    path = _make_vbd_parquet(
        {
            "Web/Inbound": ["Sent", "Click"],
        }
    )
    # Without override: Sent and Click are not in defaults → zero counts
    ia_default = ImpactAnalyzer.from_vbd(path)
    assert ia_default.ia_data.collect()["Impressions"].sum() == 0

    # With global override
    ia_custom = ImpactAnalyzer.from_vbd(
        path,
        outcome_labels={"Impressions": ["Sent"], "Accepts": ["Click"]},
    )
    collected = ia_custom.ia_data.collect()
    assert collected["Impressions"].sum() == 100
    assert collected["Accepts"].sum() == 100


def test_from_vbd_per_channel_outcome_labels():
    """Per-channel outcome_labels applied per channel; others fall back to class defaults."""
    path = _make_vbd_parquet(
        {
            "Email/Outbound": ["Sent", "Click"],  # custom channel
            "Web/Inbound": ["Impression", "Accepted"],  # uses class defaults
        }
    )
    per_channel = {
        "Email/Outbound": {
            "Impressions": ["Sent"],
            "Accepts": ["Click"],
        }
        # Web/Inbound not configured → class defaults apply
    }
    ia = ImpactAnalyzer.from_vbd(path, outcome_labels=per_channel)
    collected = ia.ia_data.collect()

    email_rows = collected.filter(pl.col("Channel") == "Email/Outbound")
    web_rows = collected.filter(pl.col("Channel") == "Web/Inbound")

    assert email_rows["Impressions"].sum() == 100  # "Sent" mapped
    assert email_rows["Accepts"].sum() == 100  # "Click" mapped
    assert web_rows["Impressions"].sum() == 100  # "Impression" via defaults
    assert web_rows["Accepts"].sum() == 100  # "Accepted" via defaults


def test_from_vbd_per_channel_fallback_for_unconfigured_channel():
    """A channel not in per-channel config uses class-level defaults."""
    path = _make_vbd_parquet(
        {
            "SMS/Outbound": ["Impression", "Accepted"],
        }
    )
    # Configure only Email, not SMS
    per_channel = {
        "Email/Outbound": {"Impressions": ["Sent"], "Accepts": ["Click"]},
    }
    ia = ImpactAnalyzer.from_vbd(path, outcome_labels=per_channel)
    sms_rows = ia.ia_data.collect().filter(pl.col("Channel") == "SMS/Outbound")
    # SMS falls back to class defaults: "Impression" and "Accepted"
    assert sms_rows["Impressions"].sum() == 100
    assert sms_rows["Accepts"].sum() == 100


def test_from_vbd():
    """Test from_vbd constructor and methods"""
    import tempfile

    from plotly.graph_objs import Figure

    vbd_data = pl.DataFrame(
        {
            "pyOutcomeTime": ["2024-01-01 10:00:00"] * 4 + ["2024-01-02 10:00:00"] * 2,
            "pyChannel": ["Web", "Web", "Web", "Email", "Web", "Email"],
            "pyDirection": [
                "Inbound",
                "Inbound",
                "Inbound",
                "Outbound",
                "Inbound",
                "Outbound",
            ],
            "pxMktValue": [
                None,
                "NBAHealth_NBAPrioritization",
                "NBAHealth_NBAPrioritization",
                None,
                None,
                None,
            ],
            "pyReason": ["Test"] * 6,
            "pxMktType": [
                None,
                "NBAPrioritization",
                "NBAPrioritization",
                None,
                None,
                None,
            ],
            "pyApplication": ["App1"] * 6,
            "pxApplicationVersion": ["1.0"] * 6,
            "pyIssue": ["Sales"] * 6,
            "pyGroup": ["Cards"] * 6,
            "pyName": ["GoldCard"] * 6,
            "pyTreatment": ["Default"] * 6,
            "pyOutcome": [
                "Impression",
                "Impression",
                "Accepted",
                "Impression",
                "Impression",
                "Impression",
            ],
            "pxAggregateCount": [100, 50, 5, 200, 150, 250],
            "pyValue": [0.0, 0.0, 25.0, 0.0, 0.0, 0.0],
        },
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        vbd_data.write_parquet(f.name)

        ia = ImpactAnalyzer.from_vbd(f.name)
        assert isinstance(ia, ImpactAnalyzer)

        collected = ia.ia_data.collect()
        assert collected.height > 0
        assert "NBA" in collected["ControlGroup"].to_list()

        control_groups = ia.summarize_control_groups().collect()
        assert "Pega_ValueLift" not in control_groups.columns

        experiments = ia.summarize_experiments().collect()
        assert "CTR_Lift" in experiments.columns
        assert "Value_Lift" in experiments.columns

        experiments_by_channel = ia.summarize_experiments("Channel").collect()
        assert "Channel" in experiments_by_channel.columns

        overall = ia.overall_summary().collect()
        assert overall.height == 1

        by_channel = ia.summary_by_channel().collect()
        assert "Channel" in by_channel.columns

        assert isinstance(ia.plot.overview(), Figure)
        assert isinstance(ia.plot.trend(facet="Channel", every="1d"), Figure)

        df = ImpactAnalyzer.from_vbd(f.name, return_df=True)
        assert isinstance(df, pl.LazyFrame)


@pytest.fixture
def minimal_vbd_parquet(tmp_path):
    """Minimal VBD data with Web and Call Center channels."""
    data = pl.DataFrame(
        {
            "outcometime": [
                "20240115T120000.000 GMT",
                "20240115T120000.000 GMT",
                "20240115T120000.000 GMT",
                "20240115T120000.000 GMT",
                "20240115T120000.000 GMT",
                "20240115T120000.000 GMT",
            ],
            "mktvalue": [
                "NBAHealth_NBA",
                "NBAHealth_NBA",
                "NBAHealth_NBAPrioritization",
                "NBAHealth_NBAPrioritization",
                "NBAHealth_NBA",
                "NBAHealth_NBA",
            ],
            "application": ["App"] * 6,
            "applicationversion": ["1.0"] * 6,
            "channel": ["Web", "Web", "Web", "Web", "Call Center", "Call Center"],
            "direction": ["Inbound", "Inbound", "Inbound", "Inbound", "Inbound", "Inbound"],
            "issue": ["Sales"] * 6,
            "group": ["Cards"] * 6,
            "name": ["Action1"] * 6,
            "treatment": ["T1"] * 6,
            "aggregatecount": [1000, 50, 900, 40, 800, 30],
            "value": [0.0, 5000.0, 0.0, 4500.0, 0.0, 3000.0],
            "outcome": [
                "Impression",
                "Clicked",
                "Impression",
                "Clicked",
                "Impression",
                "Accepted",
            ],
        }
    )
    path = tmp_path / "vbd_test.parquet"
    data.write_parquet(path)
    return str(path)


def test_from_vbd_sets_outcome_labels_used(minimal_vbd_parquet):
    """from_vbd always sets outcome_labels_used — no implicit defaults."""
    ia = ImpactAnalyzer.from_vbd(minimal_vbd_parquet)

    assert hasattr(ia, "outcome_labels_used")
    assert ia.outcome_labels_used is not None
    assert isinstance(ia.outcome_labels_used, dict)


def test_from_vbd_resolves_channel_aware_defaults(minimal_vbd_parquet):
    """Web channel uses Clicked; Call Center uses Accepted."""
    ia = ImpactAnalyzer.from_vbd(minimal_vbd_parquet)

    assert "Web/Inbound" in ia.outcome_labels_used
    assert ia.outcome_labels_used["Web/Inbound"]["Impressions"] == ["Impression"]
    assert ia.outcome_labels_used["Web/Inbound"]["Accepts"] == ["Clicked"]

    assert "Call Center/Inbound" in ia.outcome_labels_used
    assert ia.outcome_labels_used["Call Center/Inbound"]["Impressions"] == ["Impression"]
    assert ia.outcome_labels_used["Call Center/Inbound"]["Accepts"] == ["Accepted"]


def test_from_vbd_explicit_labels_preserved(minimal_vbd_parquet):
    """Explicitly provided outcome_labels are stored as-is in outcome_labels_used."""
    custom = {
        "Web/Inbound": {"Impressions": ["Impression"], "Accepts": ["Clicked"]},
    }
    ia = ImpactAnalyzer.from_vbd(minimal_vbd_parquet, outcome_labels=custom)

    assert ia.outcome_labels_used == custom


def test_from_pdc_does_not_set_outcome_labels_used(simple_ia):
    """PDC instances have no outcome_labels_used (pre-aggregated data, no raw outcomes)."""
    assert simple_ia.outcome_labels_used is None
