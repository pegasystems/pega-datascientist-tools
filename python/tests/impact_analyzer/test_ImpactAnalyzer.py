"""Testing the functionality of the ImpactAnalyzer class"""

import pathlib

import polars as pl
import pytest
from pdstools import ImpactAnalyzer

basePath = pathlib.Path(__file__).parent.parent.parent.parent
PDC_FIXTURE = basePath / "data" / "ia" / "CDH_Metrics_ImpactAnalyzer.json"


@pytest.fixture
def simple_ia():
    return ImpactAnalyzer.from_pdc(str(PDC_FIXTURE))


def test_from_pdc():
    """Test creating an ImpactAnalyzer instance from JSON data"""
    sample_path = str(PDC_FIXTURE)
    analyzer = ImpactAnalyzer.from_pdc(sample_path)

    # Verify the instance was created correctly
    assert isinstance(analyzer, ImpactAnalyzer)
    collected_data = analyzer.ia_data.collect()
    assert collected_data.shape == (24, 8)
    assert collected_data.columns == [
        "SnapshotTime",
        "Channel",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "ValuePerImpression",
        "Pega_ValueLift",
        "Pega_ValueLiftInterval",
    ]

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
        analyzer.summarize_experiments("Channel")
        .collect()
        .filter(Experiment="NBA vs Random Relevant Action")
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
        .filter(Experiment="NBA vs Only Eligibility Criteria")
        .select(pl.col("CTR_Lift").is_null().all())["CTR_Lift"]
        .item()
    )

    # Verify the lift numbers for all channels aggregated
    # Order matches default_ia_experiments (product UI order):
    #   1. NBA vs Random Relevant Action
    #   2. NBA vs Arbitrating by Propensity-only
    #   3. NBA vs Arbitrating with No Levers
    #   4. NBA vs Only Eligibility Criteria  (no data)
    #   5. Adaptive Model Propensity vs Random Propensity
    assert analyzer.summarize_experiments().select(
        pl.col("CTR_Lift").round(6),
    ).collect()["CTR_Lift"].to_list() == [0.002215, 0.003784, 0.002653, None, 0.009563]

    # Value lift isn't exactly the same overall because we can't recalculate it from the current PDC data
    # and the weighted average is perhaps not the same
    assert analyzer.summarize_experiments().select(
        pl.col("Value_Lift").round(1),
    ).collect()["Value_Lift"].to_list() == [
        round(x, 1) if x is not None else x for x in [0.911865, 0.018921, 0.002653, None, 1.032273]
    ]

    # However for a specific channel the Value Lift should be exactly the same, as this is just copied from the data
    assert analyzer.summarize_experiments("Channel").filter(Channel="SMS").select(
        pl.col("Value_Lift").round(6),
    ).collect()["Value_Lift"].to_list() == [
        0.961498,
        0.032556,
        0.012333,
        None,
        1.312399,
    ]


def test_from_pdc_with_custom_reader():
    """Test creating an ImpactAnalyzer instance from JSON data with explicit reader function"""
    import json

    sample_path = str(PDC_FIXTURE)

    def custom_reader(file_path):
        """Custom reader function for testing"""
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    # Test with custom reader
    analyzer = ImpactAnalyzer.from_pdc(sample_path, reader=custom_reader)

    # Verify the instance was created correctly
    assert isinstance(analyzer, ImpactAnalyzer)
    collected_data = analyzer.ia_data.collect()
    assert collected_data.shape == (24, 8)
    assert collected_data.columns == [
        "SnapshotTime",
        "Channel",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "ValuePerImpression",
        "Pega_ValueLift",
        "Pega_ValueLiftInterval",
    ]

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

    trend_df = simple_ia.plot.trend(return_df=True).collect()
    overview_df = simple_ia.plot.overview(return_df=True).collect()
    control_groups_df = simple_ia.plot.control_groups_trend(return_df=True).collect()
    heatmap_df = simple_ia.plot.control_fraction_heatmap(return_df=True).collect()

    assert trend_df.shape == (4, 15)
    assert trend_df.columns == [
        "Experiment",
        "Test",
        "Control",
        "SnapshotTime",
        "Impressions_Test",
        "Accepts_Test",
        "CTR_Test",
        "ValuePerImpression_Test",
        "Impressions_Control",
        "Accepts_Control",
        "CTR_Control",
        "ValuePerImpression_Control",
        "Control_Fraction",
        "CTR_Lift",
        "Value_Lift",
    ]
    assert overview_df.shape == (4, 14)
    assert overview_df.columns == [
        "Experiment",
        "Test",
        "Control",
        "Impressions_Test",
        "Accepts_Test",
        "CTR_Test",
        "ValuePerImpression_Test",
        "Impressions_Control",
        "Accepts_Control",
        "CTR_Control",
        "ValuePerImpression_Control",
        "Control_Fraction",
        "CTR_Lift",
        "Value_Lift",
    ]
    assert control_groups_df.shape == (6, 6)
    assert control_groups_df.columns == [
        "SnapshotTime",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "CTR",
        "ValuePerImpression",
    ]
    assert heatmap_df.shape == (16, 3)
    assert heatmap_df.columns == ["Channel", "Experiment", "Control_Fraction"]

    assert isinstance(simple_ia.plot.trend(), Figure)
    assert isinstance(simple_ia.plot.overview(), Figure)
    assert isinstance(simple_ia.plot.control_groups_trend(), Figure)
    assert isinstance(simple_ia.plot.control_fraction_heatmap(), Figure)


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
    assert "NBA vs Random Relevant Action" in overview_data["Experiment"].to_list()

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

    assert len(overview_fig.data) == 1
    assert len(trend_fig.data) == 4


def test_plot_experiment_color_map():
    """Test the experiment color mapping function"""
    from pdstools.impactanalyzer.Plots import Plots

    # Canonical product UI order, sourced from default_ia_experiments.
    expected_experiments = [
        "NBA vs Random Relevant Action",
        "NBA vs Arbitrating by Propensity-only",
        "NBA vs Arbitrating with No Levers",
        "NBA vs Only Eligibility Criteria",
        "Adaptive Model Propensity vs Random Propensity",
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
        assert collected.shape == (5, 15)
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

        df = ImpactAnalyzer.from_vbd(f.name, return_df=True).collect()
        assert df.shape == (5, 15)
        assert df.columns == [
            "SnapshotTime",
            "ControlGroup",
            "Application",
            "ApplicationVersion",
            "Channel",
            "Issue",
            "Group",
            "Name",
            "Treatment",
            "Impressions",
            "Accepts",
            "ValuePerImpression",
            "AggregateCount",
            "Value",
            "Outcome",
        ]


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

    assert ia.outcome_labels_used == {
        "Call Center/Inbound": {"Accepts": ["Accepted"], "Impressions": ["Impression"]},
        "Web/Inbound": {"Accepts": ["Clicked"], "Impressions": ["Impression"]},
    }


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


# ---------------------------------------------------------------------------
# from_excel tests (Pega Infinity IA Excel export — Data sheet)
# ---------------------------------------------------------------------------

EXCEL_FIXTURE = basePath / "python" / "tests" / "data" / "ia" / "ImpactAnalyzerExport_minimal.xlsx"


@pytest.fixture
def excel_ia():
    return ImpactAnalyzer.from_excel(EXCEL_FIXTURE)


def test_from_excel_returns_instance(excel_ia):
    """from_excel returns an ImpactAnalyzer with a LazyFrame."""
    assert isinstance(excel_ia, ImpactAnalyzer)
    assert excel_ia.ia_data.collect().shape == (56, 10)


def test_from_excel_control_groups(excel_ia):
    """All five canonical control groups + NBA test arm are present; unknown experiment dropped."""
    groups = excel_ia.ia_data.select(pl.col("ControlGroup").unique().sort()).collect()["ControlGroup"].to_list()
    assert groups == [
        "EngagementPolicy",
        "LeverPriority",
        "ModelControl_1",
        "ModelControl_2",
        "NBA",
        "NBAPrioritization",
        "PropensityPriority",
    ]


def test_from_excel_channels_concatenate_direction(excel_ia):
    """Channel is built as '<Channel>/<Direction>' to match the from_vbd convention."""
    channels = sorted(excel_ia.ia_data.select(pl.col("Channel").unique()).collect()["Channel"].to_list())
    assert channels == ["Email/Outbound", "Web/Inbound"]


def test_from_excel_snapshot_dates(excel_ia):
    """Both fixture dates appear in the normalized output."""
    from datetime import datetime

    snapshot_dates = sorted(
        excel_ia.ia_data.select(pl.col("SnapshotTime").unique()).collect()["SnapshotTime"].to_list()
    )
    assert snapshot_dates == [datetime(2026, 1, 1), datetime(2026, 1, 2)]


def test_from_excel_nba_dedup_across_experiments(excel_ia):
    """NBA test arm is collapsed to one row per (date, channel, action) — no 4× double count.

    Fixture has identical NBA Test counts across the four NBA-vs-X experiments
    for every (date, channel, action). For Web/Inbound × Action_A × 2026-01-01,
    NBA = 1000 impressions / 50 accepts (ch_mult=1, act_mult=1).
    """
    from datetime import datetime

    data = excel_ia.ia_data.collect()
    row = data.filter(
        (pl.col("Channel") == "Web/Inbound")
        & (pl.col("Name") == "Action_A")
        & (pl.col("ControlGroup") == "NBA")
        & (pl.col("SnapshotTime") == datetime(2026, 1, 1))
    )
    assert row.height == 1
    assert row["Impressions"].item() == 1000
    assert row["Accepts"].item() == 50


def test_from_excel_exact_control_arm_counts(excel_ia):
    """Each NBA-vs-X experiment's control arm carries its own distinct counts.

    Fixture (Web/Inbound × Action_A × 2026-01-01, ch_mult=1, act_mult=1):
      NBAPrioritization   →  100 imp /  4 acc
      PropensityPriority  →  110 imp /  5 acc
      LeverPriority       →  120 imp /  6 acc
      EngagementPolicy    →  130 imp /  7 acc
    """
    from datetime import datetime

    data = excel_ia.ia_data.collect().filter(
        (pl.col("Channel") == "Web/Inbound")
        & (pl.col("Name") == "Action_A")
        & (pl.col("SnapshotTime") == datetime(2026, 1, 1))
    )
    expected = {
        "NBAPrioritization": (100, 4),
        "PropensityPriority": (110, 5),
        "LeverPriority": (120, 6),
        "EngagementPolicy": (130, 7),
    }
    for cg, (imp, acc) in expected.items():
        row = data.filter(ControlGroup=cg)
        assert row["Impressions"].item() == imp, cg
        assert row["Accepts"].item() == acc, cg


def test_from_excel_model_control_arms_split(excel_ia):
    """AdaptiveModel experiment splits into ModelControl_2 (test) and ModelControl_1 (control)."""
    from datetime import datetime

    data = excel_ia.ia_data.collect().filter(
        (pl.col("Channel") == "Web/Inbound")
        & (pl.col("Name") == "Action_A")
        & (pl.col("SnapshotTime") == datetime(2026, 1, 1))
    )
    mc1 = data.filter(ControlGroup="ModelControl_1")
    mc2 = data.filter(ControlGroup="ModelControl_2")
    # Fixture: test=200/8, control=210/9 (ch_mult=1, act_mult=1)
    assert mc2["Impressions"].item() == 200
    assert mc2["Accepts"].item() == 8
    assert mc1["Impressions"].item() == 210
    assert mc1["Accepts"].item() == 9


def test_from_excel_value_per_impression(excel_ia):
    """ValuePerImpression = ActionValueImpression / Impressions (null when Impressions == 0)."""
    from datetime import datetime

    data = excel_ia.ia_data.collect().filter(
        (pl.col("Channel") == "Web/Inbound")
        & (pl.col("Name") == "Action_A")
        & (pl.col("SnapshotTime") == datetime(2026, 1, 1))
    )
    # NBA: value_imp=200 / impressions=1000 → 0.2
    nba = data.filter(ControlGroup="NBA")
    assert nba["ValuePerImpression"].item() == pytest.approx(0.2)
    # NBAPrioritization control: value_imp=10 / impressions=100 → 0.1
    ctrl = data.filter(ControlGroup="NBAPrioritization")
    assert ctrl["ValuePerImpression"].item() == pytest.approx(0.1)


def test_from_excel_unknown_experiment_dropped(excel_ia):
    """The 'Some Future Unknown Experiment' row is filtered out; no rows leak with that name."""
    # The unknown row's NBA-style numbers (Impressions=1, Accepts=0) would
    # show up only if the row weren't dropped. We assert the absence
    # indirectly via row counts: 2 dates x 2 channels x 2 actions x
    # (5 control arms + 1 deduped NBA + 1 ModelControl_2 test arm - 1 because
    # NBA already counts as the 4 NBA experiments' test arm) = 2*2*2*7 = 56.
    assert excel_ia.ia_data.collect().height == 56


def test_from_excel_warns_on_unknown_experiment(caplog):
    """Loading the fixture emits a warning naming the unknown experiment."""
    import logging

    with caplog.at_level(logging.WARNING, logger="pdstools.impactanalyzer.ImpactAnalyzer"):
        ImpactAnalyzer.from_excel(EXCEL_FIXTURE)
    assert any("Some Future Unknown Experiment" in rec.message for rec in caplog.records)


def test_from_excel_return_df():
    """return_df=True yields a LazyFrame instead of an ImpactAnalyzer."""
    collected = ImpactAnalyzer.from_excel(EXCEL_FIXTURE, return_df=True).collect()
    assert collected.shape == (56, 10)
    assert collected.columns == [
        "SnapshotTime",
        "Channel",
        "Issue",
        "Group",
        "Name",
        "Treatment",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "ValuePerImpression",
    ]


def test_from_excel_missing_required_column_raises(tmp_path):
    """Missing required columns in the Data sheet raise a clear ValueError."""
    bad = tmp_path / "bad.xlsx"
    pl.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Experiment Name": ["NBA vs Random relevant action"],
            # missing Issue / Group / Action / Treatment / etc.
        }
    ).write_excel(bad, worksheet="Data")
    with pytest.raises(ValueError, match="missing required columns"):
        ImpactAnalyzer.from_excel(bad)


def test_from_excel_query_filter():
    """query parameter restricts rows before aggregation."""
    ia = ImpactAnalyzer.from_excel(
        EXCEL_FIXTURE,
        query=pl.col("Channel") == "Web/Inbound",
    )
    channels = ia.ia_data.select(pl.col("Channel").unique()).collect()["Channel"].to_list()
    assert channels == ["Web/Inbound"]


def test_from_excel_outcome_labels_unused(excel_ia):
    """Excel (pre-aggregated) data has no raw outcome labels — outcome_labels_used is None."""
    assert excel_ia.outcome_labels_used is None


# ---------------------------------------------------------------------------
# Schema validation helpers (gold-standard alignment)
# ---------------------------------------------------------------------------


def _make_minimal_ia_lf() -> pl.LazyFrame:
    """Build a minimal valid Impact Analyzer LazyFrame for validation tests."""
    import datetime as _dt

    return pl.LazyFrame(
        {
            "SnapshotTime": [_dt.datetime(2024, 1, 1)],
            "ControlGroup": ["NBA"],
            "Impressions": [100.0],
            "Accepts": [10.0],
            "ValuePerImpression": [1.5],
            "Channel": ["Web"],
        }
    )


def test_validate_ia_data_accepts_valid_frame():
    """_validate_ia_data returns a frame with the schema dtypes applied when all required cols are present."""
    lf = _make_minimal_ia_lf()
    result = ImpactAnalyzer._validate_ia_data(lf)
    df = result.collect()
    assert df.height == 1
    assert set(df.columns) == {
        "SnapshotTime",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "ValuePerImpression",
        "Channel",
    }


def test_validate_ia_data_missing_single_column():
    """Missing a single required column is reported in the error message."""
    lf = _make_minimal_ia_lf().drop("Channel")
    with pytest.raises(ValueError) as exc:
        ImpactAnalyzer._validate_ia_data(lf)
    assert "Channel" in str(exc.value)
    assert "Missing required columns" in str(exc.value)


def test_validate_ia_data_missing_multiple_columns():
    """All missing required columns are listed in the error."""
    lf = _make_minimal_ia_lf().drop("Channel", "Accepts")
    with pytest.raises(ValueError) as exc:
        ImpactAnalyzer._validate_ia_data(lf)
    msg = str(exc.value)
    assert "Channel" in msg
    assert "Accepts" in msg


def test_init_invokes_validation():
    """ImpactAnalyzer(...) raises when required columns are missing."""
    lf = _make_minimal_ia_lf().drop("ValuePerImpression")
    with pytest.raises(ValueError, match="ValuePerImpression"):
        ImpactAnalyzer(lf)


def test_init_pure_no_extra_columns_added():
    """__init__ stores the raw LazyFrame unchanged — no implicit IO or rewrites."""
    lf = _make_minimal_ia_lf()
    ia = ImpactAnalyzer(lf)
    # No extra columns should have been injected by __init__.
    assert ia.ia_data.collect().columns == lf.collect().columns
    assert ia.outcome_labels_used is None


def test_required_columns_constant_matches_init_check():
    """REQUIRED_IA_COLUMNS is the single source of truth used by validation."""
    from pdstools.impactanalyzer.Schema import REQUIRED_IA_COLUMNS

    assert set(REQUIRED_IA_COLUMNS) == {
        "SnapshotTime",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "ValuePerImpression",
        "Channel",
    }


def test_summarize_control_groups_with_single_str_by(simple_ia):
    """A single string `by=` (the branch that previously needed `# type: ignore`) works."""
    result = simple_ia.summarize_control_groups(by="Channel").collect()
    assert "Channel" in result.columns
    assert "ControlGroup" in result.columns
    # Each (Channel, ControlGroup) pair should appear exactly once.
    assert result.height == result.select("Channel", "ControlGroup").unique().height


def test_summarize_control_groups_with_pl_expr_by(simple_ia):
    """A single pl.Expr `by=` is correctly wrapped into a list."""
    result = simple_ia.summarize_control_groups(by=pl.col("Channel")).collect()
    assert "Channel" in result.columns
    assert "ControlGroup" in result.columns


# ---------------------------------------------------------------------------
# Schema dtype contract — every from_* path must land on identical dtypes
# ---------------------------------------------------------------------------


_EXPECTED_IA_DTYPES = {
    "SnapshotTime": pl.Datetime,
    "ControlGroup": pl.Utf8,
    "Impressions": pl.Float64,
    "Accepts": pl.Float64,
    "ValuePerImpression": pl.Float64,
    "Channel": pl.Utf8,
}


def _assert_required_ia_dtypes(ia: ImpactAnalyzer) -> None:
    schema = ia.ia_data.collect_schema()
    for col, expected in _EXPECTED_IA_DTYPES.items():
        actual = schema[col]
        # Datetime carries time-unit / time-zone parameters — compare base type.
        assert actual.base_type() == expected, f"{col}: expected {expected}, got {actual}"


def test_from_pdc_schema_dtypes(simple_ia):
    """from_pdc lands on the ImpactAnalyzerData dtypes."""
    _assert_required_ia_dtypes(simple_ia)


def test_from_excel_schema_dtypes(excel_ia):
    """from_excel lands on the ImpactAnalyzerData dtypes."""
    _assert_required_ia_dtypes(excel_ia)


def test_from_vbd_schema_dtypes(minimal_vbd_parquet):
    """from_vbd lands on the ImpactAnalyzerData dtypes."""
    ia = ImpactAnalyzer.from_vbd(minimal_vbd_parquet)
    _assert_required_ia_dtypes(ia)
