from datetime import date, datetime

import polars as pl
import pytest
from pdstools import ImpactAnalyzer, demo_data
from pdstools.sample_data._ih import IMPRESSION_OUTCOME, POSITIVE_OUTCOME, interaction_history
from pdstools.sample_data._impact import impact_analyzer_data
from pdstools.sample_data._world import sample_world

_KEYS = ["SnapshotTime", "ControlGroup", "Channel", "Name"]


@pytest.fixture(scope="module")
def raw():
    return interaction_history(sample_world(end="canonical"), 100_000).collect()


@pytest.fixture(scope="module")
def ia(raw):
    return impact_analyzer_data(raw.lazy()).collect()


def _expected(raw: pl.DataFrame) -> pl.DataFrame:
    impression = pl.col("pyOutcome") == pl.col("pyChannel").replace_strict(IMPRESSION_OUTCOME)
    accept = pl.col("pyOutcome") == pl.col("pyChannel").replace_strict(POSITIVE_OUTCOME)
    return (
        raw.group_by("ExposureID")
        .agg(
            SnapshotTime=pl.col("pxOutcomeTime").filter(impression).min().dt.truncate("1d"),
            ControlGroup=pl.col("MktValue").first().str.strip_prefix("NBAHealth_"),
            Channel=pl.format("{}/{}", pl.col("pyChannel").first(), pl.col("pyDirection").first()),
            Name=pl.col("pyName").first(),
            Impressed=impression.any(),
            Accepted=accept.any(),
            AcceptedValue=pl.col("Value").filter(accept).first().fill_null(0.0),
        )
        .filter("Impressed")
        .group_by(_KEYS)
        .agg(
            Impressions=pl.len().cast(pl.Float64),
            Accepts=pl.col("Accepted").sum().cast(pl.Float64),
            AcceptedValue=pl.col("AcceptedValue").sum(),
        )
    )


def _assert_reconciles(raw: pl.DataFrame, impact: pl.DataFrame) -> None:
    actual = (
        impact.group_by(_KEYS)
        .agg(
            pl.col("Impressions", "Accepts").sum(),
            AcceptedValue=(pl.col("ValuePerImpression") * pl.col("Impressions")).sum(),
        )
        .sort(_KEYS)
    )
    expected = _expected(raw).sort(_KEYS).select(actual.columns)
    assert actual.drop("AcceptedValue").equals(expected.drop("AcceptedValue"))
    assert actual["AcceptedValue"].to_list() == pytest.approx(expected["AcceptedValue"].to_list(), rel=0, abs=1e-9)


def test_reconciles_exactly_with_ih(raw, ia):
    _assert_reconciles(raw, ia)


def test_delayed_and_repeated_outcomes_count_each_exposure_once():
    timestamp = datetime.fromisoformat
    rows = [
        ("email-open", timestamp("2000-01-01 23:50"), "Email", "Outbound", "Pending", None),
        ("email-open", timestamp("2000-01-01 23:55"), "Email", "Outbound", "Opened", None),
        ("email-accept", timestamp("2000-01-01 23:51"), "Email", "Outbound", "Pending", None),
        ("email-accept", timestamp("2000-01-01 23:56"), "Email", "Outbound", "Opened", None),
        ("email-accept", timestamp("2000-01-02 00:01"), "Email", "Outbound", "Accepted", 100.0),
        ("email-accept", timestamp("2000-01-02 00:02"), "Email", "Outbound", "Conversion", 500.0),
        ("web", timestamp("2000-01-01 12:00"), "Web", "Inbound", "Impression", None),
        ("web", timestamp("2000-01-01 12:01"), "Web", "Inbound", "Clicked", 40.0),
        ("web", timestamp("2000-01-01 12:02"), "Web", "Inbound", "Clicked", 40.0),
        ("call", timestamp("2000-01-01 12:00"), "Call Center", "Outbound", "Pending", None),
        ("call", timestamp("2000-01-01 12:01"), "Call Center", "Outbound", "Accepted", 150.0),
    ]
    events = pl.DataFrame(
        rows,
        schema=["ExposureID", "pxOutcomeTime", "pyChannel", "pyDirection", "pyOutcome", "Value"],
        orient="row",
    ).with_columns(
        pl.lit("NBAHealth_NBA").alias("MktValue"),
        pl.lit("Sales").alias("pyIssue"),
        pl.lit("Cards").alias("pyGroup"),
        pl.lit("Offer").alias("pyName"),
        pl.lit("Treatment").alias("pyTreatment"),
    )
    actual = (
        impact_analyzer_data(events.lazy())
        .select("SnapshotTime", "ControlGroup", "Channel", "Impressions", "Accepts", "ValuePerImpression")
        .sort("Channel")
        .collect()
    )
    expected = pl.DataFrame(
        [
            (timestamp("2000-01-01"), "NBA", "Call Center/Outbound", 1.0, 1.0, 150.0),
            (timestamp("2000-01-01"), "NBA", "Email/Outbound", 2.0, 1.0, 50.0),
            (timestamp("2000-01-01"), "NBA", "Web/Inbound", 1.0, 1.0, 40.0),
        ],
        schema=actual.schema,
        orient="row",
    )

    assert actual.equals(expected)


def test_totals_and_groups(ia):
    assert ia["Impressions"].sum() == 117509
    assert set(ia["ControlGroup"].unique()) == {
        "NBA",
        "EngagementPolicy",
        "LeverPriority",
        "ModelControl_1",
        "ModelControl_2",
        "NBAPrioritization",
        "PropensityPriority",
    }
    assert set(ia["Channel"].unique()) == {"Web/Inbound", "Mobile/Inbound", "Email/Outbound", "Call Center/Outbound"}


def test_value_per_impression_only_from_accepts(ia):
    assert (ia.filter(pl.col("Accepts") == 0)["ValuePerImpression"] == 0).all()
    assert (ia.filter(pl.col("Accepts") > 0)["ValuePerImpression"] > 0).all()


def test_call_center_arms_stay_sparse(ia):
    call_center = ia.filter(pl.col("Channel") == "Call Center/Outbound")
    per_group = call_center.group_by("ControlGroup").agg(pl.col("Impressions").sum())
    assert per_group.filter(pl.col("ControlGroup") != "NBA")["Impressions"].max() < 30


def test_from_mock_data_return_df_matches_analyzer():
    df = ImpactAnalyzer.from_mock_data(n=5_000, end="canonical", return_df=True).collect()
    ia = ImpactAnalyzer.from_mock_data(n=5_000, end="canonical")
    assert ia.ia_data.collect()["Impressions"].sum() == df["Impressions"].sum()
    assert ia.outcome_labels_used["Web/Inbound"] == {"Impressions": ["Impression"], "Accepts": ["Clicked"]}
    summary = ia.summarize_experiments().collect()
    assert summary.height > 0


@pytest.mark.parametrize("end", ["canonical", date(2026, 11, 8)])
def test_mock_data_reconciles_with_the_public_demo_interactions(end):
    ih = (
        demo_data(days=60, end=end, interactions=5_000)
        .ih.data.rename(
            {
                "Channel": "pyChannel",
                "Direction": "pyDirection",
                "Name": "pyName",
                "Outcome": "pyOutcome",
                "OutcomeTime": "pxOutcomeTime",
            },
        )
        .collect()
    )
    impact = ImpactAnalyzer.from_mock_data(days=60, n=5_000, end=end, return_df=True).collect()

    _assert_reconciles(ih, impact)
