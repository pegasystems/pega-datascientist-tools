import polars as pl
import pytest
from pdstools import ImpactAnalyzer
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
        )
        .filter("Impressed")
        .group_by(_KEYS)
        .agg(Impressions=pl.len().cast(pl.Float64), Accepts=pl.col("Accepted").sum().cast(pl.Float64))
    )


def test_reconciles_exactly_with_ih(raw, ia):
    actual = ia.group_by(_KEYS).agg(pl.col("Impressions", "Accepts").sum()).sort(_KEYS)
    expected = _expected(raw).sort(_KEYS)
    assert actual.equals(expected.select(actual.columns))


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
