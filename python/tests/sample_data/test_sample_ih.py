import datetime

import polars as pl
import pytest
from pdstools import IH
from pdstools.sample_data._ih import (
    IMPRESSION_OUTCOME,
    POSITIVE_OUTCOME,
    action_value,
    ia_group,
    ia_shares,
    interaction_history,
)
from pdstools.sample_data._world import sample_world


@pytest.fixture(scope="module")
def world():
    return sample_world(end="canonical")


@pytest.fixture(scope="module")
def raw(world):
    return interaction_history(world, 100_000).collect()


@pytest.fixture(scope="module")
def exposures(raw):
    return raw.group_by("ExposureID").agg(
        pl.col("pyChannel").first(),
        pl.col("pyName").first(),
        pl.col("MktValue").first(),
        pl.col("ExperimentGroup").first(),
        pl.col("pyOutcome"),
    )


def test_canonical_shape(raw):
    assert raw.shape == (126792, 22)
    assert raw["pxInteractionID"].n_unique() == 100000
    assert raw["pySubjectID"].n_unique() == 31715
    shown = raw.filter(pl.col("pyOutcome").is_in(["Impression", "Pending"]))["pxOutcomeTime"].dt.date()
    assert shown.min() == datetime.date(2000, 1, 3)
    assert shown.max() == datetime.date(2000, 7, 2)


def test_exposures_are_spread_across_issues(raw):
    counts = (
        raw.filter(pl.col("pyOutcome").is_in(["Impression", "Pending"]), pl.col("pyChannel") != "Call Center")
        .group_by("pyChannel", "pyIssue")
        .len()
    )
    for channel in ("Web", "Mobile", "Email"):
        issues = {row["pyIssue"]: row["len"] for row in counts.filter(pl.col("pyChannel") == channel).to_dicts()}
        assert min(issues["Sales"], issues["Service"]) > max(issues["Retention"], issues["Risk"])


def test_reproducible_and_seed_sensitive(world, raw):
    again = interaction_history(world, 100_000).collect()
    assert again.equals(raw)
    other = interaction_history(sample_world(seed=7, end="canonical"), 10_000).collect()
    assert not other.head(100).equals(raw.head(100))


def test_one_impression_per_exposure(exposures):
    impressions = exposures.with_columns(
        n=pl.col("pyOutcome").list.eval(pl.element().is_in(["Impression", "Pending"]).cast(pl.Int32)).list.sum()
    )
    assert (impressions["n"] == 1).all()


def test_channel_outcomes(raw):
    outcomes = raw.group_by("pyChannel").agg(pl.col("pyOutcome").unique())
    for channel, seen in outcomes.iter_rows():
        assert IMPRESSION_OUTCOME[channel] in seen
        assert POSITIVE_OUTCOME[channel] in seen


def test_ctr_tracks_world(world, exposures):
    for channel in world.spec.channels:
        rows = exposures.filter(pl.col("pyChannel") == channel.name)
        ctr = rows["pyOutcome"].list.contains(POSITIVE_OUTCOME[channel.name]).mean()
        days = world.channel_days(channel.name)
        expected = sum(d.ctr * d.volume for d in days) / sum(d.volume for d in days)
        assert ctr == pytest.approx(expected, rel=0.25), channel.name


def test_email_opens_precede_accepts(raw):
    email = raw.filter(pl.col("pyChannel") == "Email")
    accepted = email.filter(pl.col("pyOutcome") == "Accepted").select("ExposureID", Accepted="pxOutcomeTime")
    opened = email.filter(pl.col("pyOutcome") == "Opened").select("ExposureID", Opened="pxOutcomeTime")
    joined = accepted.join(opened, on="ExposureID", how="left")
    assert joined["Opened"].null_count() == 0
    assert (joined["Opened"] <= joined["Accepted"]).all()


def test_value_only_on_accepts(raw):
    valued = raw.filter(pl.col("Value").is_not_null())
    assert set(valued["pyOutcome"].unique()) == {"Clicked", "Accepted"}
    assert valued.group_by("pyName").agg(pl.col("Value").n_unique())["Value"].max() == 1
    assert set(valued["Value"].unique()) <= {20.0 + 10.0 * bucket for bucket in range(19)}
    assert action_value("TravelInsurance") in {20.0 + 10.0 * bucket for bucket in range(19)}


def test_conversions_favour_test_group(exposures):
    rates = (
        exposures.group_by("ExperimentGroup")
        .agg(pl.col("pyOutcome").list.contains("Conversion").mean())
        .sort("ExperimentGroup")
    )
    control, test = rates["pyOutcome"].to_list()
    assert test > control > 0

    by_channel = exposures.group_by("pyChannel", "ExperimentGroup").agg(
        pl.col("pyOutcome").list.contains("Conversion").sum().alias("conversions"),
        pl.len().alias("exposures"),
    )
    counts = {
        (row["pyChannel"], row["ExperimentGroup"]): (row["conversions"], row["exposures"])
        for row in by_channel.to_dicts()
    }
    assert counts == {
        ("Call Center", "Conversion-Control"): (3, 199),
        ("Call Center", "Conversion-Test"): (20, 828),
        ("Email", "Conversion-Control"): (23, 6403),
        ("Email", "Conversion-Test"): (127, 25704),
        ("Mobile", "Conversion-Control"): (28, 4710),
        ("Mobile", "Conversion-Test"): (125, 18261),
        ("Web", "Conversion-Control"): (76, 12316),
        ("Web", "Conversion-Test"): (486, 49459),
    }
    for channel in ("Call Center", "Email", "Mobile", "Web"):
        control_conversions, control_exposures = counts[channel, "Conversion-Control"]
        test_conversions, test_exposures = counts[channel, "Conversion-Test"]
        assert 0 < control_conversions / control_exposures < test_conversions / test_exposures


def test_impact_groups_are_shared_and_stable(raw):
    share = ia_shares()
    assert sum(share.values()) == pytest.approx(1.0)
    subjects = (
        raw.filter(pl.col("pyOutcome").is_in(["Impression", "Pending"]))
        .select("pySubjectID", "MktValue", Month=pl.col("pxOutcomeTime").dt.strftime("%Y-%m"))
        .unique()
    )
    assert subjects.group_by("pySubjectID", "Month").len()["len"].max() == 1
    observed = subjects["MktValue"].value_counts(normalize=True)
    for group, fraction in observed.iter_rows():
        assert fraction == pytest.approx(share[group.removeprefix("NBAHealth_")], abs=0.004)


def test_shifted_impact_groups_follow_actual_calendar_months():
    shifted_world = sample_world(days=60, end=datetime.date(2026, 11, 8))
    shifted = interaction_history(shifted_world, 5_000).collect()
    assert interaction_history(shifted_world, 5_000).collect().equals(shifted)

    impressions = shifted.filter(pl.col("pyOutcome").is_in(["Impression", "Pending"]))
    assignments = impressions.select(
        "pySubjectID", "MktValue", Month=pl.col("pxOutcomeTime").dt.strftime("%Y-%m")
    ).unique()
    assert assignments.group_by("pySubjectID", "Month").len()["len"].max() == 1
    for subject, group, month in assignments.iter_rows():
        assert group == f"NBAHealth_{ia_group(shifted_world.seed, subject, month)}"

    subjects = assignments.group_by("pySubjectID").agg(
        months=pl.col("Month").n_unique(), groups=pl.col("MktValue").n_unique()
    )
    assert subjects.filter((pl.col("months") > 1) & (pl.col("groups") > 1)).height > 0

    canonical = interaction_history(sample_world(days=60, end="canonical"), 5_000).collect()
    canonical_groups = (
        canonical.filter(pl.col("pyOutcome").is_in(["Impression", "Pending"]))
        .select("pxInteractionID", "MktValue")
        .unique()
    )
    shifted_groups = impressions.select("pxInteractionID", "MktValue").unique()
    matched = canonical_groups.join(shifted_groups, on="pxInteractionID")
    assert matched.filter(pl.col("MktValue") != pl.col("MktValue_right")).height > 0


def test_repeated_subject_decisions_across_months(raw):
    decisions = raw.filter(
        pl.col("pySubjectID") == "CUST-0000005",
        pl.col("pyOutcome").is_in(["Impression", "Pending"]),
    ).select("pxInteractionID", "pxOutcomeTime", "pyChannel", "pyName", "MktValue")
    assert decisions.rows() == [
        ("100000000000016369", datetime.datetime(2000, 2, 2, 7, 25, 55), "Email", "LoanRefinance", "NBAHealth_NBA"),
        (
            "100000000000017419",
            datetime.datetime(2000, 2, 3, 22, 56, 47),
            "Web",
            "VariableRateMortgage",
            "NBAHealth_NBA",
        ),
        ("100000000000017419", datetime.datetime(2000, 2, 3, 22, 56, 48), "Web", "PersonalLoan", "NBAHealth_NBA"),
        ("100000000000050838", datetime.datetime(2000, 4, 5, 10, 15, 6), "Email", "DigitalCard", "NBAHealth_NBA"),
    ]


def test_model_control_labels(raw):
    labels = raw.select("MktValue", "MktType", "ModelControlGroup").unique()
    mapping = {value: (kind, control) for value, kind, control in labels.iter_rows()}
    assert mapping["NBAHealth_ModelControl_1"] == ("ModelControl", "Control")
    assert mapping["NBAHealth_ModelControl_2"] == ("ModelControl", "Test")
    assert mapping["NBAHealth_NBA"] == (None, "NBA")
    assert mapping["NBAHealth_LeverPriority"] == ("LeverPriority", "NBA")


def test_mobile_challenger_mixes_techniques(world, raw):
    mobile = raw.filter(pl.col("pyChannel") == "Mobile").with_columns(day=pl.col("pxOutcomeTime").dt.date())
    challenger = world.dates[world.challenger_day]
    before = set(mobile.filter(pl.col("day") < challenger)["pyModelTechnique"].unique())
    after = set(mobile.filter(pl.col("day") > challenger)["pyModelTechnique"].unique())
    assert before == {"NaiveBayes"}
    assert after == {"NaiveBayes", "GradientBoost"}

    traffic = (
        mobile.filter(pl.col("day") > challenger, pl.col("pyOutcome").is_in(["Impression", "Pending"]))
        .group_by("pyModelTechnique")
        .len()
    )
    counts = dict(traffic.iter_rows())
    assert counts == {"NaiveBayes": 6262, "GradientBoost": 6160}
    assert counts["GradientBoost"] / sum(counts.values()) == pytest.approx(0.5, abs=0.01)


def test_ih_from_mock_data_wraps_generator(raw):
    ih = IH.from_mock_data(n=2_000, end="canonical")
    data = ih.data.collect()
    assert data.height > 2_000
    assert {"InteractionID", "Channel", "Outcome", "OutcomeTime", "SubjectID"}.issubset(data.columns)
    assert ih.outcome_labels_used["Email/Outbound"]["Impressions"] == ["Pending"]
