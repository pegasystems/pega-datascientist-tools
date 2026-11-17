from collections import Counter
from dataclasses import FrozenInstanceError, replace
from math import nan

import pytest
from pdstools.sample_data._spec import DEFAULT_SPEC
from pdstools.sample_data.schema import Distribution


def test_approved_catalog_manifest():
    assert DEFAULT_SPEC.manifest == {
        "issues": 4,
        "groups": 14,
        "actions": 110,
        "treatments": 532,
        "predictors": 267,
        "configurations": 7,
        "stories": 4,
        "model_contexts": 864,
        "contexts_by_configuration": {
            "CallCenterAcceptRateOutbound": 12,
            "Email_Click_Through_Rate": 200,
            "Mobile_Click_Through_Rate": 70,
            "Mobile_Click_Through_Rate_GB": 70,
            "OmniAdaptiveModel": 262,
            "Web_Click_Through_Rate_GB": 226,
            "Web_Loans_Click_Through_Rate_GB": 24,
        },
    }


def test_channel_volumes_and_rates():
    assert [(c.name, c.baseline_probability, c.daily_responses, c.window_responses) for c in DEFAULT_SPEC.channels] == [
        ("Web", 0.03, 150_000, None),
        ("Mobile", 0.02, 60_000, None),
        ("Email", 0.015, 100_000, None),
        ("Call Center", 0.1, None, 700),
    ]


def test_action_volume_weights_by_channel_and_issue():
    expected = {
        "Web": {"Sales": 0.37, "Service": 0.41, "Retention": 0.12, "Risk": 0.10},
        "Mobile": {"Sales": 0.42, "Service": 0.30, "Retention": 0.17, "Risk": 0.11},
        "Email": {"Sales": 0.32, "Service": 0.38, "Retention": 0.19, "Risk": 0.11},
        "Call Center": {"Sales": 0.35, "Service": 0.32, "Retention": 0.23, "Risk": 0.10},
    }
    for channel, targets in expected.items():
        for issue, target in targets.items():
            total = sum(
                membership.volume_weight
                for action in DEFAULT_SPEC.actions
                if action.issue == issue
                for membership in action.channels
                if membership.channel == channel
            )
            assert total == pytest.approx(target, abs=1e-7)


def test_exact_lifecycle_and_channel_memberships():
    actions = DEFAULT_SPEC.actions
    assert sum(a.start_day == 0 for a in actions) == 80
    assert sum(a.start_day > 0 for a in actions) == 30
    stopped = [a for a in actions if a.stop_day is not None and a.stop_day < 182]
    assert len(stopped) == 10
    assert sum(a.stop_day is None or a.stop_day >= 182 for a in actions) == 100
    assert sum(len(a.channels) > 1 for a in actions) == 91
    assert Counter(c.channel for a in actions for c in a.channels) == {
        "Web": 100,
        "Email": 80,
        "Mobile": 70,
        "Call Center": 12,
    }
    assert Counter(c.configuration for c in DEFAULT_SPEC.model_contexts if c.stop_day is None or c.stop_day >= 182) == {
        "CallCenterAcceptRateOutbound": 11,
        "Email_Click_Through_Rate": 185,
        "Mobile_Click_Through_Rate": 62,
        "Mobile_Click_Through_Rate_GB": 62,
        "OmniAdaptiveModel": 237,
        "Web_Click_Through_Rate_GB": 205,
        "Web_Loans_Click_Through_Rate_GB": 22,
    }


def test_exact_treatment_baseline():
    treatment = DEFAULT_SPEC.treatments[0]
    assert (treatment.id, treatment.rate_multiplier, treatment.volume_share, treatment.baseline_probability) == (
        "Sales/CreditCards/EverydayCard/Web/Hero",
        0.85,
        0.5,
        0.01472625,
    )
    action = DEFAULT_SPEC.actions[0]
    assert (action.start_day, action.stop_day) == (0, 91)


def test_configured_predictor_counts_and_loans_subset():
    configurations = {c.name: c for c in DEFAULT_SPEC.configurations}
    assert {name: len(c.predictor_ids) for name, c in configurations.items()} == {
        "Web_Click_Through_Rate_GB": 136,
        "Web_Loans_Click_Through_Rate_GB": 21,
        "Mobile_Click_Through_Rate": 136,
        "Mobile_Click_Through_Rate_GB": 136,
        "Email_Click_Through_Rate": 111,
        "CallCenterAcceptRateOutbound": 101,
        "OmniAdaptiveModel": 151,
    }
    loans = set(configurations["Web_Loans_Click_Through_Rate_GB"].predictor_ids)
    assert loans == {
        "Customer.WebVisitsLast1Day",
        "Customer.WebVisitsLast7Days",
        "Customer.WebVisitsLast30Days",
        "Customer.WebVisitsLast90Days",
        "Customer.HomePageVisitsLast1Day",
        "Customer.HomePageVisitsLast7Days",
        "Customer.HomePageVisitsLast30Days",
        "Customer.HomePageVisitsLast90Days",
        "Customer.TotalActiveMinsWebPagesLast1Day",
        "Customer.TotalActiveMinsWebPagesLast7Days",
        "Customer.AvgActiveMinsWebPagesLast1Day",
        "Customer.AvgActiveMinsWebPagesLast7Days",
        "IH.Web.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
        "IH.Web.Inbound.Clicked.pyHistoricalOutcomeCount",
        "IH.Web.Inbound.NoResponse.pxLastOutcomeTime.DaysSince",
        "IH.Web.Inbound.NoResponse.pyHistoricalOutcomeCount",
        "Param.CreativeFormat",
        "Param.MessageLengthBand",
        "Param.CallToActionStyle",
        "Param.Placement",
        "Param.ContentVariant",
    }
    assert loans < set(configurations["Web_Click_Through_Rate_GB"].predictor_ids)
    assert loans == {p.id for p in DEFAULT_SPEC.predictors if p.loan_allowed}
    excluded = {p.id for p in DEFAULT_SPEC.predictors if not p.baseline_eligible}
    assert len(excluded) == 57
    assert {"Customer.Age", "Customer.Gender", "Customer.HasCriticalIllness"} <= excluded
    assert all(not excluded.intersection(c.predictor_ids) for c in DEFAULT_SPEC.configurations)


def test_external_churn_score_is_excluded_from_the_loans_policy():
    name = "Customer.ChurnRiskModelScore"
    predictor = next(p for p in DEFAULT_SPEC.predictors if p.id == name)
    assert (predictor.family, predictor.data_type, predictor.baseline_eligible, predictor.loan_allowed) == (
        "external_model",
        "numeric",
        True,
        False,
    )
    assert predictor.distribution == Distribution(
        kind="binned_numeric",
        weights=(0.4, 0.25, 0.18, 0.1, 0.07),
        edges=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    for configuration in DEFAULT_SPEC.configurations:
        assert (name in configuration.predictor_ids) == (configuration.role != "single-group specialization")

    loans = next(c for c in DEFAULT_SPEC.configurations if c.role == "single-group specialization")
    unsafe_loans = replace(loans, predictor_ids=(*loans.predictor_ids, name))
    with pytest.raises(ValueError, match="Loans allowlist"):
        replace(
            DEFAULT_SPEC,
            configurations=tuple(unsafe_loans if c == loans else c for c in DEFAULT_SPEC.configurations),
        )


def test_exclusive_routing_and_context_lifetimes():
    contexts = DEFAULT_SPEC.model_contexts
    identities = [(c.configuration, c.action_id, c.channel, c.treatment_id) for c in contexts]
    assert len(set(identities)) == 864
    actions = {a.id: a for a in DEFAULT_SPEC.actions}
    for context in contexts:
        action = actions[context.action_id]
        if context.configuration == "Web_Loans_Click_Through_Rate_GB":
            assert (action.issue, action.group, context.channel) == ("Sales", "Loans", "Web")
        elif context.configuration == "Web_Click_Through_Rate_GB":
            assert (action.issue, action.group) != ("Sales", "Loans")
        if context.configuration == "Mobile_Click_Through_Rate_GB":
            assert context.start_day == max(84, action.start_day)
            assert context.prehistory is False
        if context.configuration == "Mobile_Click_Through_Rate":
            assert context.prehistory is False
        if context.configuration == "OmniAdaptiveModel":
            assert context.treatment_id is None
        if action.start_day > 0:
            assert context.prehistory is False


def test_metadata_is_immutable_and_manifest_is_not_shared():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_SPEC.actions[0].name = "changed"
    with pytest.raises(FrozenInstanceError):
        DEFAULT_SPEC.actions[0].channels[0].volume_weight = 0
    assert hash(DEFAULT_SPEC) == hash(replace(DEFAULT_SPEC))
    manifest = DEFAULT_SPEC.manifest
    manifest["actions"] = 0
    assert DEFAULT_SPEC.manifest["actions"] == 110


def test_rationale_description_is_generated_from_spec():
    description = DEFAULT_SPEC.describe()
    assert "| model_contexts | 864 |" in description
    for records in (
        DEFAULT_SPEC.channels,
        DEFAULT_SPEC.configurations,
        DEFAULT_SPEC.predictor_groups,
        DEFAULT_SPEC.stories,
    ):
        for record in records:
            assert record.name in description
            assert record.rationale in description
            assert ", ".join(record.serves) in description


@pytest.mark.parametrize(
    "collection",
    ["channels", "configurations", "actions", "treatments", "predictors", "predictor_groups", "stories"],
)
def test_reject_duplicate_identifiers(collection):
    rows = getattr(DEFAULT_SPEC, collection)
    with pytest.raises(ValueError, match="identifiers must be unique"):
        replace(DEFAULT_SPEC, **{collection: (*rows, rows[0])})


@pytest.mark.parametrize("changes", [{"rationale": " "}, {"serves": ()}, {"serves": ("",)}])
def test_reject_missing_rationale(changes):
    changed = replace(DEFAULT_SPEC.channels[0], **changes)
    with pytest.raises(ValueError, match="rationale"):
        replace(DEFAULT_SPEC, channels=(changed, *DEFAULT_SPEC.channels[1:]))


@pytest.mark.parametrize(
    "changes",
    [
        {"baseline_probability": nan},
        {"baseline_probability": -0.1},
        {"baseline_probability": 1.1},
        {"daily_responses": None},
        {"window_responses": 700},
        {"daily_responses": 0},
        {"daily_responses": True},
        {"daily_responses": 2.5},
    ],
)
def test_reject_invalid_channel_values(changes):
    changed = replace(DEFAULT_SPEC.channels[0], **changes)
    with pytest.raises(ValueError):
        replace(DEFAULT_SPEC, channels=(changed, *DEFAULT_SPEC.channels[1:]))


def test_reject_unsafe_or_unknown_configuration_predictors():
    configurations = DEFAULT_SPEC.configurations
    for predictor in ["Customer.Age", "UnknownPredictor"]:
        changed = replace(configurations[0], predictor_ids=(predictor,))
        with pytest.raises(ValueError, match="unknown or excluded"):
            replace(DEFAULT_SPEC, configurations=(changed, *configurations[1:]))
    changed = replace(configurations[1], predictor_ids=("Customer.AnnualIncome",))
    with pytest.raises(ValueError, match="Loans allowlist"):
        replace(DEFAULT_SPEC, configurations=(configurations[0], changed, *configurations[2:]))


def test_reject_invalid_configuration_introduction():
    changed = replace(DEFAULT_SPEC.configurations[0], start_day=182)
    with pytest.raises(ValueError, match="introduction day"):
        replace(DEFAULT_SPEC, configurations=(changed, *DEFAULT_SPEC.configurations[1:]))


@pytest.mark.parametrize(
    "changes",
    [
        {"family": "unknown"},
        {"distribution": None},
        {"missing_probability": None},
        {"loan_allowed": True, "baseline_eligible": False},
        {"latent_effect_size": nan},
    ],
)
def test_reject_invalid_predictor_metadata(changes):
    predictors = DEFAULT_SPEC.predictors
    index = next(i for i, p in enumerate(predictors) if p.baseline_eligible)
    changed = replace(predictors[index], **changes)
    with pytest.raises(ValueError):
        replace(DEFAULT_SPEC, predictors=(*predictors[:index], changed, *predictors[index + 1 :]))


def test_reject_excluded_predictor_without_reason():
    predictors = DEFAULT_SPEC.predictors
    changed = replace(predictors[0], exclusion_reason=None)
    with pytest.raises(ValueError, match="exclusion reason"):
        replace(DEFAULT_SPEC, predictors=(changed, *predictors[1:]))


def test_reject_baseline_excluded_loans_predictor():
    predictors = DEFAULT_SPEC.predictors
    changed = replace(predictors[0], loan_allowed=True)
    with pytest.raises(ValueError, match="allowed for Loans but excluded"):
        replace(DEFAULT_SPEC, predictors=(changed, *predictors[1:]))


@pytest.mark.parametrize(
    "distribution",
    [
        Distribution("binned_numeric", ()),
        Distribution("binned_numeric", (0.2, 0.2), (0, 1, 2)),
        Distribution("binned_numeric", (nan, 1), (0, 1, 2)),
        Distribution("binned_numeric", (1.1, -0.1), (0, 1, 2)),
        Distribution("binned_numeric", (0.5, 0.5), (0, 2, 1)),
        Distribution("binned_numeric", (0.5, 0.5), (0, 1)),
        Distribution("binned_numeric", (1,), (0, nan)),
        Distribution("categorical", (1,), values=("same", "same")),
        Distribution("categorical", (0.5, 0.5), values=("same", "same")),
        Distribution("unknown", (1,)),
    ],
)
def test_reject_invalid_distribution(distribution):
    predictor = next(p for p in DEFAULT_SPEC.predictors if p.data_type == "numeric" and p.baseline_eligible)
    with pytest.raises(ValueError):
        DEFAULT_SPEC._validate_distribution(replace(predictor, distribution=distribution))


@pytest.mark.parametrize(
    "changes",
    [{"start_day": -1}, {"stop_day": 0}, {"stop_day": 183}, {"rate_multiplier": nan}, {"channels": ()}],
)
def test_reject_invalid_action(changes):
    changed = replace(DEFAULT_SPEC.actions[0], **changes)
    with pytest.raises(ValueError):
        replace(DEFAULT_SPEC, actions=(changed, *DEFAULT_SPEC.actions[1:]))


@pytest.mark.parametrize(
    "changes",
    [
        {"channel": "Unknown"},
        {"direction": "Outbound"},
        {"configuration": "Unknown"},
        {"configuration": "Web_Loans_Click_Through_Rate_GB"},
        {"challenger_configuration": "Mobile_Click_Through_Rate_GB"},
        {"volume_weight": 0},
        {"baseline_probability": 0},
        {"treatment_ids": ()},
        {"treatment_ids": ("Unknown",)},
    ],
)
def test_reject_invalid_action_membership(changes):
    action = DEFAULT_SPEC.actions[0]
    membership = replace(action.channels[0], **changes)
    changed = replace(action, channels=(membership, *action.channels[1:]))
    with pytest.raises(ValueError):
        replace(DEFAULT_SPEC, actions=(changed, *DEFAULT_SPEC.actions[1:]))


@pytest.mark.parametrize(
    "changes",
    [{"action_id": "Unknown"}, {"baseline_probability": 0}, {"volume_share": 0.2}],
)
def test_reject_inconsistent_treatment(changes):
    changed = replace(DEFAULT_SPEC.treatments[0], **changes)
    with pytest.raises(ValueError):
        replace(DEFAULT_SPEC, treatments=(changed, *DEFAULT_SPEC.treatments[1:]))


def test_reject_orphan_treatment():
    extra = replace(DEFAULT_SPEC.treatments[0], id="UnownedTreatment")
    with pytest.raises(ValueError, match="exactly one"):
        replace(DEFAULT_SPEC, treatments=(*DEFAULT_SPEC.treatments, extra))


def test_exact_story_timings_and_effects():
    assert [
        (s.name, s.start_day, s.duration_days, s.recovery_days, s.ctr_multiplier, s.positive_only)
        for s in DEFAULT_SPEC.stories
    ] == [
        ("email_feed_dip", 112, 14, 21, 0.65, False),
        ("email_positive_only", 14, 7, 7, 1, True),
        ("mobile_positive_only", 49, 7, 7, 1, True),
        ("web_positive_only", 133, 7, 7, 1, True),
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"channel": "Unknown"},
        {"start_day": -1},
        {"duration_days": 0},
        {"recovery_days": -1},
        {"start_day": 181},
        {"ctr_multiplier": nan},
    ],
)
def test_reject_invalid_story(changes):
    changed = replace(DEFAULT_SPEC.stories[0], **changes)
    with pytest.raises(ValueError):
        replace(DEFAULT_SPEC, stories=(changed, *DEFAULT_SPEC.stories[1:]))
