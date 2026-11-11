"""ADM datamart view of the shared sample world."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import exp, pi, sin
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

import polars as pl

from ._calendar import DEFAULT_DAYS
from ._random import entity_rng
from ._world import OMNI, SampleWorld, binomial

if TYPE_CHECKING:
    from ._spec import SampleSpec
    from .schema import ModelContext

APPLIES_TO_CLASS = "Data-Decision-Request-Customer"
"""Class the sample's ADM models apply to."""

_ID_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/pegasystems/pega-datascientist-tools/sample-data/adm")
_SNAPSHOT_TIME = "T220349.000 GMT"
_OMNI_CONFIGURATION = "OmniAdaptiveModel"

_LEARNING_POSITIVES = {"NaiveBayes": 150.0, "GradientBoost": 120.0}
_PLATEAU = {
    "Web_Click_Through_Rate_GB": 68.0,
    "Web_Loans_Click_Through_Rate_GB": 66.0,
    "Mobile_Click_Through_Rate": 69.0,
    "Mobile_Click_Through_Rate_GB": 72.0,
    "Email_Click_Through_Rate": 63.0,
    "CallCenterAcceptRateOutbound": 57.0,
    _OMNI_CONFIGURATION: 64.0,
}
_PLATEAU_SD = 4.5
_PLATEAU_RANGE = (53.0, 82.0)
_LOW_SIGNAL_PLATEAU = {("Web", "Risk"): 54.0, ("Email", "Service"): 54.0}
_LOW_SIGNAL_PLATEAU_SD = 1.5
_ACTIVE_SHARE = {"NaiveBayes": (0.20, 0.03, 0.45), "GradientBoost": (0.40, 0.20, 0.80)}
_PREHISTORY_DAYS = (90.0, 270.0)
_SHORT_PREHISTORY_DAYS = (7.0, 28.0)
_SEEDED_CASES = {"stuck": 12, "weak": 12, "overfit": 4}
_SEEDED_MIN_EXPECTED_POSITIVES = 250
_DEFAULT_REPORT_ACTIVE = 30
_NOVELTY_BOOST = 3.0
_NOVELTY_DAYS = 14.0

ZERO_WIDTH_SPACE = "\u200b"
NO_BREAK_SPACE = "\u00a0"

_SCHEMA = {
    "pyModelID": pl.Utf8,
    "pyAppliesToClass": pl.Utf8,
    "pyConfigurationName": pl.Utf8,
    "pyModelTechnique": pl.Utf8,
    "pyIssue": pl.Utf8,
    "pyGroup": pl.Utf8,
    "pyName": pl.Utf8,
    "pyChannel": pl.Utf8,
    "pyDirection": pl.Utf8,
    "pyTreatment": pl.Utf8,
    "pySnapshotTime": pl.Utf8,
    "pyPositives": pl.Float64,
    "pyNegatives": pl.Float64,
    "pyResponseCount": pl.Float64,
    "pyPerformance": pl.Float64,
    "pyActivePredictors": pl.Int64,
    "pyTotalPredictors": pl.Int64,
}


@dataclass(frozen=True)
class AdmModel:
    """One action-level ADM model of the sample world."""

    model_id: str
    """Stable identifier derived from the model context, independent of the seed."""
    context: ModelContext
    """Configuration, action, channel and treatment this model learns for."""
    technique: str
    """``NaiveBayes`` or ``GradientBoost``."""
    issue: str
    group: str
    name: str
    treatment: str | None
    total_predictors: int
    weight: float
    """Relative share of its traffic pool."""
    rate: float
    """Mature weekday success probability."""
    launched: bool = False
    """Whether the action launched during the window, so ADM explores it more at first."""
    naming_case: str | None = None
    """Deliberate naming defect: ``hidden_name``, ``hidden_group`` or ``empty_group``."""


def model_id(configuration: str, action_id: str, channel: str, treatment_id: str | None) -> str:
    """Return the seed-independent ModelID of one model context."""
    return str(uuid5(_ID_NAMESPACE, f"{configuration}|{action_id}|{channel}|{treatment_id or ''}"))


def summary_model_id(configuration: str) -> str:
    """Return the ModelID of an AGB configuration's summary model."""
    return str(uuid5(_ID_NAMESPACE, f"{configuration}|summary"))


@cache
def adm_models(spec: SampleSpec) -> tuple[AdmModel, ...]:
    """Enumerate the action-level ADM models of a specification.

    Includes the deliberate naming defects reviewed by the Health Check:
    two treatments of one Web action carry a zero-width space in their
    action name, one Mobile action's models carry a trailing non-breaking
    space in their group, and three models have an empty group.
    """
    actions = {a.id: a for a in spec.actions}
    treatments = {t.id: t for t in spec.treatments}
    configurations = {c.name: c for c in spec.configurations}
    memberships = {(a.id, m.channel): m for a in spec.actions for m in a.channels}

    full_window = [
        a for a in spec.actions if a.start_day == 0 and _runs_to_end(a.stop_day) and a.model_category is None
    ]
    hidden_name_action = next(a for a in full_window if any(len(m.treatment_ids) == 3 for m in a.channels))
    hidden_name_membership = next(m for m in hidden_name_action.channels if len(m.treatment_ids) == 3)
    hidden_name_treatments = set(
        sorted(hidden_name_membership.treatment_ids, key=lambda t: -treatments[t].volume_share)[1:]
    )
    hidden_group_action = next(
        a for a in full_window if a is not hidden_name_action and any(m.channel == "Mobile" for m in a.channels)
    )
    empty_group = {
        (a.id, a.channels[0].treatment_ids[0])
        for a in [
            *[a for a in reversed(full_window) if a.channels[0].channel == "Web"][:2],
            *[a for a in reversed(full_window) if a.channels[0].channel == "Email"][:1],
        ]
    }

    models = []
    for context in spec.model_contexts:
        action = actions[context.action_id]
        membership = memberships[(context.action_id, context.channel)]
        configuration = configurations[context.configuration]
        if context.treatment_id is None:
            treatment = None
            weight = membership.volume_weight
            rate = membership.baseline_probability
        else:
            treatment = treatments[context.treatment_id]
            weight = membership.volume_weight * treatment.volume_share
            rate = treatment.baseline_probability
        name, group, case = action.name, action.group, None
        if action is hidden_name_action and context.treatment_id in hidden_name_treatments:
            name, case = name + ZERO_WIDTH_SPACE, "hidden_name"
        elif action is hidden_group_action and context.channel == "Mobile":
            group, case = group + NO_BREAK_SPACE, "hidden_group"
        elif (action.id, context.treatment_id) in empty_group and context.configuration != _OMNI_CONFIGURATION:
            group, case = "", "empty_group"
        models.append(
            AdmModel(
                model_id(context.configuration, context.action_id, context.channel, context.treatment_id),
                context,
                configuration.technique,
                action.issue,
                group,
                name,
                None if treatment is None else treatment.name,
                len(configuration.predictor_ids),
                weight,
                rate,
                action.start_day > 0,
                case,
            )
        )
    return tuple(models)


def default_report_model(spec: SampleSpec) -> AdmModel:
    """Return the Mobile NB model used by the default Model Report.

    It is the busiest Mobile Naive Bayes model that exists for the whole
    window, so it is mature with about thirty active predictors.
    """
    candidates = [
        m
        for m in adm_models(spec)
        if m.context.configuration == "Mobile_Click_Through_Rate"
        and m.context.start_day == 0
        and _runs_to_end(m.context.stop_day)
        and m.naming_case is None
    ]
    return max(candidates, key=lambda m: (m.weight, m.model_id))


def _runs_to_end(stop_day: int | None) -> bool:
    return stop_day is None or stop_day >= DEFAULT_DAYS


def _window_bounds(world: SampleWorld, context: ModelContext) -> tuple[int, int]:
    start = world.window_day(context.start_day) if context.start_day > 0 else 0
    stop_day = context.stop_day
    if stop_day is None or _runs_to_end(stop_day):
        return start, world.days
    return start, world.window_day(stop_day)


def _pool(world: SampleWorld, model: AdmModel, day: int) -> tuple[str, float]:
    """Traffic pool of a model on one day and the pool's share of channel volume.

    Multi-channel models learn from the same responses as the channel models,
    so they form a separate pool per channel rather than competing for traffic.
    """
    channel = model.context.channel
    if model.context.configuration == _OMNI_CONFIGURATION:
        return f"{channel}|{OMNI}", 1.0
    if channel == "Mobile" and day >= world.challenger_day:
        return f"{channel}|{model.technique}", 0.5
    return channel, 1.0


def _weight(model: AdmModel, start: int, day: int) -> float:
    if not model.launched:
        return model.weight
    return model.weight * (1 + _NOVELTY_BOOST * exp(-(day - start) / _NOVELTY_DAYS))


def _has_prehistory(model: AdmModel) -> bool:
    return model.context.prehistory and model.context.channel != "Mobile"


@cache
def seeded_cases(world: SampleWorld) -> dict[str, str]:
    """Map ModelIDs of the seeded quality cases to ``stuck``, ``weak`` or ``overfit``.

    Cases are drawn from models with prehistory in valid channels that run
    for the whole window, so their counts make them mature.
    """
    models = adm_models(world.spec)
    initial_totals: dict[str, float] = {}
    for model in models:
        start, stop = _window_bounds(world, model.context)
        if start == 0 and stop > 0:
            pool, _ = _pool(world, model, 0)
            initial_totals[pool] = initial_totals.get(pool, 0.0) + _weight(model, start, 0)

    eligible = sorted(
        (
            m
            for m in models
            if _has_prehistory(m)
            and m.context.channel in ("Web", "Email")
            and _runs_to_end(m.context.stop_day)
            and m.naming_case is None
            and m.model_id != default_report_model(world.spec).model_id
            and (
                world.channel_days(m.context.channel)[0].volume
                / world.scale
                * (world.days + _PREHISTORY_DAYS[0])
                * m.weight
                / initial_totals[_pool(world, m, 0)[0]]
                * m.rate
                >= _SEEDED_MIN_EXPECTED_POSITIVES
            )
        ),
        key=lambda m: (m.weight, m.model_id),
    )
    rng = entity_rng(world.seed, "adm", "seeded-cases")
    quiet = eligible[: len(eligible) // 4]
    overfit = rng.sample(quiet, _SEEDED_CASES["overfit"])
    rest = [m for m in eligible if m not in overfit]
    chosen = rng.sample(rest, _SEEDED_CASES["stuck"] + _SEEDED_CASES["weak"])
    cases = {m.model_id: "overfit" for m in overfit}
    cases.update({m.model_id: "stuck" for m in chosen[: _SEEDED_CASES["stuck"]]})
    cases.update({m.model_id: "weak" for m in chosen[_SEEDED_CASES["stuck"] :]})
    return cases


def _pool_totals(world: SampleWorld, models: tuple[AdmModel, ...]) -> list[dict[str, float]]:
    totals: list[dict[str, float]] = [{} for _ in range(world.days)]
    for model in models:
        start, stop = _window_bounds(world, model.context)
        for day in range(start, stop):
            pool, _ = _pool(world, model, day)
            totals[day][pool] = totals[day].get(pool, 0.0) + _weight(model, start, day)
    return totals


def model_snapshots(world: SampleWorld) -> pl.LazyFrame:
    """Sample daily ADM model snapshots in the ``PR_DATA_DM_ADMMART_MDL_FACT`` shape.

    Every model has one row per day from its start. Counts and AUC are
    cumulative: models that existed before the window start with history,
    Mobile models learn from zero, and stopped actions keep their last
    state. Each AGB configuration also emits a daily summary row with empty
    context keys, which pdstools drops on load.

    Parameters
    ----------
    world : SampleWorld
        Shared latent state.

    Returns
    -------
    pl.LazyFrame
        Raw model rows, ready for :class:`pdstools.ADMDatamart`.
    """
    models = adm_models(world.spec)
    cases = seeded_cases(world)
    report_model = default_report_model(world.spec).model_id
    totals = _pool_totals(world, models)
    timestamps = [f"{when:%Y%m%d}{_SNAPSHOT_TIME}" for when in world.dates]
    volumes = {channel: world.channel_days(channel)[0].volume for channel in world.channels if channel != OMNI}
    rows: dict[str, list] = {name: [] for name in _SCHEMA}
    summaries: dict[tuple[str, int], list[float]] = {}

    for model in models:
        context = model.context
        rng = entity_rng(world.seed, "adm", model.model_id)
        case = cases.get(model.model_id)
        start, stop = _window_bounds(world, context)
        if start >= world.days:
            continue
        days = world.channel_days(context.channel)
        feedback = world.feedback(context.channel, model.technique)
        baseline = world.baseline(context.channel)
        low, high = _PLATEAU_RANGE
        context_plateau = _LOW_SIGNAL_PLATEAU.get((context.channel, model.issue))
        mean = _PLATEAU[context.configuration] if context_plateau is None else context_plateau
        deviation = _PLATEAU_SD if context_plateau is None else _LOW_SIGNAL_PLATEAU_SD
        plateau = min(high, max(low, rng.gauss(mean, deviation)))
        median, floor_, ceiling = _ACTIVE_SHARE[model.technique]
        share = min(ceiling, max(floor_, median * exp(rng.gauss(0.0, 0.55))))
        active = round(model.total_predictors * share)
        if model.model_id == report_model:
            active = _DEFAULT_REPORT_ACTIVE
        fixed_auc = {
            "stuck": 50.0,
            "weak": rng.uniform(50.6, 51.8),
            "overfit": rng.uniform(86.0, 92.0),
        }.get(case or "")
        if case == "stuck":
            active = 0
        tau = _LEARNING_POSITIVES[model.technique]
        phase = rng.uniform(0, 2 * pi)

        positives = negatives = 0
        if _has_prehistory(model):
            pool, pool_share = _pool(world, model, start)
            expected = volumes[context.channel] * pool_share * model.weight / totals[start][pool]
            span = _SHORT_PREHISTORY_DAYS if context.channel == "Call Center" else _PREHISTORY_DAYS
            history = round(expected * rng.uniform(*span))
            positives = round(history * model.rate)
            negatives = history - positives

        for day in range(start, world.days):
            if day < stop:
                pool, pool_share = _pool(world, model, day)
                trials = round(volumes[context.channel] * pool_share)
                share_of_pool = _weight(model, start, day) / totals[day][pool]
                responses = binomial(trials, share_of_pool, rng.gauss(0.0, 1.0))
                rate = model.rate * days[day].ctr / baseline
                new_positives = binomial(responses, min(1.0, rate), rng.gauss(0.0, 1.0))
                withheld, burst = feedback[day]
                positives += new_positives
                negatives += 0 if withheld else round((responses - new_positives) * burst)
            learning = 1 - exp(-positives / tau)
            if fixed_auc is not None:
                auc = fixed_auc
            else:
                offset = 0.0 if context.channel == "Mobile" else 0.5 * (days[day].auc - days[0].auc)
                auc = 50 + (plateau - 50) * learning + learning * (0.4 * sin(2 * pi * day / 61 + phase) + offset)
            values = (
                model.model_id,
                APPLIES_TO_CLASS,
                context.configuration,
                model.technique,
                model.issue,
                model.group,
                model.name,
                context.channel,
                context.direction,
                model.treatment,
                timestamps[day],
                float(positives),
                float(negatives),
                float(positives + negatives),
                round(min(100.0, max(50.0, auc)), 4),
                round(active * learning),
                model.total_predictors,
            )
            for column, value in zip(_SCHEMA, values, strict=True):
                rows[column].append(value)
            if model.technique == "GradientBoost":
                summary = summaries.setdefault((context.configuration, day), [0.0, 0.0, 0.0])
                summary[0] += positives
                summary[1] += negatives
                summary[2] += auc * (positives + negatives)

    totals_by_configuration = {c.name: len(c.predictor_ids) for c in world.spec.configurations}
    for (configuration, day), (sum_positives, sum_negatives, weighted_auc) in sorted(summaries.items()):
        summary_responses = sum_positives + sum_negatives
        summary_auc = weighted_auc / summary_responses if summary_responses else 50.0
        summary_values = (
            summary_model_id(configuration),
            APPLIES_TO_CLASS,
            configuration,
            "GradientBoost",
            None,
            None,
            None,
            None,
            None,
            None,
            timestamps[day],
            sum_positives,
            sum_negatives,
            summary_responses,
            round(min(100.0, max(50.0, summary_auc)), 4),
            totals_by_configuration[configuration],
            totals_by_configuration[configuration],
        )
        for column, value in zip(_SCHEMA, summary_values, strict=True):
            rows[column].append(value)

    return pl.LazyFrame(rows, schema=_SCHEMA)
