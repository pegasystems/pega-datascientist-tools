"""Interaction History view of the shared sample world.

Each interaction is one decision for one subject in one channel. The
decision draws candidate actions active that day, applies synthetic
engagement policies and arbitrates them according to the subject's Impact
Analyzer control group. Every offered action is an exposure with its own
outcomes, so Impact Analyzer counts and score calibration can be derived
from the same rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from functools import cache
from math import exp
from typing import TYPE_CHECKING

import polars as pl

from ._adm import _runs_to_end
from ._random import entity_rng, entity_seed

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

    from ._world import SampleWorld
    from .schema import Action, ActionChannel, Treatment

IA_TARGET_FRACTION = 0.01
"""Configured target of each Impact Analyzer control group."""
IA_GROUPS = ("EngagementPolicy", "ModelControl", "LeverPriority", "PropensityPriority", "NBAPrioritization")
"""Active Impact Analyzer experiment types, in stock hash-range order."""
POSITIVE_OUTCOME = {"Web": "Clicked", "Mobile": "Clicked", "Email": "Accepted", "Call Center": "Accepted"}
"""Channel-specific positive engagement outcome."""
IMPRESSION_OUTCOME = {"Web": "Impression", "Mobile": "Impression", "Email": "Pending", "Call Center": "Pending"}
"""Outcome recorded when an action is offered."""
POLICY_PASS = {"eligibility": 0.95, "applicability": 0.85, "suitability": 0.9}
"""Probability that an action passes each synthetic engagement policy for a subject."""

_CALL_CENTER_SHARE = 0.01
_CANDIDATES = 8
_SECOND_OFFER = {"Web": 0.3, "Mobile": 0.2}
_AFFINITY_SIGMA = {"Web": 0.9, "Mobile": 0.8, "Email": 0.6, "Call Center": 0.4}
_CONVERSION = {"Conversion-Test": 0.3, "Conversion-Control": 0.2}
_CONVERSION_CONTROL_SHARE = 0.2
_SUBJECTS_PER_INTERACTION = 1 / 3
_FIRST_SECOND = 6 * 3600
_LAST_SECOND = 23 * 3600

_SCHEMA = pl.Schema(
    {
        "pxInteractionID": pl.Utf8(),
        "ExposureID": pl.Utf8(),
        "pySubjectID": pl.Utf8(),
        "pyChannel": pl.Utf8(),
        "pyDirection": pl.Utf8(),
        "pyIssue": pl.Utf8(),
        "pyGroup": pl.Utf8(),
        "pyName": pl.Utf8(),
        "pyTreatment": pl.Utf8(),
        "pyModelTechnique": pl.Utf8(),
        "ExperimentGroup": pl.Utf8(),
        "MktValue": pl.Utf8(),
        "MktType": pl.Utf8(),
        "ModelControlGroup": pl.Utf8(),
        "pxOutcomeTime": pl.Datetime("us"),
        "pyOutcome": pl.Utf8(),
        "pyPropensity": pl.Float64(),
        "ModelPropensity": pl.Float64(),
        "RandomPropensity": pl.Float64(),
        "DecisionPropensity": pl.Float64(),
        "OpenPropensity": pl.Float64(),
        "Value": pl.Float64(),
    },
)


def stable_bucket(salt: str, action_id: str, buckets: int) -> int:
    """Return a seed-independent bucket of an action, stable across processes.

    Parameters
    ----------
    salt : str
        Purpose of the bucket, such as ``"value:"``.
    action_id : str
        Clean action identifier.
    buckets : int
        Number of buckets.

    Returns
    -------
    int
        Bucket between 0 and ``buckets - 1``.
    """
    digest = hashlib.sha256((salt + action_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def action_value(action_id: str) -> float:
    """Synthetic value points earned when an exposure of this action is accepted."""
    return 20.0 + 10.0 * stable_bucket("value:", action_id, 19)


def context_weight(action_id: str) -> float:
    """Synthetic context weight used in NBA arbitration."""
    return 0.8 + 0.1 * stable_bucket("context:", action_id, 5)


def business_lever(action_id: str) -> float:
    """Synthetic business lever used in NBA arbitration."""
    return 0.8 + 0.2 * stable_bucket("lever:", action_id, 3)


def ia_shares() -> dict[str, float]:
    """Share of subjects in each Impact Analyzer control group.

    Each active type targets one percent relative to the shared NBA group,
    so its share of all subjects is its odds over one plus the summed odds.
    ModelControl is split evenly into ``ModelControl_1`` and ``ModelControl_2``.

    Returns
    -------
    dict[str, float]
        Control group to share of subjects; shares sum to one.
    """
    odds = IA_TARGET_FRACTION / (1 - IA_TARGET_FRACTION)
    share = odds / (1 + len(IA_GROUPS) * odds)
    shares: dict[str, float] = {}
    for group in IA_GROUPS:
        if group == "ModelControl":
            shares["ModelControl_1"] = shares["ModelControl_2"] = share / 2
        else:
            shares[group] = share
    shares["NBA"] = 1 - sum(shares.values())
    return shares


def ia_group(seed: int, subject_id: str, period: str) -> str:
    """Assign a subject to an Impact Analyzer control group for one month."""
    draw = entity_seed(seed, "ih", "ia-group", subject_id, period) / 2**256
    cumulative = 0.0
    for group, share in ia_shares().items():
        cumulative += share
        if draw < cumulative:
            return group
    return "NBA"


@dataclass(frozen=True)
class _Candidate:
    action: Action
    membership: ActionChannel
    treatments: tuple[Treatment, ...]
    shares: tuple[float, ...]
    technique: str
    value: float
    context: float
    lever: float


@dataclass
class _Offer:
    candidate: _Candidate
    treatment: Treatment
    probability: float
    score: float
    random_score: float
    eligible: bool
    passes: bool


@cache
def _catalog(world: SampleWorld) -> dict[str, tuple[tuple[_Candidate, int, int], ...]]:
    treatments = {t.id: t for t in world.spec.treatments}
    techniques = {c.name: c.technique for c in world.spec.configurations}
    catalog: dict[str, list[tuple[_Candidate, int, int]]] = {c.name: [] for c in world.spec.channels}
    for action in world.spec.actions:
        start = world.window_day(action.start_day) if action.start_day > 0 else 0
        stop = world.days if _runs_to_end(action.stop_day) else world.window_day(action.stop_day or 0)
        for membership in action.channels:
            options = tuple(treatments[t] for t in membership.treatment_ids)
            candidate = _Candidate(
                action,
                membership,
                options,
                tuple(t.volume_share for t in options),
                techniques[membership.configuration],
                action_value(action.id),
                context_weight(action.id),
                business_lever(action.id),
            )
            catalog[membership.channel].append((candidate, start, stop))
    return {channel: tuple(entries) for channel, entries in catalog.items()}


def _active(world: SampleWorld, channel: str, day: int, memo: dict) -> tuple[list[_Candidate], list[float]]:
    key = (channel, day)
    if key not in memo:
        active = [c for c, start, stop in _catalog(world)[channel] if start <= day < stop]
        memo[key] = (active, [c.membership.volume_weight for c in active])
    return memo[key]


def _channel_shares(world: SampleWorld) -> dict[str, float]:
    volumes = {c.name: sum(d.volume for d in world.channel_days(c.name)) for c in world.spec.channels}
    others = sum(v for channel, v in volumes.items() if channel != "Call Center")
    shares = {channel: (1 - _CALL_CENTER_SHARE) * v / others for channel, v in volumes.items()}
    shares["Call Center"] = _CALL_CENTER_SHARE
    return shares


def _score_noise(auc: float, technique: str) -> float:
    noise = max(0.15, (85.0 - auc) / 25.0)
    return noise - 0.1 if technique == "GradientBoost" else noise


def _noisy(value: float, noise: float, rng: random.Random) -> float:
    return min(0.999, value * exp(noise * rng.gauss(0.0, 1.0) - noise * noise / 2))


def _rank(group: str, offers: Sequence[_Offer]) -> list[_Offer]:
    passing = [o for o in offers if o.passes]
    pool = [o for o in offers if o.eligible] if group == "EngagementPolicy" else passing
    pool = pool or [o for o in offers if o.eligible] or list(offers)

    def priority(o: _Offer) -> float:
        c = o.candidate
        if group == "ModelControl_1":
            return o.random_score
        if group in ("ModelControl_2", "PropensityPriority"):
            return o.score
        if group == "LeverPriority":
            return o.score * c.value * c.context
        return o.score * c.value * c.context * c.lever

    if group == "NBAPrioritization":
        return sorted(pool, key=lambda o: o.random_score, reverse=True)
    return sorted(pool, key=priority, reverse=True)


def outcome_labels(world: SampleWorld) -> dict[str, dict[str, list[str]]]:
    """Return the explicit per-``Channel/Direction`` outcome mapping of the sample.

    Outbound channels record ``Pending`` rather than ``Impression``, which the
    generic label resolver does not treat as an impression.
    """
    return {
        f"{c.name}/{c.direction}": {"Impressions": [IMPRESSION_OUTCOME[c.name]], "Accepts": [POSITIVE_OUTCOME[c.name]]}
        for c in world.spec.channels
    }


def interaction_history(world: SampleWorld, n: int) -> pl.LazyFrame:
    """Sample raw Interaction History rows in the ``pxStrategyResult`` export shape.

    Parameters
    ----------
    world : SampleWorld
        Shared latent state.
    n : int
        Number of distinct interactions.

    Returns
    -------
    pl.LazyFrame
        One row per outcome, sorted by interaction, outcome time and exposure.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    rng = entity_rng(world.seed, "ih", "interactions", str(n))
    channels = {c.name: c for c in world.spec.channels}
    shares = _channel_shares(world)
    names, weights = list(shares), list(shares.values())
    subjects = max(1, round(n * _SUBJECTS_PER_INTERACTION))
    start = datetime.combine(world.dates[0], time())

    plan = []
    for _ in range(n):
        channel = rng.choices(names, weights)[0]
        days = world.channel_days(channel)
        day = rng.choices(range(world.days), [d.volume for d in days])[0]
        second = rng.randrange(_FIRST_SECOND, _LAST_SECOND)
        subject = rng.randrange(subjects)
        plan.append((day, second, channel, subject))
    plan.sort()

    rows: dict[str, list] = {name: [] for name in _SCHEMA}
    memo: dict = {}
    groups: dict[tuple[int, str], str] = {}
    policies: dict[tuple[int, str], tuple[random.Random, dict[str, tuple[bool, bool]]]] = {}
    end = datetime.combine(world.dates[-1], time()) + timedelta(days=1)

    for index, (day, second, channel, subject) in enumerate(plan):
        state = world.channel_days(channel)[day]
        when = start + timedelta(days=day, seconds=second)
        period = f"{when:%Y-%m}"
        subject_id = f"CUST-{subject:07d}"
        interaction_id = f"{10**17 + index}"
        group = groups.get((subject, period))
        if group is None:
            group = groups[(subject, period)] = ia_group(world.seed, subject_id, period)
        key = (subject, period)
        if key not in policies:
            policies[key] = (entity_rng(world.seed, "ih", "policy", subject_id, period), {})
        policy_rng, passed = policies[key]

        active, volume_weights = _active(world, channel, day, memo)
        drawn = {c.action.id: c for c in rng.choices(active, volume_weights, k=_CANDIDATES)}
        mobile_challenger = channel == "Mobile" and day >= world.challenger_day
        technique = rng.choice(("NaiveBayes", "GradientBoost")) if mobile_challenger else None
        scale = state.ctr / (1 + state.lift) / world.baseline(channel)
        sigma = _AFFINITY_SIGMA[channel]
        offers = []
        for candidate in drawn.values():
            action_id = candidate.action.id
            if action_id not in passed:
                eligible = policy_rng.random() < POLICY_PASS["eligibility"]
                rest = policy_rng.random() < POLICY_PASS["applicability"]
                rest &= policy_rng.random() < POLICY_PASS["suitability"]
                passed[action_id] = (eligible, eligible and rest)
            eligible, passes = passed[action_id]
            treatment = rng.choices(candidate.treatments, candidate.shares)[0]
            affinity = exp(sigma * rng.gauss(0.0, 1.0) - sigma * sigma / 2)
            probability = min(0.95, treatment.baseline_probability * scale * affinity)
            noise = _score_noise(state.auc, technique or candidate.technique)
            offers.append(
                _Offer(
                    candidate,
                    treatment,
                    probability,
                    _noisy(probability, noise, rng),
                    rng.random(),
                    eligible,
                    passes,
                )
            )
        count = 1 + (rng.random() < _SECOND_OFFER.get(channel, 0.0))
        chosen = _rank(group, offers)[:count]
        experiment = "Conversion-Control" if rng.random() < _CONVERSION_CONTROL_SHARE else "Conversion-Test"
        mkt_type = "ModelControl" if group.startswith("ModelControl") else (None if group == "NBA" else group)
        model_control = {"ModelControl_1": "Control", "ModelControl_2": "Test"}.get(group, "NBA")

        for rank, offer in enumerate(chosen):
            candidate, treatment = offer.candidate, offer.treatment
            decision = {"ModelControl_1": offer.random_score, "ModelControl_2": offer.score}.get(group)
            open_probability = min(1.0, max(offer.probability, 0.08 + 4 * offer.probability))
            open_score = None
            if channel == "Email":
                open_score = _noisy(open_probability, _score_noise(state.auc, candidate.technique), rng)
            fixed = (
                interaction_id,
                f"{interaction_id}-{rank + 1}",
                subject_id,
                channel,
                channels[channel].direction,
                candidate.action.issue,
                candidate.action.group,
                candidate.action.name,
                treatment.name,
                technique or candidate.technique,
                experiment,
                f"NBAHealth_{group}",
                mkt_type,
                model_control,
            )
            scores = (
                offer.score,
                offer.score,
                offer.random_score if group == "ModelControl_1" else None,
                decision,
                open_score,
            )
            shown = when + timedelta(seconds=rank)
            events: list[tuple[datetime, str, float | None]] = [(shown, IMPRESSION_OUTCOME[channel], None)]
            positive_time = None
            if channel == "Email":
                if rng.random() < open_probability:
                    opened = shown + timedelta(minutes=rng.expovariate(1 / 360))
                    events.append((opened, "Opened", None))
                    if rng.random() < offer.probability / open_probability:
                        positive_time = opened + timedelta(minutes=rng.expovariate(1 / 20))
            elif rng.random() < offer.probability:
                delay = rng.expovariate(1 / 3) if channel == "Call Center" else rng.expovariate(1 / 0.7)
                positive_time = shown + timedelta(minutes=delay)
            if positive_time is not None:
                events.append((positive_time, POSITIVE_OUTCOME[channel], candidate.value))
                if rng.random() < _CONVERSION[experiment]:
                    converted = positive_time + timedelta(hours=rng.expovariate(1 / 30))
                    if converted < end:
                        events.append((converted, "Conversion", None))
            for event_time, outcome, value in events:
                for column, item in zip(_SCHEMA, (*fixed, event_time, outcome, *scores, value), strict=True):
                    rows[column].append(item)

    return pl.LazyFrame(rows, schema=_SCHEMA).sort("pxInteractionID", "pxOutcomeTime", "ExposureID", "pyOutcome")
