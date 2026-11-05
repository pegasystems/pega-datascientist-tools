"""Predictor binning view of the sample world.

Emits the latest ``PR_DATA_DM_ADMMART_PRED`` snapshot of every model in
:mod:`._adm`: one row per predictor bin for Naive Bayes models, including a
Classifier, and one row per predictor for Gradient Boost models, including
the context keys and each configuration's summary model. Bin counts are
integers that reconcile with the model totals, and every stored predictor
Performance is the AUC of its own bins.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from math import exp, log
from typing import TYPE_CHECKING

import polars as pl

from ._adm import AdmModel, adm_models, model_snapshots, summary_model_id
from ._random import entity_rng

if TYPE_CHECKING:
    from random import Random

    from ._spec import SampleSpec
    from ._world import SampleWorld
    from .schema import Distribution, Predictor

_MOBILE_SHARED = (
    "Param.ContentVariant",
    "Param.MessageLengthBand",
    "IH.Mobile.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
)
"""Predictors that rank on top of both Mobile configurations, for the NB/GB comparison."""
_NEVER_ACTIVE_SHARE = 0.08
_FAMILY_BOOST = {"ih": 1.5, "param": 1.3}
_OWN_CHANNEL_BOOST = 1.4
_RANK_NOISE = 0.35
_AUC_SLOPE = 0.105
_AUC_NOISE = 0.3
_AUC_RANGE = (0.503, 0.78)
_INACTIVE_SCALE = 0.08
_NUMERIC_BIN_COUNTS = (3, 4, 5, 6, 7)
_NUMERIC_BIN_WEIGHTS = (0.25, 0.35, 0.2, 0.12, 0.08)
_CUT_QUANTILES = tuple(q / 10 for q in range(1, 10))
_NB_CLASSIFIER_BINS = (12, 21)
_GB_CLASSIFIER_BINS = (15, 19)
_POSITIVES_PER_CLASSIFIER_BIN = 10
_CONTEXT_KEY_WEIGHT = {"Name": 0.9, "Treatment": 0.7, "Group": 0.35, "Issue": 0.25}

_SCHEMA = {
    "pyModelID": pl.Utf8,
    "pySnapshotTime": pl.Utf8,
    "pyPredictorName": pl.Utf8,
    "pyEntryType": pl.Utf8,
    "pyType": pl.Utf8,
    "pyBinIndex": pl.Int64,
    "pyTotalBins": pl.Int64,
    "pyBinType": pl.Utf8,
    "pyBinSymbol": pl.Utf8,
    "pyBinLowerBound": pl.Float64,
    "pyBinUpperBound": pl.Float64,
    "pyContents": pl.Utf8,
    "pyBinPositives": pl.Float64,
    "pyBinNegatives": pl.Float64,
    "pyPositives": pl.Float64,
    "pyNegatives": pl.Float64,
    "pyResponseCount": pl.Float64,
    "pyPerformance": pl.Float64,
    "pyGroupIndex": pl.Int64,
    "pyFeatureImportance": pl.Float64,
}


@dataclass(frozen=True)
class _Bin:
    kind: str
    symbol: str
    lower: float | None
    upper: float | None
    share: float
    z: float


@dataclass
class _Totals:
    snapshot: str
    positives: int
    negatives: int
    active: int
    performance: float


def _auc(positives: list[int], negatives: list[int]) -> float:
    """Return the AUC of binned counts, like :func:`cdh_utils.auc_from_bincounts`."""
    total_positives, total_negatives = sum(positives), sum(negatives)
    if total_positives == 0 or total_negatives == 0:
        return 0.5
    bins = sorted(
        ((p / (p + n), p, n) for p, n in zip(positives, negatives, strict=True) if p + n > 0),
        key=lambda b: -b[0],
    )
    area = true_positives = 0.0
    for _, p, n in bins:
        area += n * (2 * true_positives + p) / 2
        true_positives += p
    auc = area / (total_positives * total_negatives)
    return max(auc, 1 - auc)


def _expected_auc(shares: list[float], z: list[float], scale: float) -> float:
    """Approximate AUC when positives are proportional to ``share * exp(scale * z)``."""
    order = sorted(range(len(z)), key=lambda i: z[i])
    weights = [shares[i] * exp(scale * z[i]) for i in order]
    total_weight, total_share = sum(weights), sum(shares)
    if total_weight <= 0 or total_share <= 0:
        return 0.5
    area = below = 0.0
    for weight, i in zip(weights, order, strict=True):
        area += weight * (below + shares[i] / 2)
        below += shares[i]
    return area / (total_weight * total_share)


def _solve_scale(shares: list[float], z: list[float], target: float) -> float:
    low, high = 0.0, 12.0
    if _expected_auc(shares, z, high) <= target:
        return high
    for _ in range(30):
        middle = (low + high) / 2
        if _expected_auc(shares, z, middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _allocate(total: int, weights: list[float], caps: list[int] | None = None) -> list[int]:
    """Split an integer total proportionally to weights, respecting optional caps."""
    size = len(weights)
    limit = caps if caps is not None else [total] * size
    shares = [0.0] * size
    open_bins = [i for i in range(size) if limit[i] > 0 and weights[i] > 0]
    remaining = float(total)
    while open_bins and remaining > 1e-9:
        weight = sum(weights[i] for i in open_bins)
        overflow = [i for i in open_bins if shares[i] + remaining * weights[i] / weight > limit[i]]
        if not overflow:
            for i in open_bins:
                shares[i] += remaining * weights[i] / weight
            break
        for i in overflow:
            remaining -= limit[i] - shares[i]
            shares[i] = float(limit[i])
        open_bins = [i for i in open_bins if i not in overflow]
    counts = [min(limit[i], int(shares[i])) for i in range(size)]
    order = sorted(range(size), key=lambda i: (-(shares[i] - int(shares[i])), i))
    missing = total - sum(counts)
    for i in order + list(range(size)):
        if missing <= 0:
            break
        if counts[i] < limit[i]:
            counts[i] += 1
            missing -= 1
    return counts


def _log_odds(positives: list[int], negatives: list[int]) -> list[float]:
    total_positives, total_negatives, size = sum(positives), sum(negatives), len(positives)
    return [
        log(p + 1 / size) - log(total_positives + 1) - log(n + 1 / size) + log(total_negatives + 1)
        for p, n in zip(positives, negatives, strict=True)
    ]


def _importance(positives: list[int], negatives: list[int]) -> float:
    """Return Naive Bayes feature importance as the platform computes it."""
    responses = [p + n for p, n in zip(positives, negatives, strict=True)]
    total = sum(responses)
    if total == 0:
        return 0.0
    return sum(abs(o) * r for o, r in zip(_log_odds(positives, negatives), responses, strict=True)) / total


def _format(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _decimals(predictor: Predictor) -> int:
    edges = predictor.distribution.edges if predictor.distribution else ()
    integral = all(float(e).is_integer() for e in edges)
    return 0 if integral and edges and edges[-1] - edges[0] >= 20 else 2


def _distribution(predictor: Predictor) -> Distribution:
    if predictor.distribution is None:
        raise ValueError(f"Predictor {predictor.id!r} has no baseline distribution to bin")
    return predictor.distribution


def _cdf(predictor: Predictor, value: float) -> float:
    distribution = _distribution(predictor)
    edges, weights = distribution.edges, distribution.weights
    if value <= edges[0]:
        return 0.0
    if value >= edges[-1]:
        return 1.0
    i = bisect_right(edges, value) - 1
    return sum(weights[:i]) + weights[i] * (value - edges[i]) / (edges[i + 1] - edges[i])


@cache
def _cut_grid(predictor: Predictor) -> tuple[float, ...]:
    """Candidate interval boundaries shared by every model using a numeric predictor."""
    distribution = _distribution(predictor)
    edges, weights = distribution.edges, distribution.weights
    decimals = _decimals(predictor)
    cuts = []
    for quantile in _CUT_QUANTILES:
        cumulative = 0.0
        for i, weight in enumerate(weights):
            if cumulative + weight >= quantile:
                fraction = (quantile - cumulative) / weight if weight else 0.0
                cuts.append(round(edges[i] + fraction * (edges[i + 1] - edges[i]), decimals))
                break
            cumulative += weight
    return tuple(sorted({c for c in cuts if edges[0] < c < edges[-1]}))


def _symbols(predictor: Predictor) -> list[str]:
    return [str(v).lower() if isinstance(v, bool) else str(v) for v in _distribution(predictor).values]


def _missing_share(predictor: Predictor, rng: Random) -> float:
    return min(0.95, (predictor.missing_probability or 0.0) * exp(rng.gauss(0.0, 0.15)))


def _active_bins(predictor: Predictor, rng: Random) -> tuple[list[_Bin], str]:
    """Return the MISSING bin and equibehavior bins of an active predictor."""
    missing = _missing_share(predictor, rng)
    bins = [_Bin("MISSING", "MISSING", None, None, missing, rng.gauss(0.0, 0.4))]
    distribution = _distribution(predictor)
    if predictor.data_type == "numeric":
        edges = distribution.edges
        grid = _cut_grid(predictor)
        count = min(rng.choices(_NUMERIC_BIN_COUNTS, _NUMERIC_BIN_WEIGHTS)[0], len(grid) + 1)
        cuts = sorted(rng.sample(grid, count - 1))
        bounds = [edges[0], *cuts, edges[-1]]
        direction = rng.choice((-1.0, 1.0))
        middle = (count - 1) / 2 or 1.0
        for i, (lower, upper) in enumerate(pairwise(bounds)):
            if i == 0:
                symbol = f"<{_format(upper)}"
            elif i == count - 1:
                symbol = f">={_format(lower)}"
            else:
                symbol = f"[{_format(lower)}, {_format(upper)}>"
            share = (1 - missing) * (_cdf(predictor, upper) - _cdf(predictor, lower))
            z = direction * (i - (count - 1) / 2) / middle + rng.gauss(0.0, 0.15)
            bins.append(_Bin("EQUIBEHAVIOR", symbol, lower, upper, share, z))
        contents = f"Range : [{_format(edges[0])}, {_format(edges[-1])}]"
    else:
        symbols = _symbols(predictor)
        weights = dict(zip(symbols, distribution.weights, strict=True))
        effects = {s: rng.gauss(0.0, 1.0) for s in symbols}
        ordered = sorted(symbols, key=lambda s: effects[s])
        count = rng.randint(min(2, len(symbols)), len(symbols))
        groups = [ordered[i * len(ordered) // count : (i + 1) * len(ordered) // count] for i in range(count)]
        for group in groups:
            share = (1 - missing) * sum(weights[s] for s in group)
            z = sum(effects[s] for s in group) / len(group)
            bins.append(_Bin("EQUIBEHAVIOR", ", ".join(group), None, None, share, z))
        bins.append(_Bin("RESIDUAL", "Residual", None, None, 0.0, 0.0))
        contents = f"No. symbols = {len(symbols)}"
    return bins, contents


def _inactive_bins(predictor: Predictor, rng: Random) -> tuple[list[_Bin], str]:
    missing = _missing_share(predictor, rng)
    if predictor.data_type == "numeric":
        edges = _distribution(predictor).edges
        contents = f"Range : [{_format(edges[0])}, {_format(edges[-1])}]"
    else:
        contents = f"No. symbols = {len(_symbols(predictor))}"
    return [
        _Bin("MISSING", "MISSING", None, None, missing, rng.gauss(0.0, 1.0)),
        _Bin("EQUIBEHAVIOR", "NON-MISSING", None, None, 1 - missing, rng.gauss(0.0, 1.0)),
    ], contents


def _counts(bins: list[_Bin], positives: int, negatives: int, scale: float) -> tuple[list[int], list[int]]:
    responses = _allocate(positives + negatives, [b.share for b in bins])
    weights = [r * exp(scale * b.z) for r, b in zip(responses, bins, strict=True)]
    bin_positives = _allocate(positives, weights, responses)
    return bin_positives, [r - p for r, p in zip(responses, bin_positives, strict=True)]


@cache
def _group_indices(spec: SampleSpec) -> dict[str, int]:
    """Correlation group of each predictor; engagement windows group per base metric."""
    keys = {}
    for predictor in spec.predictors:
        key = predictor.correlation_family
        if key == "engagement_windows":
            key = re.sub(r"Last\d+Days?$", "", predictor.id)
        keys[predictor.id] = key
    index = {key: i for i, key in enumerate(sorted(set(keys.values())), start=1)}
    return {predictor_id: index[key] for predictor_id, key in keys.items()}


@cache
def _never_active(spec: SampleSpec, seed: int) -> dict[str, frozenset[str]]:
    """Predictors each configuration never activates, for the unused-predictor review."""
    never = {}
    predictors = {p.id: p for p in spec.predictors}
    for configuration in spec.configurations:
        rng = entity_rng(seed, "never_active", configuration.name)
        candidates = [
            p
            for p in configuration.predictor_ids
            if predictors[p].family not in _FAMILY_BOOST and p not in _MOBILE_SHARED
        ]
        size = round(len(configuration.predictor_ids) * _NEVER_ACTIVE_SHARE)
        never[configuration.name] = frozenset(rng.sample(sorted(candidates), min(size, len(candidates))))
    return never


def _strength(predictor: Predictor, channel: str) -> float:
    strength = (predictor.latent_effect_size or 0.0) * _FAMILY_BOOST.get(predictor.family, 1.0)
    if predictor.family == "ih" and f".{channel}." in predictor.id:
        strength *= _OWN_CHANNEL_BOOST
    return strength


def _ranking(world: SampleWorld, model: AdmModel, rng: Random) -> list[str]:
    configuration = next(c for c in world.spec.configurations if c.name == model.context.configuration)
    predictors = {p.id: p for p in world.spec.predictors}
    never = _never_active(world.spec, world.seed)[configuration.name]
    pinned = configuration.name.startswith("Mobile_")
    scores = {
        p: _strength(predictors[p], model.context.channel) * exp(rng.gauss(0.0, _RANK_NOISE))
        + (100.0 if pinned and p in _MOBILE_SHARED else 0.0)
        for p in configuration.predictor_ids
        if p not in never
    }
    return sorted(scores, key=lambda p: (-scores[p], p))


def _last_totals(world: SampleWorld) -> dict[str, _Totals]:
    last = (
        model_snapshots(world)
        .group_by("pyModelID")
        .agg(pl.all().sort_by("pySnapshotTime").last())
        .select("pyModelID", "pySnapshotTime", "pyPositives", "pyNegatives", "pyActivePredictors", "pyPerformance")
        .collect()
    )
    return {row[0]: _Totals(row[1], round(row[2]), round(row[3]), row[4], row[5] / 100) for row in last.iter_rows()}


class _Rows:
    def __init__(self) -> None:
        self.columns: dict[str, list] = {name: [] for name in _SCHEMA}

    def add(self, *values: object) -> None:
        for column, value in zip(_SCHEMA, values, strict=True):
            self.columns[column].append(value)

    def binned(
        self,
        model_id: str,
        totals: _Totals,
        name: str,
        entry_type: str,
        data_type: str,
        bins: list[_Bin],
        contents: str,
        counts: tuple[list[int], list[int]],
        group_index: int,
    ) -> float:
        positives, negatives = counts
        performance = _auc(positives, negatives)
        importance = 0.0 if entry_type == "Classifier" else _importance(positives, negatives)
        for index, (b, p, n) in enumerate(zip(bins, positives, negatives, strict=True), start=1):
            self.add(
                model_id,
                totals.snapshot,
                name,
                entry_type,
                data_type,
                index,
                len(bins),
                b.kind,
                b.symbol,
                b.lower,
                b.upper,
                contents,
                float(p),
                float(n),
                float(totals.positives),
                float(totals.negatives),
                float(totals.positives + totals.negatives),
                performance,
                group_index,
                importance,
            )
        return performance


def _classifier_bins(totals: _Totals, rng: Random, sizes: tuple[int, int]) -> tuple[list[_Bin], str]:
    """Score-distribution bins in the Naive Bayes log-odds score space."""
    center = (log(totals.positives + 1) - log(totals.negatives + 1)) / (1 + totals.active)
    half_range = 0.25 + 1.5 * totals.active**0.5 / (1 + totals.active)
    lowest, highest = round(center - 4 * half_range - 1, 2), round(center + 4 * half_range + 1, 2)
    contents = f"Range : [{_format(lowest)}, {_format(highest)}]"
    if totals.active == 0 or totals.positives == 0 or totals.negatives == 0:
        return [_Bin("EQUIBEHAVIOR", "NON-MISSING", lowest, highest, 1.0, 0.0)], contents
    count = min(rng.randint(*sizes), 1 + totals.positives // _POSITIVES_PER_CLASSIFIER_BIN)
    if count < 2:
        return [_Bin("EQUIBEHAVIOR", "NON-MISSING", lowest, highest, 1.0, 0.0)], contents
    cuts = [round(center - half_range + 2 * half_range * i / count, 2) for i in range(1, count)]
    bounds = [lowest, *cuts, highest]
    bins = []
    for i, (lower, upper) in enumerate(pairwise(bounds)):
        u = (i - (count - 1) / 2) / ((count - 1) / 2)
        if i == 0:
            symbol = f"<{_format(upper)}"
        elif i == count - 1:
            symbol = f">={_format(lower)}"
        else:
            symbol = f"[{_format(lower)}, {_format(upper)}>"
        bins.append(_Bin("EQUIBEHAVIOR", symbol, lower, upper, exp(-(u**2) / 0.405) + 0.02, u))
    return bins, contents


def _naive_bayes(world: SampleWorld, model: AdmModel, totals: _Totals, rows: _Rows) -> None:
    rng = entity_rng(world.seed, "binning", model.model_id)
    predictors = {p.id: p for p in world.spec.predictors}
    groups = _group_indices(world.spec)
    configuration = next(c for c in world.spec.configurations if c.name == model.context.configuration)
    active = set(_ranking(world, model, rng)[: totals.active])
    for name in configuration.predictor_ids:
        predictor = predictors[name]
        if name in active:
            bins, contents = _active_bins(predictor, rng)
            low, high = _AUC_RANGE
            target = 0.5 + _AUC_SLOPE * _strength(predictor, model.context.channel) * exp(rng.gauss(0.0, _AUC_NOISE))
            scale = _solve_scale([b.share for b in bins], [b.z for b in bins], min(high, max(low, target)))
        else:
            bins, contents = _inactive_bins(predictor, rng)
            scale = rng.uniform(0.0, _INACTIVE_SCALE)
        counts = _counts(bins, totals.positives, totals.negatives, scale)
        if name in active and predictor.data_type == "symbolic":
            order = [0, *sorted(range(1, len(bins) - 1), key=lambda i: _propensity(counts, i)), len(bins) - 1]
            bins = [bins[i] for i in order]
            counts = ([counts[0][i] for i in order], [counts[1][i] for i in order])
        entry_type = "Active" if name in active else "Inactive"
        rows.binned(model.model_id, totals, name, entry_type, predictor.data_type, bins, contents, counts, groups[name])
    _classifier(model.model_id, totals, rng, _NB_CLASSIFIER_BINS, rows)


def _propensity(counts: tuple[list[int], list[int]], i: int) -> float:
    positives, negatives = counts[0][i], counts[1][i]
    return positives / (positives + negatives) if positives + negatives else 0.0


def _classifier(model_id: str, totals: _Totals, rng: Random, sizes: tuple[int, int], rows: _Rows) -> None:
    bins, contents = _classifier_bins(totals, rng, sizes)
    target = min(0.97, max(0.5, totals.performance))
    scale = _solve_scale([b.share for b in bins], [b.z for b in bins], target) if len(bins) > 1 else 0.0
    counts = _counts(bins, totals.positives, totals.negatives, scale)
    rows.binned(model_id, totals, "Classifier", "Classifier", "numeric", bins, contents, counts, 0)


def _gradient_boost(
    world: SampleWorld,
    model: AdmModel,
    totals: _Totals,
    rows: _Rows,
    importances: dict[str, dict[str, float]],
) -> None:
    rng = entity_rng(world.seed, "binning", model.model_id)
    predictors = {p.id: p for p in world.spec.predictors}
    groups = _group_indices(world.spec)
    configuration = next(c for c in world.spec.configurations if c.name == model.context.configuration)
    ranking = _ranking(world, model, rng)
    active = ranking[: totals.active]
    raw = {p: _strength(predictors[p], model.context.channel) ** 2 * exp(rng.gauss(0.0, 0.5)) for p in active}
    total = sum(raw.values())
    importance = {p: 100 * v / total for p, v in raw.items()} if total else {}
    responses = totals.positives + totals.negatives
    accumulated = importances.setdefault(configuration.name, {})
    for name in configuration.predictor_ids:
        value = importance.get(name, 0.0)
        accumulated[name] = accumulated.get(name, 0.0) + value * responses
        entry_type = "Active" if name in importance else "Inactive"
        _single_row(rows, model.model_id, totals, name, entry_type, predictors[name].data_type, groups[name], value)
    for key in configuration.context_keys:
        _single_row(rows, model.model_id, totals, f"py{key}", "Inactive", "symbolic", 0, 0.0)
    _classifier(model.model_id, totals, rng, _GB_CLASSIFIER_BINS, rows)


def _single_row(
    rows: _Rows,
    model_id: str,
    totals: _Totals,
    name: str,
    entry_type: str,
    data_type: str,
    group_index: int,
    importance: float,
) -> None:
    rows.add(
        model_id,
        totals.snapshot,
        name,
        entry_type,
        data_type,
        1,
        1,
        "NONE",
        None,
        None,
        None,
        None,
        float(totals.positives),
        float(totals.negatives),
        float(totals.positives),
        float(totals.negatives),
        float(totals.positives + totals.negatives),
        0.0,
        group_index,
        importance,
    )


def _summaries(
    world: SampleWorld,
    models: tuple[AdmModel, ...],
    last: dict[str, _Totals],
    importances: dict[str, dict[str, float]],
    rows: _Rows,
) -> None:
    """Add the configuration-level importance of each AGB summary model."""
    predictors = {p.id: p for p in world.spec.predictors}
    groups = _group_indices(world.spec)
    for configuration in world.spec.configurations:
        summary_id = summary_model_id(configuration.name)
        if configuration.technique != "GradientBoost" or summary_id not in last:
            continue
        totals = last[summary_id]
        importance = dict(importances.get(configuration.name, {}))
        top = max(importance.values(), default=0.0)
        members = [m for m in models if m.context.configuration == configuration.name]
        values: dict[str, set[str | None]] = {
            "Issue": {m.issue for m in members},
            "Group": {m.group for m in members},
            "Name": {m.name for m in members},
            "Treatment": {m.treatment for m in members},
            "Channel": {m.context.channel for m in members},
            "Direction": {m.context.direction for m in members},
        }
        context = {
            key: top * _CONTEXT_KEY_WEIGHT.get(key, 0.0) if len(values[key]) > 1 else 0.0
            for key in configuration.context_keys
        }
        total = sum(importance.values()) + sum(context.values())
        scale = 100 / total if total else 0.0
        for name in configuration.predictor_ids:
            value = importance.get(name, 0.0) * scale
            entry_type = "Active" if value > 0 else "Inactive"
            _single_row(rows, summary_id, totals, name, entry_type, predictors[name].data_type, groups[name], value)
        for key, value in context.items():
            entry_type = "Active" if value > 0 else "Inactive"
            _single_row(rows, summary_id, totals, f"py{key}", entry_type, "symbolic", 0, value * scale)


def predictor_binning(world: SampleWorld) -> pl.LazyFrame:
    """Sample the latest ADM predictor binning snapshot in the ``PR_DATA_DM_ADMMART_PRED`` shape.

    Naive Bayes models get Active, Inactive and Classifier bins whose
    integer counts add up to the model's positives and negatives. Gradient
    Boost models get one row per predictor with ``FeatureImportance``, their
    context keys as inactive predictors, and a Classifier; each Gradient
    Boost configuration's summary model carries the global importance,
    including the context keys that vary across its actions.

    Parameters
    ----------
    world : SampleWorld
        Shared latent state.

    Returns
    -------
    pl.LazyFrame
        Raw predictor rows with ``py`` prefixed columns, ready for
        :class:`pdstools.ADMDatamart`.
    """
    models = adm_models(world.spec)
    last = _last_totals(world)
    rows = _Rows()
    importances: dict[str, dict[str, float]] = {}
    for model in models:
        totals = last.get(model.model_id)
        if totals is None:
            continue
        if model.technique == "NaiveBayes":
            _naive_bayes(world, model, totals, rows)
        else:
            _gradient_boost(world, model, totals, rows, importances)
    _summaries(world, models, last, importances, rows)

    by_predictor = ("pyModelID", "pyPredictorName")
    frame = pl.LazyFrame(rows.columns, schema=_SCHEMA)
    return frame.with_columns(
        pyBinResponseCount=pl.col("pyBinPositives") + pl.col("pyBinNegatives"),
        pyLift=(
            pl.col("pyBinPositives")
            * (pl.col("pyPositives") + pl.col("pyNegatives"))
            / ((pl.col("pyBinPositives") + pl.col("pyBinNegatives")) * pl.col("pyPositives"))
        ).fill_nan(None),
        pyZRatio=_z_ratio().over(by_predictor).fill_nan(None),
    )


def _z_ratio() -> pl.Expr:
    positive_fraction = pl.col("pyBinPositives") / pl.col("pyBinPositives").sum()
    negative_fraction = pl.col("pyBinNegatives") / pl.col("pyBinNegatives").sum()
    variance = (
        positive_fraction * (1 - positive_fraction) / pl.col("pyBinPositives").sum()
        + negative_fraction * (1 - negative_fraction) / pl.col("pyBinNegatives").sum()
    )
    return (positive_fraction - negative_fraction) / variance.sqrt()


__all__ = ["predictor_binning"]
