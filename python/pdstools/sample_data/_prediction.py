"""Prediction Studio snapshot view of the shared sample world."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

import polars as pl

from ._random import entity_rng
from ._world import OMNI, SampleWorld, binomial

if TYPE_CHECKING:
    from datetime import date

PREDICTION_CLASS = "DATA-DECISION-REQUEST-CUSTOMER"
PREDICTIONS = {
    "Web": "PredictWebPropensity",
    "Mobile": "PredictMobilePropensity",
    "Email": "PredictOutboundEmailPropensity",
    "Call Center": "PredictOutboundCallCenterPropensity",
    OMNI: "PredictActionPropensity",
}
"""Standard prediction name per channel; names follow pdstools' channel mapping."""

MODEL_CONTROL_SHARE = 0.01
"""Configured target of the Impact Analyzer model-control experiment, split 50/50."""
NBA_RELATIVE_CTR = 0.97
"""NBA arm success rate relative to the model-driven Test arm."""
_NO_NBA = frozenset({"Call Center"})
_AUC_NOISE = 75.0
_MAX_AUC_NOISE = 6.0
_WEEKEND_AUC = -0.5
ARM_CORRELATION = 0.97
"""Share of the daily outcome shock common to all arms of one prediction.

Arms of one day see the same customers, content and context, so their
success rates move together. Coupling them keeps each count's binomial
spread while the small model-control arms still reveal the lift story.
"""

_SCHEMA = {
    "pySnapShotTime": pl.Utf8,
    "pyModelId": pl.Utf8,
    "pyModelType": pl.Utf8,
    "pySnapshotType": pl.Utf8,
    "pyDataUsage": pl.Utf8,
    "pyName": pl.Utf8,
    "pyValue": pl.Float64,
    "pyPositives": pl.Float64,
    "pyNegatives": pl.Float64,
    "pyCount": pl.Float64,
}


def _timestamp(when: date, hour: int) -> str:
    return f"{when:%Y%m%d}T{hour:02d}0000.000 GMT"


def prediction_snapshots(world: SampleWorld) -> pl.LazyFrame:
    """Sample daily prediction snapshots in the ``PR_DATA_DM_SNAPSHOTS`` shape.

    Every prediction emits, per day, one ``Daily`` AUC row at noon with
    full-population counts, plus ``Test`` and ``Control`` rows (and ``NBA``
    except for Call Center) at midnight, mirroring real exports.

    Parameters
    ----------
    world : SampleWorld
        Shared latent state.

    Returns
    -------
    pl.LazyFrame
        Raw prediction rows, ready for :class:`pdstools.Prediction`.
    """
    rows: dict[str, list] = {name: [] for name in _SCHEMA}

    def emit(when: date, model_id: str, usage: str | None, positives: int, negatives: int, auc=None) -> None:
        daily = usage is None
        values = (
            _timestamp(when, 12 if daily else 0),
            model_id,
            "PREDICTION",
            "Daily" if daily else None,
            usage,
            "auc" if daily else None,
            auc,
            float(positives),
            float(negatives),
            float(positives + negatives),
        )
        for name, value in zip(_SCHEMA, values, strict=True):
            rows[name].append(value)

    for channel in world.channels:
        name = PREDICTIONS[channel]
        model_id = f"{PREDICTION_CLASS}!{name.upper()}"
        rng = entity_rng(world.seed, "prediction", name)
        has_nba = channel not in _NO_NBA
        for state in world.channel_days(channel):
            total = max(0, round(rng.gauss(state.volume, 0.03 * state.volume)))
            test = binomial(total, MODEL_CONTROL_SHARE / 2, rng.gauss(0.0, 1.0))
            control = binomial(
                total - test, MODEL_CONTROL_SHARE / 2 / (1 - MODEL_CONTROL_SHARE / 2), rng.gauss(0.0, 1.0)
            )
            rest = total - test - control
            arms = {
                "Test": (test, state.ctr),
                "Control": (control, state.ctr / (1 + state.lift)),
                "NBA" if has_nba else "": (rest, state.ctr * (NBA_RELATIVE_CTR if has_nba else 1.0)),
            }
            shared = rng.gauss(0.0, 1.0)
            counts = {}
            for usage, (size, rate) in arms.items():
                z = ARM_CORRELATION * shared + sqrt(1 - ARM_CORRELATION**2) * rng.gauss(0.0, 1.0)
                positives = binomial(size, rate, z)
                negatives = 0 if state.positive_only else round((size - positives) * state.negative_burst)
                counts[usage] = (positives, negatives)
            positives = sum(p for p, _ in counts.values())
            negatives = sum(n for _, n in counts.values())
            noise = min(_MAX_AUC_NOISE, _AUC_NOISE / sqrt(positives + 1))
            auc = state.auc + rng.gauss(0, noise) + (_WEEKEND_AUC if state.date.weekday() >= 5 else 0)
            emit(state.date, model_id, None, positives, negatives, round(min(100.0, max(50.0, auc)), 4))
            for usage in ("Test", "Control", "NBA"):
                if usage in counts:
                    emit(state.date, model_id, usage, *counts[usage])

    return pl.LazyFrame(rows, schema=_SCHEMA)
