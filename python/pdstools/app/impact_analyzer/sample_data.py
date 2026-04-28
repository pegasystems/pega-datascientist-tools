"""Live data generator for the Impact Analyzer Streamlit app.

Generates synthetic PDC-like aggregate data that can be fed directly into
:class:`ImpactAnalyzer` without going through ``from_pdc`` parsing.  Each
call to :func:`generate_snapshot` produces one snapshot's worth of data
(one row per ControlGroup × Channel combination).

The generator profiles mirror the standalone transparency app's profiles
but produce aggregate rows instead of event-level data.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import polars as pl

# ── Experiment definitions ────────────────────────────────────────────────
# Must match ImpactAnalyzer.default_ia_experiments which expects these
# ControlGroup values.  Each experiment is (control_group, test_group).
EXPERIMENTS: list[tuple[str, str]] = [
    ("NBAPrioritization", "NBA"),  # NBA vs Random
    ("PropensityPriority", "NBA"),  # NBA vs Propensity Only
    ("LeverPriority", "NBA"),  # NBA vs No Levers
    ("EngagementPolicy", "NBA"),  # NBA vs Only Eligibility Rules
    ("ModelControl_1", "ModelControl_2"),  # Adaptive Models vs Random Propensity
]

# All unique ControlGroup values that appear in the data
ALL_CONTROL_GROUPS: list[str] = sorted({cg for pair in EXPERIMENTS for cg in pair})
# => EngagementPolicy, LeverPriority, ModelControl_1, ModelControl_2, NBA,
#    NBAPrioritization, PropensityPriority

DEFAULT_CHANNELS = ["Web", "Mobile", "Email"]

# ── Generator profiles ────────────────────────────────────────────────────
# Per control-group parameters:
#   (accept_rate, impressions_per_snapshot, avg_value_per_impression)

ProfileSpec = dict[str, tuple[float, int, float]]

GENERATOR_PROFILES: dict[str, dict[str, Any]] = {
    "Realistic NBA (asymmetric traffic)": {
        "description": ("Mirrors real Pega IA: ~98 % test traffic, low accept rates, NBA wins on most experiments."),
        "groups": {
            # Test groups (high traffic)
            "NBA": (0.0115, 5000, 0.85),
            "ModelControl_2": (0.0985, 3000, 0.78),
            # Control groups (low traffic)
            "NBAPrioritization": (0.0100, 100, 0.60),
            "PropensityPriority": (0.0200, 400, 0.70),
            "LeverPriority": (0.0100, 100, 0.55),
            "EngagementPolicy": (0.0960, 100, 0.72),
            "ModelControl_1": (0.0960, 100, 0.75),
        },
    },
    "Clear NBA winner (high lift)": {
        "description": "NBA clearly beats all baselines — significance reached quickly.",
        "groups": {
            "NBA": (0.12, 4000, 1.20),
            "ModelControl_2": (0.10, 2500, 1.00),
            "NBAPrioritization": (0.03, 400, 0.50),
            "PropensityPriority": (0.06, 400, 0.65),
            "LeverPriority": (0.05, 400, 0.55),
            "EngagementPolicy": (0.08, 400, 0.70),
            "ModelControl_1": (0.04, 400, 0.60),
        },
    },
    "Neck-and-neck (tiny lift)": {
        "description": "Very small differences — many snapshots needed for significance.",
        "groups": {
            "NBA": (0.051, 4000, 0.76),
            "ModelControl_2": (0.049, 2500, 0.73),
            "NBAPrioritization": (0.050, 400, 0.74),
            "PropensityPriority": (0.050, 400, 0.75),
            "LeverPriority": (0.050, 400, 0.75),
            "EngagementPolicy": (0.050, 400, 0.76),
            "ModelControl_1": (0.050, 400, 0.76),
        },
    },
    "NBA losing (negative lift)": {
        "description": "Control outperforms NBA — shows negative lift and red indicators.",
        "groups": {
            "NBA": (0.03, 4000, 0.50),
            "ModelControl_2": (0.02, 2500, 0.40),
            "NBAPrioritization": (0.05, 400, 0.80),
            "PropensityPriority": (0.06, 400, 0.85),
            "LeverPriority": (0.04, 400, 0.70),
            "EngagementPolicy": (0.05, 400, 0.90),
            "ModelControl_1": (0.05, 400, 0.90),
        },
    },
}

DEFAULT_PROFILE = "Realistic NBA (asymmetric traffic)"


def generate_snapshot(
    snapshot_time: datetime,
    profile_name: str = DEFAULT_PROFILE,
    channels: list[str] | None = None,
    noise: float = 0.10,
) -> pl.DataFrame:
    """Generate one snapshot of aggregate IA data.

    Parameters
    ----------
    snapshot_time : datetime
        Timestamp for the snapshot.
    profile_name : str
        Key into :data:`GENERATOR_PROFILES`.
    channels : list[str] | None
        Channel names. Defaults to ``["Web", "Mobile", "Email"]``.
    noise : float
        Relative noise applied to impressions/accept-rate (0 = deterministic).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns matching ``REQUIRED_IA_COLUMNS``:
        SnapshotTime, ControlGroup, Impressions, Accepts,
        ValuePerImpression, Channel.
    """
    profile = GENERATOR_PROFILES.get(profile_name, GENERATOR_PROFILES[DEFAULT_PROFILE])
    groups: ProfileSpec = profile["groups"]
    channels = channels or DEFAULT_CHANNELS

    rows: list[dict[str, Any]] = []
    for channel in channels:
        # Each channel gets a slightly different share of traffic
        channel_factor = 1.0 + random.uniform(-0.2, 0.2)
        for cg, (accept_rate, base_impressions, avg_vpi) in groups.items():
            impr = max(1, int(base_impressions * channel_factor * (1 + random.uniform(-noise, noise))))
            rate = max(0, accept_rate * (1 + random.uniform(-noise, noise)))
            accepts = sum(1 for _ in range(impr) if random.random() < rate)
            vpi = avg_vpi * (1 + random.uniform(-noise / 2, noise / 2)) if accepts > 0 else 0.0

            rows.append(
                {
                    "SnapshotTime": snapshot_time,
                    "ControlGroup": cg,
                    "Impressions": impr,
                    "Accepts": accepts,
                    "ValuePerImpression": round(vpi, 4),
                    "Channel": channel,
                }
            )

    return pl.DataFrame(rows).cast({"Impressions": pl.Float64, "Accepts": pl.Float64})


def generate_multi_snapshot(
    n_snapshots: int = 7,
    profile_name: str = DEFAULT_PROFILE,
    channels: list[str] | None = None,
    start_date: datetime | None = None,
    interval: timedelta | None = None,
    noise: float = 0.10,
) -> pl.DataFrame:
    """Generate multiple snapshots of aggregate IA data.

    Parameters
    ----------
    n_snapshots : int
        Number of daily snapshots to generate.
    profile_name : str
        Key into :data:`GENERATOR_PROFILES`.
    channels : list[str] | None
        Channel names.
    start_date : datetime | None
        Start date (defaults to *n_snapshots* days ago).
    interval : timedelta | None
        Time between snapshots (defaults to 1 day).
    noise : float
        Relative noise per snapshot.

    Returns
    -------
    pl.DataFrame
        Concatenated snapshots ready for ``ImpactAnalyzer(df.lazy())``.
    """
    interval = interval or timedelta(days=1)
    start = start_date or (datetime.now() - interval * n_snapshots)
    frames = [generate_snapshot(start + interval * i, profile_name, channels, noise) for i in range(n_snapshots)]
    return pl.concat(frames)
