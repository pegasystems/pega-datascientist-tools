"""Shared latent ground truth from which every sample view is drawn.

The world holds expected, noise-free channel behaviour per day: volume,
model-driven success rate, model lift over random selection and model
quality. Views (predictions, ADM snapshots, interaction history) sample
observations from this state with their own independent random streams,
so they stay mutually consistent without depending on each other.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache
from math import exp, floor, sqrt
from statistics import NormalDist
from typing import TYPE_CHECKING

from ._calendar import DEFAULT_DAYS, EndDate, SampleCalendar, sample_calendar
from ._random import DEFAULT_SEED
from ._spec import DEFAULT_SPEC, SampleSpec

if TYPE_CHECKING:
    from .schema import Story

CANONICAL_WEEKS = DEFAULT_DAYS // 7
OMNI = "Omni"
"""Multi-channel pseudo channel served by ``OmniAdaptiveModel``."""

_WEEKEND_CTR = {"Web": 0.92, "Email": 0.92}
_TARGET_LIFT = {"Web": 0.60, "Email": 0.35, "Call Center": 0.20, OMNI: 0.45}
_STABLE_AUC = {"Web": 76.0, "Email": 68.0, "Call Center": 55.0, OMNI: 66.0}
_OMNI_DAILY_RESPONSES = 40_000
_OMNI_BASELINE = 0.025
_MOBILE_START_CTR = 0.012
_MOBILE_CHALLENGER = "Mobile_Click_Through_Rate_GB"
_NEGATIVE_BURST = 2.0


@dataclass(frozen=True)
class ChannelDay:
    """Expected, noise-free behaviour of one channel on one window day."""

    day: int
    """Zero-based index into the sample window."""
    date: date
    """Calendar date of this day."""
    volume: float
    """Expected number of responses."""
    ctr: float
    """Expected success rate when the models choose (prediction Test arm)."""
    lift: float
    """Model lift over random selection; Control rate is ``ctr / (1 + lift)``."""
    auc: float
    """Underlying model quality on Pega's 50-100 scale."""
    positive_only: bool
    """Whether negative feedback is delayed on this day."""
    negative_burst: float
    """Multiplier on negatives while delayed feedback catches up."""


def _round_half_up(value: float) -> int:
    return floor(value + 0.5)


@dataclass(frozen=True)
class SampleWorld:
    """Seeded latent state shared by all views of one synthetic world.

    Parameters
    ----------
    spec : SampleSpec
        Catalogs, channels and stories.
    seed : int
        Root seed for every view's random streams.
    calendar : SampleCalendar
        Exact window and its whole-week date shift.
    scale : float
        Multiplier on all response volumes.
    """

    spec: SampleSpec
    seed: int
    calendar: SampleCalendar
    scale: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)) or not self.scale > 0:
            raise ValueError("scale must be a positive number.")

    @property
    def days(self) -> int:
        """Number of days in the window."""
        return self.calendar.days

    @property
    def dates(self) -> tuple[date, ...]:
        """Window dates, oldest first."""
        start = self.calendar.start
        return tuple(start + timedelta(days=i) for i in range(self.days))

    def window_day(self, canonical_day: int) -> int:
        """Map a canonical 26-week day to this window, preserving the weekday.

        Whole weeks are rescaled to the window's full weeks; a partial first
        week shifts everything by its length. The default window is the identity.
        """
        full_weeks, lead = divmod(self.days, 7)
        week = _round_half_up((canonical_day // 7) * full_weeks / CANONICAL_WEEKS)
        return lead + 7 * min(week, full_weeks - 1) + canonical_day % 7

    def window_span(self, canonical_days: int) -> int:
        """Rescale a duration of whole canonical weeks, keeping at least one week."""
        if canonical_days <= 0:
            return 0
        weeks = max(1, _round_half_up((canonical_days / 7) * (self.days // 7) / CANONICAL_WEEKS))
        return 7 * weeks

    def _story_phase(self, story: Story, day: int) -> tuple[str, float]:
        start = self.window_day(story.start_day)
        duration = self.window_span(story.duration_days)
        recovery = self.window_span(story.recovery_days)
        if start <= day < start + duration:
            return "event", (day - start) / duration
        if start + duration <= day < start + duration + recovery:
            return "recovery", (day - start - duration) / recovery
        return "none", 0.0

    @property
    def challenger_day(self) -> int:
        """Window day on which the Mobile AGB challenger starts."""
        configuration = next(c for c in self.spec.configurations if c.name == _MOBILE_CHALLENGER)
        return self.window_day(configuration.start_day)

    @property
    def channels(self) -> tuple[str, ...]:
        """Channel names including the multi-channel Omni stream."""
        return (*(c.name for c in self.spec.channels), OMNI)

    def channel_days(self, channel: str) -> tuple[ChannelDay, ...]:
        """Return the expected behaviour of one channel for every window day."""
        return _channel_days(self, channel)

    def feedback(self, channel: str, technique: str) -> tuple[tuple[bool, float], ...]:
        """Return per-day delayed-feedback state of one channel and model technique.

        Positive-only stories hit every model of the affected channel and
        technique, including multi-channel models, so the weekly success rate
        of that facet is exactly one.

        Returns
        -------
        tuple[tuple[bool, float], ...]
            Per window day: whether negatives are withheld, and the multiplier
            on negatives while they catch up.
        """
        return _feedback(self, channel, technique)

    def baseline(self, channel: str) -> float:
        """Mature weekday success rate of a channel, including the Omni stream."""
        return self._daily_volume(channel)[1]

    def _daily_volume(self, channel: str) -> tuple[float, float]:
        if channel == OMNI:
            return _OMNI_DAILY_RESPONSES * self.scale, _OMNI_BASELINE
        spec_channel = next((c for c in self.spec.channels if c.name == channel), None)
        if spec_channel is None:
            raise ValueError(f"Unknown channel {channel!r}; expected one of {self.channels}.")
        if spec_channel.daily_responses is not None:
            daily = float(spec_channel.daily_responses)
        else:
            daily = (spec_channel.window_responses or 0) / DEFAULT_DAYS
        return daily * self.scale, spec_channel.baseline_probability

    def _mobile(self, day: int, baseline: float) -> tuple[float, float, float]:
        challenger = self.challenger_day
        nb_learning = 1 - exp(-day / 9)
        nb_lift = 0.05 + 0.37 * (1 - exp(-day / 28))
        nb_auc = 50 + 20 * nb_learning
        ctr = _MOBILE_START_CTR + (baseline - _MOBILE_START_CTR) * (1 - exp(-day / 14))
        if day < challenger:
            return ctr, nb_lift, nb_auc
        age = day - challenger
        agb_learning = 1 - exp(-age / 14)
        agb_lift = 0.05 + 0.53 * (1 - exp(-age / 21))
        agb_auc = 50 + 23 * agb_learning
        return ctr, (nb_lift + agb_lift) / 2, (nb_auc + agb_auc) / 2


@cache
def _channel_days(world: SampleWorld, channel: str) -> tuple[ChannelDay, ...]:
    volume, baseline = world._daily_volume(channel)
    stories = [s for s in world.spec.stories if s.channel == channel]
    result = []
    for day, when in enumerate(world.dates):
        weekend = when.weekday() >= 5
        if channel == "Mobile":
            ctr, lift, auc = world._mobile(day, baseline)
        else:
            ctr, lift, auc = baseline, _TARGET_LIFT[channel], _STABLE_AUC[channel]
        if weekend:
            ctr *= _WEEKEND_CTR.get(channel, 1.0)
        positive_only, burst = False, 1.0
        for story in stories:
            phase, progress = world._story_phase(story, day)
            if story.positive_only:
                positive_only |= phase == "event"
                burst = _NEGATIVE_BURST if phase == "recovery" else burst
                continue
            if phase == "event":
                ctr *= story.ctr_multiplier
                lift = -0.10 if 0.4 <= progress < 0.6 else 0.05
                auc -= 4
            elif phase == "recovery":
                ctr *= story.ctr_multiplier + (1 - story.ctr_multiplier) * progress
                lift = 0.05 + (lift - 0.05) * progress
                auc -= 4 * (1 - progress)
        result.append(ChannelDay(day, when, volume, ctr, lift, auc, positive_only, burst))
    return tuple(result)


@cache
def _feedback(world: SampleWorld, channel: str, technique: str) -> tuple[tuple[bool, float], ...]:
    stories = [s for s in world.spec.stories if s.positive_only and s.channel == channel and s.technique == technique]
    result = []
    for day in range(world.days):
        phases = [world._story_phase(story, day)[0] for story in stories]
        result.append(("event" in phases, _NEGATIVE_BURST if "recovery" in phases else 1.0))
    return tuple(result)


def binomial(n: int, p: float, z: float) -> int:
    """Return the binomial count at the quantile of a standard normal draw.

    Taking the draw as an argument lets views couple related counts, such
    as the Test and Control arms of one day, by sharing part of ``z``. Uses
    exact CDF inversion for small expected counts and a rounded normal
    approximation once ``n * p * (1 - p)`` exceeds 25. Standard library only.

    Parameters
    ----------
    n : int
        Number of trials.
    p : float
        Success probability.
    z : float
        Standard normal draw selecting the quantile.

    Returns
    -------
    int
        Count between 0 and ``n``.
    """
    if n <= 0 or p <= 0:
        return 0
    if p >= 1:
        return n
    variance = n * p * (1 - p)
    if variance > 25:
        return min(n, max(0, round(n * p + z * sqrt(variance))))
    target = NormalDist().cdf(z)
    odds = p / (1 - p)
    mass = (1 - p) ** n
    cumulative, count = mass, 0
    while cumulative < target and count < n:
        mass *= (n - count) / (count + 1) * odds
        count += 1
        cumulative += mass
    return count


def sample_world(
    *,
    seed: int | None = DEFAULT_SEED,
    days: int = DEFAULT_DAYS,
    end: EndDate = None,
    scale: float = 1.0,
    spec: SampleSpec = DEFAULT_SPEC,
) -> SampleWorld:
    """Resolve arguments shared by all sample views into one world.

    Parameters
    ----------
    seed : int or None, default 42
        Root seed; ``None`` draws a fresh random seed.
    days : int, default 182
        Window length, at least 56 days.
    end : date, "canonical" or None, default None
        Window end, resolved to a Sunday (see :func:`sample_calendar`).
    scale : float, default 1.0
        Multiplier on response volumes.
    spec : SampleSpec, default DEFAULT_SPEC
        Catalog specification.

    Returns
    -------
    SampleWorld
        Immutable world; its derived channel state is cached per process.
    """
    if seed is None:
        seed = random.SystemRandom().randrange(2**63)
    return SampleWorld(spec, seed, sample_calendar(days, end), scale)
