from math import comb

import pytest
from pdstools.sample_data._world import OMNI, binomial, sample_world


def _quantile(n: int, p: float, target: float) -> int:
    cumulative = 0.0
    for k in range(n + 1):
        cumulative += comb(n, k) * p**k * (1 - p) ** (n - k)
        if cumulative >= target:
            return k
    return n


@pytest.mark.parametrize(
    ("n", "p", "z", "expected"),
    [
        (0, 0.5, 0.0, 0),
        (-3, 0.5, 0.0, 0),
        (10, 0.0, 3.0, 0),
        (10, 1.0, -3.0, 10),
        (10_000, 0.5, 0.0, 5000),
        (10_000, 0.5, 1.0, 5050),
    ],
)
def test_binomial_edges_and_normal_approximation(n, p, z, expected):
    assert binomial(n, p, z) == expected


@pytest.mark.parametrize("z", [-2.5, -1.0, -0.2, 0.0, 0.4, 1.3, 3.0])
def test_binomial_exact_inversion_matches_cdf(z):
    from statistics import NormalDist

    assert binomial(40, 0.1, z) == _quantile(40, 0.1, NormalDist().cdf(z))


def test_binomial_normal_approximation_is_clamped():
    assert binomial(1000, 0.5, 100.0) == 1000
    assert binomial(1000, 0.5, -100.0) == 0


def test_default_window_is_identity():
    world = sample_world(end="canonical")
    assert world.days == 182
    assert [world.window_day(d) for d in range(182)] == list(range(182))
    assert world.window_span(14) == 14
    assert world.challenger_day == 84


@pytest.mark.parametrize("days", [56, 60, 70])
def test_short_windows_preserve_weekdays(days):
    world = sample_world(days=days, end="canonical")
    canonical = sample_world(end="canonical")
    for day in (0, 6, 84, 120, 181):
        mapped = world.window_day(day)
        assert 0 <= mapped < days
        assert world.dates[mapped].weekday() == canonical.dates[day].weekday()
    assert world.window_span(0) == 0
    assert world.window_span(7) == 7


def test_channels_and_errors():
    world = sample_world(end="canonical")
    assert world.channels[-1] == OMNI
    assert set(world.channels) >= {"Web", "Mobile", "Email", "Call Center"}
    with pytest.raises(ValueError, match="Unknown channel"):
        world.channel_days("Fax")
    with pytest.raises(ValueError, match="scale"):
        sample_world(scale=0)
    with pytest.raises(ValueError, match="scale"):
        sample_world(scale=True)
    with pytest.raises(ValueError, match="at least 56"):
        sample_world(days=55)


def test_channel_days_cover_window_and_scale_volume():
    world = sample_world(end="canonical")
    double = sample_world(end="canonical", scale=2.0)
    for channel in world.channels:
        days = world.channel_days(channel)
        assert [d.day for d in days] == list(range(182))
        assert [d.date for d in days] == list(world.dates)
        assert all(50 <= d.auc <= 100 and d.ctr > 0 for d in days)
        assert double.channel_days(channel)[10].volume == pytest.approx(2 * days[10].volume)


def test_email_lift_is_negative_only_during_feed_dip():
    days = sample_world(end="canonical").channel_days("Email")
    negative = [d.day for d in days if d.lift < 0]
    assert negative
    dip = [d.day for d in days if d.ctr < days[0].ctr * 0.8 and not d.positive_only]
    assert set(negative) <= set(range(min(dip), max(dip) + 1))


def test_mobile_learns_and_challenger_helps():
    world = sample_world(end="canonical")
    days = world.channel_days("Mobile")
    assert days[0].ctr == pytest.approx(0.012)
    assert days[0].lift == pytest.approx(0.05)
    assert days[-1].ctr > 0.019 and days[-1].lift > 0.45 and days[-1].auc > 70


def test_positive_only_days_have_recovery_burst():
    for channel in sample_world(end="canonical").channels:
        days = sample_world(end="canonical").channel_days(channel)
        assert all(d.negative_burst == 1.0 for d in days if d.positive_only)
    bursts = [
        d for c in ("Web", "Email") for d in sample_world(end="canonical").channel_days(c) if d.negative_burst > 1
    ]
    assert bursts and all(d.negative_burst == 2.0 for d in bursts)


def test_seed_none_draws_fresh_seed():
    assert sample_world(seed=None).seed != sample_world(seed=None).seed
