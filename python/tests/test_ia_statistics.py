"""Tests for pdstools.impactanalyzer.statistics.

Test categories
---------------
1. Pure formula tests with hand-calculable values
2. PDC validation — 16 experiment/channel pairs matching Pega's
   reported engagement lift
2b. Infinity validation — engagement lift, value lift, and CI against
    a real Pega Infinity Scenario Planner Actuals export
3. VBD validation — 18 experiment/channel pairs from a Scenario Planner
   Actuals export
4. Edge cases & robustness
"""

import pytest

from pdstools.impactanalyzer.statistics import (
    LiftResult,
    accept_rate,
    binomial_ci,
    calculate_engagement_lift,
    calculate_lift,
    calculate_value_lift,
    error_propagation,
    is_significant,
    lift_pl,
    required_sample_size,
)


# ============================================================================
# 1. Pure formula tests — hand-calculable known values
# ============================================================================


class TestAcceptRate:
    def test_basic(self):
        assert accept_rate(100, 1000) == 0.1

    def test_zero_impressions(self):
        assert accept_rate(50, 0) == 0.0

    def test_negative_impressions(self):
        assert accept_rate(50, -1) == 0.0

    def test_all_accepted(self):
        assert accept_rate(500, 500) == 1.0

    def test_none_accepted(self):
        assert accept_rate(0, 500) == 0.0


class TestBinomialCI:
    def test_known_value(self):
        # p=0.1, n=1000 -> CI = 1.96 * sqrt(0.1 * 0.9 / 1000) ≈ 0.018594
        ci = binomial_ci(100, 1000)
        assert abs(ci - 0.018594) < 0.0001

    def test_zero_impressions(self):
        assert binomial_ci(10, 0) == 0.0

    def test_zero_rate(self):
        assert binomial_ci(0, 1000) == 0.0

    def test_all_accepted(self):
        assert binomial_ci(1000, 1000) == 0.0

    def test_large_sample(self):
        ci = binomial_ci(5000, 100000)
        assert ci < 0.002


class TestCalculateLift:
    def test_basic(self):
        assert abs(calculate_lift(0.12, 0.10) - 0.2) < 1e-10

    def test_negative_lift(self):
        assert abs(calculate_lift(0.08, 0.10) - (-0.2)) < 1e-10

    def test_zero_control(self):
        assert calculate_lift(0.1, 0.0) == 0.0

    def test_equal(self):
        assert calculate_lift(0.1, 0.1) == 0.0


class TestLiftPl:
    """Test the Polars expression variant of lift."""

    def test_basic(self):
        import polars as pl

        df = pl.DataFrame({"test": [0.12], "control": [0.10]})
        result = df.select(lift=lift_pl("test", "control"))["lift"][0]
        assert abs(result - 0.2) < 1e-10

    def test_matches_scalar(self):
        import polars as pl

        df = pl.DataFrame({"t": [0.08, 0.15], "c": [0.10, 0.10]})
        results = df.select(lift=lift_pl("t", "c"))["lift"].to_list()
        for pl_val, (t, c) in zip(results, [(0.08, 0.10), (0.15, 0.10)]):
            assert abs(pl_val - calculate_lift(t, c)) < 1e-10


class TestErrorPropagation:
    def test_round_4(self):
        """Error propagation must round to 4 decimals to match Pega."""
        result = error_propagation(0.12, 0.10, 0.001, 0.002)
        assert result == round(result, 4)

    def test_zero_control(self):
        assert error_propagation(0.1, 0.0, 0.01, 0.02) == 0.0

    def test_known_value(self):
        # term1 = (0.001/0.10)^2 = 0.0001
        # term2 = (0.12*0.002/0.01)^2 = 0.024^2 = 0.000576
        # sqrt(0.0001 + 0.000576) = sqrt(0.000676) = 0.026
        result = error_propagation(0.12, 0.10, 0.001, 0.002)
        assert result == 0.026


class TestIsSignificant:
    def test_positive_significant(self):
        assert is_significant(0.20, 0.10) is True

    def test_not_significant(self):
        assert is_significant(0.05, 0.10) is False

    def test_negative_significant(self):
        assert is_significant(-0.20, 0.10) is True

    def test_exactly_at_boundary(self):
        assert is_significant(0.10, 0.10) is False


class TestCalculateEngagementLift:
    def test_returns_lift_result(self):
        result = calculate_engagement_lift(100, 1000, 80, 1000)
        assert isinstance(result, LiftResult)
        assert result.lift == calculate_lift(0.1, 0.08)

    def test_known_scenario(self):
        """Scenario: test 12 % accept rate, control 10 %, both n=10 000."""
        result = calculate_engagement_lift(1200, 10000, 1000, 10000)
        assert abs(result.lift - 0.2) < 0.001
        assert result.test_rate == 0.12
        assert result.control_rate == 0.10
        assert result.ci == round(result.ci, 4)


class TestCalculateValueLift:
    def test_basic(self):
        result = calculate_value_lift(2000.0, 1000, 1500.0, 1000)
        assert isinstance(result, LiftResult)
        expected_lift = (2.0 - 1.5) / 1.5
        assert abs(result.lift - expected_lift) < 1e-6

    def test_zero_impressions(self):
        result = calculate_value_lift(1000.0, 0, 500.0, 0)
        assert result.lift == 0.0

    def test_zero_test_impressions(self):
        # test has 0 impressions → vpi_test=0, control has data → lift = -1.0
        result = calculate_value_lift(1000.0, 0, 500.0, 1000)
        assert result.lift == -1.0


class TestRequiredSampleSize:
    def test_returns_positive(self):
        n = required_sample_size(0.05)
        assert n > 0

    def test_higher_baseline_needs_fewer(self):
        n_low = required_sample_size(0.01)
        n_high = required_sample_size(0.10)
        assert n_low > n_high

    def test_smaller_mde_needs_more(self):
        n_small = required_sample_size(0.05, mde=0.01)
        n_large = required_sample_size(0.05, mde=0.10)
        assert n_small > n_large


# ============================================================================
# 2. PDC validation — 16 experiments, matching Pega's reported numbers
# ============================================================================

PDC_GROUND_TRUTH = [
    ("NBAHealth_NBAPrioritization", "DirectMail", -0.003369949),
    ("NBAHealth_PropensityPriority", "DirectMail", -0.003931344),
    ("NBAHealth_LeverPriority", "DirectMail", 0.003564049),
    ("NBAHealth_ModelControl", "DirectMail", -0.014190719),
    ("NBAHealth_NBAPrioritization", "Email", 0.012584196),
    ("NBAHealth_PropensityPriority", "Email", 0.007881790),
    ("NBAHealth_LeverPriority", "Email", 0.007616910),
    ("NBAHealth_ModelControl", "Email", -0.009173467),
    ("NBAHealth_NBAPrioritization", "Push", 0.006642613),
    ("NBAHealth_PropensityPriority", "Push", 0.003514139),
    ("NBAHealth_LeverPriority", "Push", -0.012555372),
    ("NBAHealth_ModelControl", "Push", 0.044032414),
    ("NBAHealth_NBAPrioritization", "SMS", -0.006766899),
    ("NBAHealth_PropensityPriority", "SMS", 0.007751584),
    ("NBAHealth_LeverPriority", "SMS", 0.012333454),
    ("NBAHealth_ModelControl", "SMS", 0.018398743),
]

PDC_RAW = {
    ("NBAHealth_NBAPrioritization", "DirectMail"): (9985661, 996905, 90525, 9068),
    ("NBAHealth_PropensityPriority", "DirectMail"): (9985661, 996905, 104093, 10433),
    ("NBAHealth_LeverPriority", "DirectMail"): (9985661, 996905, 98292, 9778),
    ("NBAHealth_ModelControl", "DirectMail"): (56051, 5581, 41860, 4228),
    ("NBAHealth_NBAPrioritization", "Email"): (9985677, 998568, 90525, 8940),
    ("NBAHealth_PropensityPriority", "Email"): (9985677, 998568, 104094, 10328),
    ("NBAHealth_LeverPriority", "Email"): (9985677, 998568, 98293, 9755),
    ("NBAHealth_ModelControl", "Email"): (56055, 5570, 41860, 4198),
    ("NBAHealth_NBAPrioritization", "Push"): (9985651, 997925, 90525, 8987),
    ("NBAHealth_PropensityPriority", "Push"): (9985651, 997925, 104091, 10366),
    ("NBAHealth_LeverPriority", "Push"): (9985651, 997925, 98294, 9948),
    ("NBAHealth_ModelControl", "Push"): (56053, 5792, 41860, 4143),
    ("NBAHealth_NBAPrioritization", "SMS"): (9985457, 998417, 90525, 9113),
    ("NBAHealth_PropensityPriority", "SMS"): (9985457, 998417, 104094, 10328),
    ("NBAHealth_LeverPriority", "SMS"): (9985457, 998417, 98290, 9708),
    ("NBAHealth_ModelControl", "SMS"): (56052, 5601, 41857, 4107),
}


class TestPDCEngagementLiftValidation:
    """Validate engagement lift against real Pega PDC output."""

    @pytest.mark.parametrize(
        "experiment,channel,expected_lift",
        PDC_GROUND_TRUTH,
        ids=[f"{e}_{c}" for e, c, _ in PDC_GROUND_TRUTH],
    )
    def test_engagement_lift_matches_pdc(self, experiment, channel, expected_lift):
        impr_test, acc_test, impr_ctrl, acc_ctrl = PDC_RAW[(experiment, channel)]
        result = calculate_engagement_lift(acc_test, impr_test, acc_ctrl, impr_ctrl)
        assert abs(result.lift - expected_lift) < 1e-6, (
            f"{experiment}/{channel}: expected {expected_lift:.9f}, got {result.lift:.9f}"
        )


# ============================================================================
# 2b. Infinity validation — Scenario Planner Actuals export
# ============================================================================

# Raw counts from a Pega Infinity Impact Analyzer export
# (ConfidenceIntervalCalculation sheet, Grand Total rows)
INFINITY_RAW = {
    "NBA vs NBA without levers": (58, 5035, 1, 100, 4350.0, 75.0),
    "NBA vs Arbitrating with propensity only": (13, 1189, 2, 100, 975.0, 150.0),
    "NBA vs eligibility polices only": (4124, 41855, 81, 843, 309300.0, 6075.0),
}

# Pega-reported values from the Summary data sheet of the same export
INFINITY_EXPECTED = [
    ("NBA vs NBA without levers", 0.15193644, 0.1519364449, 2.2657),
    ("NBA vs Arbitrating with propensity only", -0.45332212, -0.4533221194, 0.8062),
    ("NBA vs eligibility polices only", 0.0254485277, 0.0254485282, 0.2144),
]


class TestInfinityExportValidation:
    """Validate against a real Pega Infinity Impact Analyzer export.

    Data extracted from a Scenario Planner Actuals export
    (``ImpactAnalyzerExport_7days``).  This covers engagement lift,
    value lift, and the error-propagated CI for each experiment that
    has non-zero control data.
    """

    @pytest.mark.parametrize(
        "experiment,expected_eng,expected_val,expected_ci",
        INFINITY_EXPECTED,
        ids=[e for e, *_ in INFINITY_EXPECTED],
    )
    def test_engagement_lift(self, experiment, expected_eng, expected_val, expected_ci):
        acc_t, impr_t, acc_c, impr_c, val_t, val_c = INFINITY_RAW[experiment]
        result = calculate_engagement_lift(acc_t, impr_t, acc_c, impr_c)
        assert abs(result.lift - expected_eng) < 1e-6, f"Engagement lift: expected {expected_eng}, got {result.lift}"

    @pytest.mark.parametrize(
        "experiment,expected_eng,expected_val,expected_ci",
        INFINITY_EXPECTED,
        ids=[e for e, *_ in INFINITY_EXPECTED],
    )
    def test_value_lift(self, experiment, expected_eng, expected_val, expected_ci):
        acc_t, impr_t, acc_c, impr_c, val_t, val_c = INFINITY_RAW[experiment]
        result = calculate_value_lift(val_t, impr_t, val_c, impr_c)
        assert abs(result.lift - expected_val) < 1e-6, f"Value lift: expected {expected_val}, got {result.lift}"

    @pytest.mark.parametrize(
        "experiment,expected_eng,expected_val,expected_ci",
        INFINITY_EXPECTED,
        ids=[e for e, *_ in INFINITY_EXPECTED],
    )
    def test_engagement_ci(self, experiment, expected_eng, expected_val, expected_ci):
        acc_t, impr_t, acc_c, impr_c, val_t, val_c = INFINITY_RAW[experiment]
        result = calculate_engagement_lift(acc_t, impr_t, acc_c, impr_c)
        assert result.ci == expected_ci, f"Engagement CI: expected {expected_ci}, got {result.ci}"

    def test_na_when_zero_control(self):
        """Experiments with zero control accepts should produce lift=0 (N/A)."""
        # NBA vs Random: 0 control accepts → lift undefined
        result = calculate_engagement_lift(59, 5162, 0, 100)
        assert result.lift == 0.0
        assert result.control_rate == 0.0


# ============================================================================
# 3. VBD validation
# ============================================================================

VBD_RAW = {
    ("NBA vs Random", "Email"): (19_339_719, 364_381, 161_634, 2_581),
    ("NBA vs Random", "Mobile"): (846_481, 42_448, 9_130, 124),
    ("NBA vs Random", "InboundApp"): (32_762_470, 304_474, 304_413, 1_913),
    ("NBA vs Random", "MobileApp"): (82_070_272, 1_998_201, 570_891, 9_853),
    ("NBA vs Random", "Web"): (1_736_769, 8_171, 15_788, 25),
    ("NBA vs Random", "OVERALL"): (148_145_231, 2_717_675, 1_063_732, 14_496),
    ("NBA vs Propensity Only", "Email"): (19_339_719, 364_381, 147_939, 2_537),
    ("NBA vs Propensity Only", "Mobile"): (846_481, 42_448, 6_579, 171),
    ("NBA vs Propensity Only", "InboundApp"): (32_762_470, 304_474, 252_792, 2_153),
    ("NBA vs Propensity Only", "MobileApp"): (82_070_272, 1_998_201, 490_711, 11_388),
    ("NBA vs Propensity Only", "Web"): (1_736_769, 8_171, 16_571, 41),
    ("NBA vs Propensity Only", "OVERALL"): (148_145_231, 2_717_675, 916_489, 16_290),
    ("Adaptive Models", "Email"): (74_800, 1_292, 203_108, 2_709),
    ("Adaptive Models", "Mobile"): (3_034, 70, 4_788, 82),
    ("Adaptive Models", "InboundApp"): (120_349, 1_034, 191_812, 749),
    ("Adaptive Models", "MobileApp"): (220_613, 5_388, 253_270, 3_682),
    ("Adaptive Models", "Web"): (8_172, 8, 7_852, 22),
    ("Adaptive Models", "OVERALL"): (427_943, 7_792, 662_569, 7_244),
}

VBD_COMPUTED = [
    ("NBA vs Random", "Email", +0.179914),
    ("NBA vs Random", "Mobile", +2.692233),
    ("NBA vs Random", "InboundApp", +0.478842),
    ("NBA vs Random", "MobileApp", +0.410711),
    ("NBA vs Random", "Web", +1.971120),
    ("NBA vs Random", "OVERALL", +0.346151),
    ("NBA vs Propensity Only", "Email", +0.098671),
    ("NBA vs Propensity Only", "Mobile", +0.929318),
    ("NBA vs Propensity Only", "InboundApp", +0.091171),
    ("NBA vs Propensity Only", "MobileApp", +0.049136),
    ("NBA vs Propensity Only", "Web", +0.901507),
    ("NBA vs Propensity Only", "OVERALL", +0.032086),
    ("Adaptive Models", "Email", +0.295027),
    ("Adaptive Models", "Mobile", +0.347171),
    ("Adaptive Models", "InboundApp", +1.200250),
    ("Adaptive Models", "MobileApp", +0.679950),
    ("Adaptive Models", "Web", -0.650603),
    ("Adaptive Models", "OVERALL", +0.665389),
]


class TestVBDValidation:
    """Validate engagement lift against a Scenario Planner Actuals VBD export."""

    @pytest.mark.parametrize(
        "experiment,channel,expected_lift",
        VBD_COMPUTED,
        ids=[f"VBD_{e}_{c}" for e, c, _ in VBD_COMPUTED],
    )
    def test_vbd_engagement_lift(self, experiment, channel, expected_lift):
        impr_test, acc_test, impr_ctrl, acc_ctrl = VBD_RAW[(experiment, channel)]
        result = calculate_engagement_lift(acc_test, impr_test, acc_ctrl, impr_ctrl)
        assert abs(result.lift - expected_lift) < 1e-5, (
            f"VBD {experiment}/{channel}: expected {expected_lift:+.6f}, got {result.lift:+.6f}"
        )


# ============================================================================
# 4. Edge cases & robustness
# ============================================================================


class TestEdgeCases:
    def test_zero_impressions_both(self):
        result = calculate_engagement_lift(0, 0, 0, 0)
        assert result.lift == 0.0
        assert result.ci == 0.0
        assert result.significant is False

    def test_zero_control_impressions(self):
        result = calculate_engagement_lift(100, 1000, 0, 0)
        assert result.lift == 0.0
        assert result.significant is False

    def test_single_impression(self):
        result = calculate_engagement_lift(1, 1, 0, 1)
        assert result.lift == 0.0

    def test_very_large_numbers(self):
        """Large samples should produce small CIs."""
        result = calculate_engagement_lift(1_000_000, 10_000_000, 20_000, 200_000)
        assert result.ci < 0.02
        assert result.lift == pytest.approx(0.0, abs=0.01)

    def test_tiny_control_group(self):
        """Tiny control = huge CI = not significant."""
        result = calculate_engagement_lift(100, 1000, 1, 10)
        assert result.ci > 0.5
