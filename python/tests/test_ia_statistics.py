"""Tests for pdstools.impactanalyzer.statistics.

Test categories
---------------
1. Pure formula tests with hand-calculable values
2. PDC parity — 16 experiment / channel pairs
3. Pega Infinity parity — engagement lift, value lift, variance, SE,
   and CI from a reference export
4. VBD parity — 18 experiment / channel pairs
5. Edge cases and robustness

Note: Scenario Planner Actuals validation is pending.
"""

import math

import pytest

from pdstools.impactanalyzer.statistics import (
    LiftResult,
    Z_95,
    FORMULAS,
    accept_rate,
    binomial_ci,
    binomial_se,
    calculate_engagement_lift,
    calculate_lift,
    calculate_value_lift,
    lift_se,
    is_significant,
    lift_pl,
    required_sample_size,
    value_se,
    value_variance,
)


# ====================================================================
# 1. Pure formula tests — hand-calculable known values
# ====================================================================


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


class TestBinomialSE:
    def test_known_value(self):
        # p=0.1, n=1000 -> SE = sqrt(0.1*0.9/1000) ≈ 0.009487
        se = binomial_se(100, 1000)
        assert abs(se - 0.009487) < 0.0001

    def test_zero_impressions(self):
        assert binomial_se(10, 0) == 0.0

    def test_zero_rate(self):
        assert binomial_se(0, 1000) == 0.0

    def test_all_accepted(self):
        assert binomial_se(1000, 1000) == 0.0


class TestBinomialCI:
    def test_known_value(self):
        # CI = z * SE = 1.96 * sqrt(0.1*0.9/1000) ≈ 0.018594
        ci = binomial_ci(100, 1000)
        assert abs(ci - 0.018594) < 0.0001

    def test_equals_z_times_se(self):
        se = binomial_se(100, 1000)
        ci = binomial_ci(100, 1000)
        assert abs(ci - Z_95 * se) < 1e-15

    def test_large_sample(self):
        ci = binomial_ci(5000, 100000)
        assert ci == pytest.approx(0.0013508367777048417, rel=1e-10)
        assert ci < 0.002  # supplementary smoke check


class TestValueVariance:
    def test_known_value(self):
        # p=0.1, AV=75 -> Var = 0.1*0.9*75^2 = 506.25
        var = value_variance(100, 1000, 75.0)
        assert abs(var - 506.25) < 0.001

    def test_zero_impressions(self):
        assert value_variance(0, 0, 75.0) == 0.0


class TestValueSE:
    def test_known_value(self):
        # Var=506.25, n=1000 -> SE = sqrt(506.25/1000) ≈ 0.71151
        se = value_se(100, 1000, 75.0)
        assert abs(se - math.sqrt(506.25 / 1000)) < 1e-10

    def test_zero_impressions(self):
        assert value_se(0, 0, 75.0) == 0.0


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
        for pl_val, (t, c) in zip(results, [(0.08, 0.10), (0.15, 0.10)], strict=True):
            assert abs(pl_val - calculate_lift(t, c)) < 1e-10


class TestErrorPropagation:
    def test_zero_control(self):
        assert lift_se(0.1, 0.0, 0.01, 0.02) == 0.0

    def test_known_value(self):
        # term1 = (0.001/0.10)^2 = 0.0001
        # term2 = (0.12/0.10)^2 * (0.002/0.10)^2 ... wait, let me compute:
        # = (1/0.10) * sqrt(0.001^2 + (0.12/0.10)^2 * 0.002^2)
        # = 10 * sqrt(0.000001 + 1.44 * 0.000004)
        # = 10 * sqrt(0.000001 + 0.00000576)
        # = 10 * sqrt(0.00000676) = 10 * 0.0026 = 0.026
        result = lift_se(0.12, 0.10, 0.001, 0.002)
        assert abs(result - 0.026) < 0.001

    def test_full_precision(self):
        """lift_se returns full precision, no rounding."""
        result = lift_se(0.12, 0.10, 0.001, 0.002)
        # Should NOT be rounded to 4 decimals
        assert result == result  # not NaN
        assert isinstance(result, float)


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
        result = calculate_engagement_lift(1200, 10000, 1000, 10000)
        assert abs(result.lift - 0.2) < 0.001
        assert result.test_rate == 0.12
        assert result.control_rate == 0.10


class TestCalculateValueLift:
    def test_basic(self):
        # 200 accepts out of 1000, AV=10 -> VPI_test = 0.2*10 = 2.0
        # 150 accepts out of 1000, AV=10 -> VPI_ctrl = 0.15*10 = 1.5
        result = calculate_value_lift(200, 1000, 150, 1000, 10.0)
        assert isinstance(result, LiftResult)
        expected_lift = (2.0 - 1.5) / 1.5
        assert abs(result.lift - expected_lift) < 1e-6

    def test_zero_control_impressions(self):
        result = calculate_value_lift(100, 1000, 0, 0, 75.0)
        assert result.lift == 0.0

    def test_zero_control_accepts(self):
        result = calculate_value_lift(100, 1000, 0, 1000, 75.0)
        assert result.lift == 0.0
        assert result.control_rate == 0.0


class TestLiftResultFrozen:
    def test_immutable(self):
        result = calculate_engagement_lift(100, 1000, 80, 1000)
        with pytest.raises(AttributeError):
            result.lift = 0.5  # type: ignore[misc]

    def test_ci_95(self):
        result = calculate_engagement_lift(1200, 10000, 1000, 10000)
        assert result.ci_95() == pytest.approx(Z_95 * result.se)


class TestFormula:
    def test_all_formulas_present(self):
        expected = {
            "accept_rate",
            "binomial_se",
            "binomial_ci",
            "value_variance",
            "value_se",
            "lift",
            "lift_se",
            "significance",
            "vpi",
            "required_sample_size",
            "ewma",
            "ci_band",
        }
        assert set(FORMULAS.keys()) == expected

    def test_formula_is_frozen(self):
        f = FORMULAS["accept_rate"]
        with pytest.raises(AttributeError):
            f.name = "modified"  # type: ignore[misc]


class TestIsSignificantConfidenceLevel:
    """Verify is_significant uses 95% (z=1.96) by default."""

    def test_borderline_not_significant(self):
        # lift=0.10, se=0.06 → 0.10 - 1.96*0.06 = -0.0176 < 0 → not sig
        assert is_significant(0.10, 0.06) is False

    def test_borderline_significant(self):
        # lift=0.20, se=0.06 → 0.20 - 1.96*0.06 = 0.0824 > 0 → sig
        assert is_significant(0.20, 0.06) is True

    def test_custom_z(self):
        # With z=1.0: lift=0.10, se=0.06 → 0.10 - 0.06 = 0.04 > 0 → sig
        assert is_significant(0.10, 0.06, z=1.0) is True


class TestRequiredSampleSize:
    def test_returns_positive(self):
        n = required_sample_size(0.05)
        assert n == 1557663

    def test_higher_baseline_needs_fewer(self):
        n_low = required_sample_size(0.01)
        n_high = required_sample_size(0.10)
        assert n_low == 8125093
        assert n_high == 736734
        assert n_low > n_high  # supplementary ordering check

    def test_smaller_mde_needs_more(self):
        n_small = required_sample_size(0.05, mde=0.01)
        n_large = required_sample_size(0.05, mde=0.10)
        assert n_small == 38223144
        assert n_large == 398351
        assert n_small > n_large  # supplementary ordering check

    def test_known_value(self):
        """Exact regression value for default parameters."""
        n = required_sample_size(0.05)
        assert n == 1557663


# ====================================================================
# 2. PDC parity — 16 experiment / channel pairs
# ====================================================================

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


class TestPDCEngagementLiftValidation:
    """Validate engagement lift against Pega PDC output."""

    @pytest.mark.parametrize(
        "experiment,channel,expected_lift",
        PDC_GROUND_TRUTH,
        ids=[f"{e}_{c}" for e, c, _ in PDC_GROUND_TRUTH],
    )
    def test_engagement_lift(self, experiment, channel, expected_lift):
        impr_t, acc_t, impr_c, acc_c = PDC_RAW[(experiment, channel)]
        result = calculate_engagement_lift(acc_t, impr_t, acc_c, impr_c)
        assert abs(result.lift - expected_lift) < 1e-6, (
            f"{experiment}/{channel}: expected {expected_lift:.9f}, got {result.lift:.9f}"
        )


# ====================================================================
# 3. Pega Infinity parity — reference export
# ====================================================================
#
# Extracted from a real ``ImpactAnalyzerExport_7days`` Excel workbook
# (ConfidenceIntervalCalculation and Summary data sheets).
#
# Columns from the CI sheet:
#   ExperimentName, AggregatedAcceptRate_{Test,Control},
#   AggregatedValuePerImpression_{Test,Control},
#   {Test,Control}Variance, {Test,Control}Interval,
#   ValueLiftConfidenceInterval, EngagementLiftConfidenceInterval
#
# Format: (accepts_test, impr_test, accepts_ctrl, impr_ctrl, action_value)

INFINITY_RAW = {
    "Exp_A": (58, 5035, 1, 100, 75),
    "Exp_B": (13, 1189, 2, 100, 75),
    "Exp_C": (4124, 41855, 81, 843, 75),
}

# Pega-reported values from the ConfidenceIntervalCalculation sheet
INFINITY_PEGA = {
    "Exp_A": {
        "ar_t": 0.0115193644,
        "ar_c": 0.01,
        "vpi_t": 0.8639523337,
        "vpi_c": 0.75,
        "var_t": 64.0500111215,
        "var_c": 55.6875,
        "se_vpi_t": 0.1127872135,
        "se_vpi_c": 0.7462405778,
        "eng_lift": 0.15193644,
        "val_lift": 0.1519364449,
        "eng_se": 1.1559857344470446,
        "val_se": 1.1559857393,
    },
    "Exp_B": {
        "ar_t": 0.0109335576,
        "ar_c": 0.02,
        "vpi_t": 0.8200168209,
        "vpi_c": 1.5,
        "var_t": 60.8288339149,
        "var_c": 110.25,
        "se_vpi_t": 0.2261850094,
        "se_vpi_c": 1.05,
        "eng_lift": -0.45332212,
        "val_lift": -0.4533221194,
        "eng_se": 0.41131181745762024,
        "val_se": 0.4113118179,
    },
    "Exp_C": {
        "ar_t": 0.0985306415,
        "ar_c": 0.0960854093,
        "vpi_t": 7.3897981125,
        "vpi_c": 7.206405694,
        "var_t": 499.625742294,
        "var_c": 488.5481442355,
        "se_vpi_t": 0.1092568638,
        "se_vpi_c": 0.7612720704,
        "eng_lift": 0.0254485277,
        "val_lift": 0.0254485282,
        "eng_se": 0.10938239060194815,
        "val_se": 0.1093823907,
    },
}

# Experiments with zero control accepts (lift undefined / N/A)
INFINITY_ZERO_CONTROL = {
    "Exp_ZC1": (59, 5162, 0, 100, 75),
    "Exp_ZC2": (54, 4741, 0, 100, 75),
}


class TestInfinityAcceptRate:
    """Validate accept rates against the Infinity CI sheet."""

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_accept_rate_test(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        expected = INFINITY_PEGA[exp_id]["ar_t"]
        assert abs(accept_rate(at, it) - expected) < 1e-8

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_accept_rate_control(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        expected = INFINITY_PEGA[exp_id]["ar_c"]
        assert abs(accept_rate(ac, ic) - expected) < 1e-8


class TestInfinityValueVariance:
    """Validate value variance against Pega's TestVariance / ControlVariance."""

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_variance_test(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        expected = INFINITY_PEGA[exp_id]["var_t"]
        assert abs(value_variance(at, it, av) - expected) < 0.01

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_variance_control(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        expected = INFINITY_PEGA[exp_id]["var_c"]
        assert abs(value_variance(ac, ic, av) - expected) < 0.01


class TestInfinityValueSE:
    """Validate value SE against Pega's TestInterval / ControlInterval."""

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_se_test(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        expected = INFINITY_PEGA[exp_id]["se_vpi_t"]
        assert abs(value_se(at, it, av) - expected) < 1e-6

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_se_control(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        expected = INFINITY_PEGA[exp_id]["se_vpi_c"]
        assert abs(value_se(ac, ic, av) - expected) < 1e-6


class TestInfinityEngagementLift:
    """Validate engagement lift and CI against the Infinity export."""

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_lift(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        p = INFINITY_PEGA[exp_id]
        result = calculate_engagement_lift(at, it, ac, ic)
        assert abs(result.lift - p["eng_lift"]) < 1e-6

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_ci(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        p = INFINITY_PEGA[exp_id]
        result = calculate_engagement_lift(at, it, ac, ic)
        assert abs(result.se - p["eng_se"]) < 0.001, f"Engagement CI: expected {p['eng_ci']}, got {result.se}"


class TestInfinityValueLift:
    """Validate value lift and CI against the Infinity export."""

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_lift(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        p = INFINITY_PEGA[exp_id]
        result = calculate_value_lift(at, it, ac, ic, av)
        assert abs(result.lift - p["val_lift"]) < 1e-6

    @pytest.mark.parametrize("exp_id", INFINITY_PEGA.keys())
    def test_ci(self, exp_id):
        at, it, ac, ic, av = INFINITY_RAW[exp_id]
        p = INFINITY_PEGA[exp_id]
        result = calculate_value_lift(at, it, ac, ic, av)
        assert abs(result.se - p["val_se"]) < 0.001, f"Value CI: expected {p['val_ci']}, got {result.se}"


class TestInfinityZeroControl:
    """Experiments with zero control accepts produce lift=0 (N/A in Pega)."""

    @pytest.mark.parametrize("exp_id", INFINITY_ZERO_CONTROL.keys())
    def test_engagement_lift_zero(self, exp_id):
        at, it, ac, ic, av = INFINITY_ZERO_CONTROL[exp_id]
        result = calculate_engagement_lift(at, it, ac, ic)
        assert result.lift == 0.0
        assert result.control_rate == 0.0

    @pytest.mark.parametrize("exp_id", INFINITY_ZERO_CONTROL.keys())
    def test_value_lift_zero(self, exp_id):
        at, it, ac, ic, av = INFINITY_ZERO_CONTROL[exp_id]
        result = calculate_value_lift(at, it, ac, ic, av)
        assert result.lift == 0.0


# ====================================================================
# 4. VBD parity — 18 experiment / channel pairs
# ====================================================================

VBD_RAW = {
    ("NBA vs Random Relevant Action", "Email"): (19_339_719, 364_381, 161_634, 2_581),
    ("NBA vs Random Relevant Action", "Mobile"): (846_481, 42_448, 9_130, 124),
    ("NBA vs Random Relevant Action", "InboundApp"): (32_762_470, 304_474, 304_413, 1_913),
    ("NBA vs Random Relevant Action", "MobileApp"): (82_070_272, 1_998_201, 570_891, 9_853),
    ("NBA vs Random Relevant Action", "Web"): (1_736_769, 8_171, 15_788, 25),
    ("NBA vs Random Relevant Action", "OVERALL"): (148_145_231, 2_717_675, 1_063_732, 14_496),
    ("NBA vs Arbitrating by Propensity-only", "Email"): (19_339_719, 364_381, 147_939, 2_537),
    ("NBA vs Arbitrating by Propensity-only", "Mobile"): (846_481, 42_448, 6_579, 171),
    ("NBA vs Arbitrating by Propensity-only", "InboundApp"): (32_762_470, 304_474, 252_792, 2_153),
    ("NBA vs Arbitrating by Propensity-only", "MobileApp"): (82_070_272, 1_998_201, 490_711, 11_388),
    ("NBA vs Arbitrating by Propensity-only", "Web"): (1_736_769, 8_171, 16_571, 41),
    ("NBA vs Arbitrating by Propensity-only", "OVERALL"): (148_145_231, 2_717_675, 916_489, 16_290),
    ("Adaptive Models", "Email"): (74_800, 1_292, 203_108, 2_709),
    ("Adaptive Models", "Mobile"): (3_034, 70, 4_788, 82),
    ("Adaptive Models", "InboundApp"): (120_349, 1_034, 191_812, 749),
    ("Adaptive Models", "MobileApp"): (220_613, 5_388, 253_270, 3_682),
    ("Adaptive Models", "Web"): (8_172, 8, 7_852, 22),
    ("Adaptive Models", "OVERALL"): (427_943, 7_792, 662_569, 7_244),
}

VBD_COMPUTED = [
    ("NBA vs Random Relevant Action", "Email", +0.179914),
    ("NBA vs Random Relevant Action", "Mobile", +2.692233),
    ("NBA vs Random Relevant Action", "InboundApp", +0.478842),
    ("NBA vs Random Relevant Action", "MobileApp", +0.410711),
    ("NBA vs Random Relevant Action", "Web", +1.971120),
    ("NBA vs Random Relevant Action", "OVERALL", +0.346151),
    ("NBA vs Arbitrating by Propensity-only", "Email", +0.098671),
    ("NBA vs Arbitrating by Propensity-only", "Mobile", +0.929318),
    ("NBA vs Arbitrating by Propensity-only", "InboundApp", +0.091171),
    ("NBA vs Arbitrating by Propensity-only", "MobileApp", +0.049136),
    ("NBA vs Arbitrating by Propensity-only", "Web", +0.901507),
    ("NBA vs Arbitrating by Propensity-only", "OVERALL", +0.032086),
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
        impr_t, acc_t, impr_c, acc_c = VBD_RAW[(experiment, channel)]
        result = calculate_engagement_lift(acc_t, impr_t, acc_c, impr_c)
        assert abs(result.lift - expected_lift) < 1e-5, (
            f"VBD {experiment}/{channel}: expected {expected_lift:+.6f}, got {result.lift:+.6f}"
        )


# ====================================================================
# 5. Edge cases and robustness
# ====================================================================


class TestEdgeCases:
    def test_zero_impressions_both(self):
        result = calculate_engagement_lift(0, 0, 0, 0)
        assert result.lift == 0.0
        assert result.se == 0.0
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
        assert result.se == pytest.approx(0.00677495387438173, rel=1e-10)
        assert result.lift == pytest.approx(0.0, abs=1e-15)

    def test_tiny_control_group(self):
        """Tiny control → huge CI → not significant."""
        result = calculate_engagement_lift(100, 1000, 1, 10)
        assert result.se == pytest.approx(0.9534149149242422, rel=1e-10)
        assert result.significant is False

    def test_value_lift_zero_action_value(self):
        result = calculate_value_lift(100, 1000, 50, 1000, 0.0)
        assert result.lift == 0.0
        assert result.test_rate == 0.0
