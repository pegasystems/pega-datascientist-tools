from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Literal

ExperimentKey = Literal[
    "nba_random_relevant_action",
    "nba_propensity_only",
    "nba_no_levers",
    "adaptive_vs_random_propensity",
    "nba_eligibility_only",
]
Arm = Literal["test", "control"]
EventType = Literal["container", "capture"]
CaptureOutcome = Literal["Accepted", "Clicked", "Rejected", "Ignored"]

Z_95 = 1.96

EXPERIMENTS: list[dict[str, str]] = [
    {
        "key": "nba_random_relevant_action",
        "title": "How is Next-Best-Action performing against a random relevant action?",
        "testLabel": "NBA",
        "controlLabel": "Random relevant action",
    },
    {
        "key": "nba_propensity_only",
        "title": "How is Next-Best-Action performing against arbitrating by propensity-only?",
        "testLabel": "NBA",
        "controlLabel": "Propensity-only",
    },
    {
        "key": "nba_no_levers",
        "title": "How is Next-Best-Action performing against arbitrating with no levers?",
        "testLabel": "NBA",
        "controlLabel": "No levers",
    },
    {
        "key": "adaptive_vs_random_propensity",
        "title": "How is adaptive model selection performing against random propensity?",
        "testLabel": "Adaptive model",
        "controlLabel": "Random propensity",
    },
    {
        "key": "nba_eligibility_only",
        "title": "How is Next-Best-Action performing against applying only eligibility criteria?",
        "testLabel": "NBA",
        "controlLabel": "Eligibility only",
    },
]

VALID_EXPERIMENT_KEYS = {e["key"] for e in EXPERIMENTS}
VALID_ARMS = {"test", "control"}
VALID_EVENT_TYPES = {"container", "capture", "pdc_summary"}
VALID_OUTCOMES = {"Accepted", "Clicked", "Rejected", "Ignored"}


@dataclass
class ArmMetrics:
    impressions: int = 0
    accepts: int = 0
    accepted_value: float = 0.0


@dataclass
class TestResult:
    definition: dict[str, str]
    test: ArmMetrics
    control: ArmMetrics
    accept_rate_test: float
    accept_rate_control: float
    se_test: float
    se_control: float
    lift: float | None
    lift_se: float | None
    lift_se_rounded4: float | None
    significant: bool | None
    note: str | None
    formulas: dict[str, str]
    # VPI (Value Per Impression) fields
    action_value_test: float
    action_value_control: float
    vpi_test: float
    vpi_control: float
    vpi_var_test: float
    vpi_var_control: float
    vpi_se_test: float
    vpi_se_control: float
    value_lift: float | None
    value_lift_se: float | None
    value_lift_significant: bool | None


def validate_event(event: dict[str, Any], idx: int | None = None) -> dict[str, Any]:
    prefix = f"Event {idx}: " if idx is not None else ""

    experiment_key = event.get("experimentKey")
    if experiment_key not in VALID_EXPERIMENT_KEYS:
        raise ValueError(f"{prefix}invalid experimentKey '{experiment_key}'")

    arm = event.get("arm")
    if arm not in VALID_ARMS:
        raise ValueError(f"{prefix}invalid arm '{arm}'")

    event_type = event.get("eventType")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"{prefix}invalid eventType '{event_type}'")

    outcome = event.get("outcome")
    if outcome is not None and outcome not in VALID_OUTCOMES:
        raise ValueError(f"{prefix}invalid outcome '{outcome}'")

    value = event.get("value")
    if value is not None and not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}value must be numeric when provided")

    result = {
        "timestamp": event.get("timestamp"),
        "customerId": event.get("customerId"),
        "experimentKey": experiment_key,
        "arm": arm,
        "eventType": event_type,
        "outcome": outcome,
        "value": float(value) if value is not None else None,
    }

    # Pass through PDC summary fields
    if event_type == "pdc_summary":
        for k in ("impressions_test", "accepts_test", "impressions_control",
                   "accepts_control", "accepted_value_test", "accepted_value_control",
                   "engagement_lift", "value_lift",
                   "engagement_lift_interval", "value_lift_interval",
                   "is_significant", "color", "heading", "guidance", "channel"):
            if k in event:
                result[k] = event[k]

    return result


def parse_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Payload must be a JSON array of events")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Event {idx}: must be an object")
        out.append(validate_event(item, idx))
    return out


def _accept_rate(accepts: int, impressions: int) -> float:
    if impressions <= 0:
        return 0.0
    return accepts / impressions


def _binomial_se(rate: float, impressions: int) -> float:
    if impressions <= 0:
        return 0.0
    if rate <= 0 or rate >= 1:
        return 0.0
    return sqrt(rate * (1 - rate) / impressions)


def _lift(rate_test: float, rate_control: float) -> float | None:
    if rate_control <= 0:
        return None
    return (rate_test - rate_control) / rate_control


def _lift_se(rate_test: float, rate_control: float, se_test: float, se_control: float) -> float | None:
    if rate_control <= 0:
        return None
    term1 = (se_test / rate_control) ** 2
    term2 = ((rate_test * se_control) / (rate_control**2)) ** 2
    return sqrt(term1 + term2)


def _significant(lift: float | None, lift_se: float | None, z: float = Z_95) -> bool | None:
    if lift is None or lift_se is None:
        return None
    half = z * lift_se
    return (lift - half > 0) or (lift + half < 0)


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _fmt_num(value: float | None, digits: int = 10) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def calculate_results(events: list[dict[str, Any]]) -> list[TestResult]:
    bucket: dict[str, dict[str, ArmMetrics]] = {
        e["key"]: {"test": ArmMetrics(), "control": ArmMetrics()} for e in EXPERIMENTS
    }

    for event in events:
        key = event["experimentKey"]
        arm = event["arm"]
        metrics = bucket[key][arm]

        if event["eventType"] == "pdc_summary":
            # PDC summary: pre-aggregated counts
            bucket[key]["test"].impressions += event.get("impressions_test", 0)
            bucket[key]["test"].accepts += event.get("accepts_test", 0)
            bucket[key]["control"].impressions += event.get("impressions_control", 0)
            bucket[key]["control"].accepts += event.get("accepts_control", 0)
            # Use pre-computed accepted values (different AV per arm for value lift)
            bucket[key]["test"].accepted_value += event.get("accepted_value_test", event.get("accepts_test", 0) * 75.0)
            bucket[key]["control"].accepted_value += event.get("accepted_value_control", event.get("accepts_control", 0) * 75.0)
            continue

        if event["eventType"] == "container":
            metrics.impressions += 1

        if event["eventType"] == "capture" and event.get("outcome") in {"Accepted", "Clicked"}:
            metrics.accepts += 1
            metrics.accepted_value += event.get("value") or 0.0

    results: list[TestResult] = []

    for definition in EXPERIMENTS:
        key = definition["key"]
        test = bucket[key]["test"]
        control = bucket[key]["control"]

        rate_t = _accept_rate(test.accepts, test.impressions)
        rate_c = _accept_rate(control.accepts, control.impressions)

        se_t = _binomial_se(rate_t, test.impressions)
        se_c = _binomial_se(rate_c, control.impressions)

        lift = _lift(rate_t, rate_c)
        lift_se = _lift_se(rate_t, rate_c, se_t, se_c)
        lift_se_round = _round4(lift_se)
        significant = _significant(lift, lift_se)

        note = None
        if control.impressions == 0:
            note = "No control impressions available; lift cannot be computed."
        elif rate_c == 0:
            note = "Control accept rate is 0; relative lift is undefined (N/A)."

        # -- VPI (Value Per Impression) calculations -----------------------
        # Action value = total accepted value / accepts (average value per accept)
        av_t = (test.accepted_value / test.accepts) if test.accepts > 0 else 0.0
        av_c = (control.accepted_value / control.accepts) if control.accepts > 0 else 0.0

        vpi_t = rate_t * av_t
        vpi_c = rate_c * av_c

        # Bernoulli variance: Var(VPI) = p(1-p) * AV^2
        vpi_var_t = rate_t * (1 - rate_t) * (av_t ** 2)
        vpi_var_c = rate_c * (1 - rate_c) * (av_c ** 2)

        # SE(VPI) = sqrt(Var / n)
        vpi_se_t = sqrt(vpi_var_t / test.impressions) if test.impressions > 0 and vpi_var_t > 0 else 0.0
        vpi_se_c = sqrt(vpi_var_c / control.impressions) if control.impressions > 0 and vpi_var_c > 0 else 0.0

        # Value lift = (VPI_test - VPI_ctrl) / VPI_ctrl
        if vpi_c > 0:
            val_lift = (vpi_t - vpi_c) / vpi_c
            # Delta-method SE for value lift
            term1 = (vpi_se_t / vpi_c) ** 2
            term2 = ((vpi_t * vpi_se_c) / (vpi_c ** 2)) ** 2
            val_lift_se = sqrt(term1 + term2)
            val_lift_sig = _significant(val_lift, val_lift_se)
        else:
            val_lift = None
            val_lift_se = None
            val_lift_sig = None

        formulas = {
            "accept_rate_test": (
                f"p_t = accepts_t / impressions_t = {test.accepts} / {test.impressions} = {_fmt_num(rate_t, 12)}"
            ),
            "accept_rate_control": (
                f"p_c = accepts_c / impressions_c = {control.accepts} / {control.impressions} = {_fmt_num(rate_c, 12)}"
            ),
            "se_test": (
                f"SE_t = sqrt(p_t*(1-p_t)/n_t) = sqrt({_fmt_num(rate_t, 12)}*(1-{_fmt_num(rate_t, 12)})/{test.impressions}) = {_fmt_num(se_t, 12)}"
            ),
            "se_control": (
                f"SE_c = sqrt(p_c*(1-p_c)/n_c) = sqrt({_fmt_num(rate_c, 12)}*(1-{_fmt_num(rate_c, 12)})/{control.impressions}) = {_fmt_num(se_c, 12)}"
            ),
            "lift": (
                "Lift = (p_t - p_c) / p_c = N/A (control rate is 0)"
                if lift is None
                else f"Lift = (p_t - p_c) / p_c = ({_fmt_num(rate_t, 12)} - {_fmt_num(rate_c, 12)}) / {_fmt_num(rate_c, 12)} = {_fmt_num(lift, 12)}"
            ),
            "lift_se": (
                "SE_lift = sqrt((SE_t/p_c)^2 + (p_t*SE_c/p_c^2)^2) = N/A (control rate is 0)"
                if lift_se is None
                else f"SE_lift = sqrt((SE_t/p_c)^2 + (p_t*SE_c/p_c^2)^2) = {_fmt_num(lift_se, 12)}; rounded(4dp) = {_fmt_num(lift_se_round, 4)}"
            ),
            "significance": (
                "significant = N/A"
                if significant is None
                else f"significant = (lift - 1.96*SE_lift > 0) OR (lift + 1.96*SE_lift < 0) => {'TRUE' if significant else 'FALSE'}"
            ),
        }

        results.append(
            TestResult(
                definition=definition,
                test=test,
                control=control,
                accept_rate_test=rate_t,
                accept_rate_control=rate_c,
                se_test=se_t,
                se_control=se_c,
                lift=lift,
                lift_se=lift_se,
                lift_se_rounded4=lift_se_round,
                significant=significant,
                note=note,
                formulas=formulas,
                action_value_test=av_t,
                action_value_control=av_c,
                vpi_test=vpi_t,
                vpi_control=vpi_c,
                vpi_var_test=vpi_var_t,
                vpi_var_control=vpi_var_c,
                vpi_se_test=vpi_se_t,
                vpi_se_control=vpi_se_c,
                value_lift=val_lift,
                value_lift_se=val_lift_se,
                value_lift_significant=val_lift_sig,
            )
        )

    return results


def as_percent(value: float | None) -> str:
    return _fmt_pct(value)


def as_num(value: float | None, digits: int = 6) -> str:
    return _fmt_num(value, digits)


# ---------------------------------------------------------------------------
# Time-series tracking: build cumulative snapshots as events arrive
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """One cumulative snapshot for a single experiment at a point in time."""
    event_index: int
    impressions_test: int
    impressions_control: int
    accepts_test: int
    accepts_control: int
    rate_test: float
    rate_control: float
    lift: float | None
    lift_se: float | None
    ci_lower: float | None
    ci_upper: float | None
    significant: bool | None
    # Value lift fields
    value_lift: float | None = None
    value_lift_se: float | None = None
    value_ci_lower: float | None = None
    value_ci_upper: float | None = None
    value_significant: bool | None = None


@dataclass
class HealthScore:
    """Overall IA health assessment."""
    label: str  # "Good", "Fair", "Not enough data"
    score: float  # 0..1 (for gauge)
    color: str
    reasons: list[str]
    formula_steps: list[str]


def compute_health(results: list[TestResult]) -> HealthScore:
    """Compute overall health the same way Pega does.

    Logic (reverse-engineered from Infinity):
    - Each experiment scores points:
        +1 if it has any control impressions
        +1 if it meets minimum sample size (n_min ≈ 381 per arm for 5% baseline, 50% MDE)
        +1 if the result is statistically significant
    - Total = sum / (3 × num_experiments), mapped to Good/Fair/Not enough data.

    n_min formula:  n = (z_α + z_β)² · p(1-p) / (p·MDE)²
                    For p=0.05, MDE=0.5: n ≈ 381
    """
    from math import ceil

    n_experiments = len(results)
    if n_experiments == 0:
        return HealthScore("Not enough data", 0.0, "#e74c3c", ["No experiments configured"], [])

    z_alpha = Z_95
    z_beta = 0.84  # 80% power
    baseline = 0.05
    mde = 0.50  # 50% relative change
    delta = baseline * mde
    n_min = ceil((z_alpha + z_beta) ** 2 * baseline * (1 - baseline) / delta ** 2)

    reasons: list[str] = []
    formula_steps: list[str] = [
        f"Minimum sample size per arm (for baseline p={baseline}, MDE={mde*100:.0f}%, power=80%, α=0.05):",
        f"  n_min = (z_α + z_β)² · p(1−p) / (p·MDE)²",
        f"       = ({z_alpha} + {z_beta})² × {baseline}×{1-baseline} / ({delta})²",
        f"       = {(z_alpha+z_beta)**2:.4f} × {baseline*(1-baseline):.4f} / {delta**2:.6f}",
        f"       = ⌈{(z_alpha+z_beta)**2 * baseline*(1-baseline) / delta**2:.2f}⌉ = {n_min}",
        "",
        "Health scoring (per experiment, max 3 points each):",
        "  +1  if control arm has impressions",
        "  +1  if BOTH arms have ≥ n_min impressions",
        "  +1  if result is statistically significant",
        "",
    ]

    total_points = 0
    max_points = 3 * n_experiments

    for r in results:
        exp_name = r.definition["title"][:60]
        pts = 0

        has_ctrl = r.control.impressions > 0
        if has_ctrl:
            pts += 1

        meets_sample = (r.test.impressions >= n_min and r.control.impressions >= n_min)
        if meets_sample:
            pts += 1

        is_sig = r.significant is True
        if is_sig:
            pts += 1

        formula_steps.append(
            f"  {exp_name}:  ctrl={r.control.impressions} ({'✓' if has_ctrl else '✗'})  "
            f"n≥{n_min}? T={r.test.impressions},C={r.control.impressions} ({'✓' if meets_sample else '✗'})  "
            f"sig={'✓' if is_sig else '✗'}  → {pts}/3"
        )

        if not has_ctrl:
            reasons.append(f"{exp_name}: no control impressions")
        elif not meets_sample:
            reasons.append(f"{exp_name}: needs ≥{n_min} impressions per arm")

        total_points += pts

    score = total_points / max_points if max_points > 0 else 0.0
    formula_steps.append("")
    formula_steps.append(f"Total: {total_points} / {max_points} = {score:.2%}")

    if score >= 0.7:
        label, color = "Good", "#2ecc71"
    elif score >= 0.35:
        label, color = "Fair", "#f39c12"
    else:
        label, color = "Not enough data", "#e74c3c"
        if not reasons:
            reasons.append("Please continue to monitor as Impact Analyzer processes incoming data.")

    formula_steps.append(f"Health = {label} (threshold: ≥70% Good, ≥35% Fair, else Not enough data)")

    return HealthScore(label=label, score=score, color=color, reasons=reasons, formula_steps=formula_steps)


def build_timeseries(events: list[dict[str, Any]]) -> dict[str, list[Snapshot]]:
    """Walk through events chronologically, yielding cumulative snapshots per experiment.

    Returns a dict mapping experiment key -> list of Snapshots (one per relevant event).
    """
    bucket: dict[str, dict[str, ArmMetrics]] = {
        e["key"]: {"test": ArmMetrics(), "control": ArmMetrics()} for e in EXPERIMENTS
    }
    series: dict[str, list[Snapshot]] = {e["key"]: [] for e in EXPERIMENTS}

    for idx, event in enumerate(events):
        key = event["experimentKey"]
        arm = event["arm"]
        metrics = bucket[key][arm]

        if event["eventType"] == "pdc_summary":
            bucket[key]["test"].impressions += event.get("impressions_test", 0)
            bucket[key]["test"].accepts += event.get("accepts_test", 0)
            bucket[key]["control"].impressions += event.get("impressions_control", 0)
            bucket[key]["control"].accepts += event.get("accepts_control", 0)
            bucket[key]["test"].accepted_value += event.get("accepted_value_test", event.get("accepts_test", 0) * 75.0)
            bucket[key]["control"].accepted_value += event.get("accepted_value_control", event.get("accepts_control", 0) * 75.0)
        elif event["eventType"] == "container":
            metrics.impressions += 1
        elif event["eventType"] == "capture" and event.get("outcome") in {"Accepted", "Clicked"}:
            metrics.accepts += 1
            metrics.accepted_value += event.get("value") or 0.0

        test = bucket[key]["test"]
        control = bucket[key]["control"]

        rate_t = _accept_rate(test.accepts, test.impressions)
        rate_c = _accept_rate(control.accepts, control.impressions)
        se_t = _binomial_se(rate_t, test.impressions)
        se_c = _binomial_se(rate_c, control.impressions)

        # Engagement lift
        l = _lift(rate_t, rate_c)
        lse = _lift_se(rate_t, rate_c, se_t, se_c)
        sig = _significant(l, lse)

        ci_lo: float | None = None
        ci_hi: float | None = None
        if l is not None and lse is not None:
            ci_lo = l - Z_95 * lse
            ci_hi = l + Z_95 * lse

        # Value lift (VPI-based)
        av_t = (test.accepted_value / test.accepts) if test.accepts > 0 else 0.0
        av_c = (control.accepted_value / control.accepts) if control.accepts > 0 else 0.0
        vpi_t = rate_t * av_t
        vpi_c = rate_c * av_c

        vl: float | None = None
        vl_se: float | None = None
        vl_lo: float | None = None
        vl_hi: float | None = None
        vl_sig: bool | None = None
        if vpi_c > 0:
            vl = (vpi_t - vpi_c) / vpi_c
            vpi_var_t = rate_t * (1 - rate_t) * (av_t ** 2)
            vpi_var_c = rate_c * (1 - rate_c) * (av_c ** 2)
            vpi_se_t = sqrt(vpi_var_t / test.impressions) if test.impressions > 0 and vpi_var_t > 0 else 0.0
            vpi_se_c = sqrt(vpi_var_c / control.impressions) if control.impressions > 0 and vpi_var_c > 0 else 0.0
            t1 = (vpi_se_t / vpi_c) ** 2
            t2 = ((vpi_t * vpi_se_c) / (vpi_c ** 2)) ** 2
            vl_se = sqrt(t1 + t2)
            vl_sig = _significant(vl, vl_se)
            vl_lo = vl - Z_95 * vl_se
            vl_hi = vl + Z_95 * vl_se

        series[key].append(Snapshot(
            event_index=idx,
            impressions_test=test.impressions,
            impressions_control=control.impressions,
            accepts_test=test.accepts,
            accepts_control=control.accepts,
            rate_test=rate_t,
            rate_control=rate_c,
            lift=l,
            lift_se=lse,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            significant=sig,
            value_lift=vl,
            value_lift_se=vl_se,
            value_ci_lower=vl_lo,
            value_ci_upper=vl_hi,
            value_significant=vl_sig,
        ))

    return series
