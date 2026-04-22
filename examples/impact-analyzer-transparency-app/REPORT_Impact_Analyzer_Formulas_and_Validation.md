# Impact Analyzer — Formula Reference# Impact Analyzer — Formulas, Validation & UI Report



**Date:** April 2026**Author:** Pega Data Scientist Tools Team  

**Date:** April 21, 2026  

---**Version:** 1.0  



## Table of Contents---



1. [Overview](#1-overview)## Table of Contents

2. [Experiment Design](#2-experiment-design)

3. [Engagement Metrics](#3-engagement-metrics)1. [Executive Summary](#1-executive-summary)

4. [Engagement Lift & Confidence Interval](#4-engagement-lift--confidence-interval)2. [What is the Impact Analyzer?](#2-what-is-the-impact-analyzer)

5. [Value Per Impression (VPI)](#5-value-per-impression-vpi)3. [Experiment Design](#3-experiment-design)

6. [Value Lift & Confidence Interval](#6-value-lift--confidence-interval)4. [Complete Formula Reference](#4-complete-formula-reference)

7. [Overall Health Score](#7-overall-health-score)   - 4.1 Engagement Metrics

8. [Required Sample Size](#8-required-sample-size)   - 4.2 Engagement Lift & Confidence Interval

   - 4.3 Value Per Impression (VPI)

---   - 4.4 Value Lift & Confidence Interval

   - 4.5 Overall Health Score

## 1. Overview   - 4.6 Required Sample Size

5. [Formula Validation](#5-formula-validation)

Pega's Impact Analyzer runs **five parallel A/B experiments** inside Customer6. [Transparent UI — What You See](#6-transparent-ui--what-you-see)

Decision Hub. Each experiment compares the full Next-Best-Action strategy7. [Data Flow Architecture](#7-data-flow-architecture)

(test arm) against a deliberately degraded strategy (control arm) to8. [Key Findings & Observations](#8-key-findings--observations)

measure whether NBA delivers statistically significant improvement.9. [Appendix: PR #608 Bug Fixes](#9-appendix-pr-608-bug-fixes)



This document describes every formula used to compute engagement lift,---

value lift, confidence intervals, and statistical significance.

## 1. Executive Summary

---

The Pega Impact Analyzer is a statistical experimentation framework embedded in Customer Decision Hub (CDH) that measures the effectiveness of Next-Best-Action (NBA) strategies through controlled A/B experiments. It runs **5 parallel experiments**, each comparing NBA against a degraded control strategy, and uses **binomial proportion tests** with **delta-method confidence intervals** to determine whether NBA delivers statistically significant lift in both engagement (accept rates) and value (revenue per impression).

## 2. Experiment Design

This report documents every formula used, how each was validated against Pega's production implementation, and what the transparent replica UI displays.

| # | Experiment | Control Arm | What It Isolates |

|---|-----------|-------------|------------------|---

| 1 | NBA vs Random Relevant Action | Random relevant action | Overall NBA value |

| 2 | NBA vs Propensity-Only | Propensity-only arbitration | Value of business levers |## 2. What is the Impact Analyzer?

| 3 | NBA vs No Levers | No levers applied | Value of lever optimization |

| 4 | Adaptive Model vs Random Propensity | Random propensity | Value of adaptive modeling |The Impact Analyzer answers one question: **"Is Next-Best-Action actually better than simpler strategies?"**

| 5 | NBA vs Eligibility Only | Eligibility criteria only | Value beyond basic eligibility |

It does this by randomly assigning a fraction of customer interactions to a **control arm** (degraded strategy) while the majority go to the **test arm** (full NBA). By comparing accept rates and value outcomes between the two arms, it determines with statistical confidence whether NBA is winning.

### Events

### Why 5 Experiments?

- **Container** — one impression (customer was shown an action)

- **Capture** — outcome recorded: Accepted, Clicked, Rejected, or IgnoredEach experiment isolates a different component of the NBA stack:

- **Accepts** = Accepted + Clicked

| # | Experiment | Test Arm | Control Arm | What It Measures |

---|---|-----------|----------|-------------|------------------|

| 1 | NBA vs Random Relevant Action | Full NBA | Random relevant action | Overall NBA value |

## 3. Engagement Metrics| 2 | NBA vs Propensity-Only | Full NBA | Propensity-only arbitration | Value of business levers |

| 3 | NBA vs No Levers | Full NBA | No levers applied | Value of lever optimization |

### Accept Rate| 4 | Adaptive Model vs Random Propensity | Adaptive models | Random propensity | Value of adaptive modeling |

| 5 | NBA vs Eligibility Only | Full NBA | Eligibility criteria only | Value beyond basic eligibility |

$$p = \frac{\text{Accepts}}{\text{Impressions}}$$

---

Computed separately for each arm ($p_t$ = test, $p_c$ = control).

## 3. Experiment Design

### Standard Error (Binomial)

### Arms

$$\text{SE}(p) = \sqrt{\frac{p \cdot (1 - p)}{n}}$$- **Test arm**: Receives the full NBA strategy (or adaptive models for experiment 4)

- **Control arm**: Receives a deliberately degraded strategy

Returns 0 when $n = 0$ or $p \in \{0, 1\}$.

### Events

---Each customer interaction generates two event types:

- **Container event** (`eventType: "container"`): Counts as one **impression** — the customer was shown an action

## 4. Engagement Lift & Confidence Interval- **Capture event** (`eventType: "capture"`): Records the outcome — `Accepted`, `Clicked`, `Rejected`, or `Ignored`



### Relative Lift### Key Metrics Per Arm

- **Impressions**: Count of container events

$$\text{Lift} = \frac{p_t - p_c}{p_c}$$- **Accepts**: Count of capture events with outcome `Accepted` or `Clicked`

- **Accepted Value**: Sum of `value` field from accepted capture events (used for VPI calculations)

A lift of 0.25 means the test arm's accept rate is 25 % higher than control.

---

### Standard Error of Lift (Delta Method)

## 4. Complete Formula Reference

$$\text{SE}(\text{Lift}) = \frac{1}{p_c} \sqrt{\text{SE}_t^2 + \left(\frac{p_t}{p_c}\right)^2 \cdot \text{SE}_c^2}$$

### 4.1 Engagement Metrics

This applies the delta method for error propagation of a ratio estimator.

#### Accept Rate (Proportion)

### 95 % Confidence Interval

$$p = \frac{\text{Accepts}}{\text{Impressions}}$$

$$\text{CI}_{95\%} = \text{Lift} \pm 1.96 \times \text{SE}(\text{Lift})$$

Computed separately for each arm:

### Statistical Significance- $p_t$ = test accept rate

- $p_c$ = control accept rate

$$\text{Significant} \iff (\text{Lift} - 1.96 \times \text{SE} > 0) \;\text{OR}\; (\text{Lift} + 1.96 \times \text{SE} < 0)$$

#### Standard Error (Binomial)

The result is significant when the entire confidence interval lies on one

side of zero.$$\text{SE}(p) = \sqrt{\frac{p \cdot (1 - p)}{n}}$$



---Where $n$ is the number of impressions. This is the standard error of a binomial proportion.



## 5. Value Per Impression (VPI)**Edge cases:**

- If $n = 0$: SE = 0

### Action Value- If $p = 0$ or $p = 1$: SE = 0 (degenerate case, no variance)



$$\text{AV} = \frac{\text{Total Accepted Value}}{\text{Accepts}}$$---



Average value per accepted interaction.### 4.2 Engagement Lift & Confidence Interval



### VPI#### Relative Lift



$$\text{VPI} = p \times \text{AV}$$$$\text{Lift} = \frac{p_t - p_c}{p_c}$$



### Variance (Bernoulli)This is the **relative** improvement of the test arm over the control arm. A lift of 0.25 means the test arm has 25% higher accept rate than control.



$$\text{Var}(\text{VPI}) = p \cdot (1 - p) \times \text{AV}^2$$**Edge case:** If $p_c = 0$, lift is undefined (returns `None`).



Each impression is a Bernoulli trial yielding $\text{AV}$ with probability#### Standard Error of Lift (Delta Method)

$p$ and 0 otherwise.

$$\text{SE}(\text{Lift}) = \frac{1}{p_c} \sqrt{\text{SE}_t^2 + \left(\frac{p_t}{p_c}\right)^2 \cdot \text{SE}_c^2}$$

### Standard Error

This uses the **delta method** for error propagation of the ratio $p_t / p_c$. The formula derives from:

$$\text{SE}(\text{VPI}) = \sqrt{\frac{\text{Var}(\text{VPI})}{n}}$$

$$\text{Var}\left(\frac{X}{Y}\right) \approx \frac{1}{\mu_Y^2}\left[\text{Var}(X) + \frac{\mu_X^2}{\mu_Y^2}\text{Var}(Y)\right]$$

---

Applied to our ratio where $X = p_t - p_c$ and the denominator is $p_c$.

## 6. Value Lift & Confidence Interval

#### 95% Confidence Interval

### Value Lift

$$\text{CI}_{95\%}(\text{Lift}) = \text{Lift} \pm z \times \text{SE}(\text{Lift})$$

$$\text{ValueLift} = \frac{\text{VPI}_t - \text{VPI}_c}{\text{VPI}_c}$$

Where $z = 1.96$ for 95% confidence level.

### Standard Error (Delta Method)

#### Statistical Significance

$$\text{SE}(\text{ValueLift}) = \frac{1}{\text{VPI}_c} \sqrt{\text{SE}_{\text{vpi}_t}^2 + \left(\frac{\text{VPI}_t}{\text{VPI}_c}\right)^2 \cdot \text{SE}_{\text{vpi}_c}^2}$$

The result is **statistically significant** if and only if the confidence interval does not contain zero:

### Confidence Interval and Significance

$$\text{Significant} \iff (\text{Lift} - 1.96 \times \text{SE} > 0) \;\text{OR}\; (\text{Lift} + 1.96 \times \text{SE} < 0)$$

$$\text{CI}_{95\%} = \text{ValueLift} \pm 1.96 \times \text{SE}(\text{ValueLift})$$

**Important:** Pega uses $z = 1.96$ **only at the display/reporting level**. The underlying computation uses the exact float `1.96`, not a rounded value.

Significance follows the same rule: CI must not contain 0.

---

---

### 4.3 Value Per Impression (VPI)

## 7. Overall Health Score

VPI measures revenue efficiency — how much value each impression generates.

### Per-Experiment Scoring (max 3 points each)

#### Action Value

| Points | Condition |

|--------|-----------|$$\text{AV} = \frac{\text{Total Accepted Value}}{\text{Accepts}}$$

| +1 | Control arm has at least 1 impression |

| +1 | Both arms have ≥ minimum impressions |This is the average value per accepted interaction (average deal size / revenue per conversion).

| +1 | Result is statistically significant |

#### VPI Calculation

### Aggregate Score

$$\text{VPI} = p \times \text{AV} = \frac{\text{Accepts}}{\text{Impressions}} \times \text{AV}$$

$$\text{Score} = \frac{\text{total\_points}}{3 \times N_{\text{experiments}}}$$

Equivalently: $\text{VPI} = \frac{\text{Total Accepted Value}}{\text{Impressions}}$

| Score | Label |

|-------|-------|#### VPI Variance (Bernoulli)

| ≥ 70 % | Good |

| ≥ 35 % | Fair |$$\text{Var}(\text{VPI}) = p \cdot (1 - p) \times \text{AV}^2$$

| < 35 % | Not enough data |

This is the **Bernoulli per-observation variance**. Each impression is a Bernoulli trial: with probability $p$ it yields value $\text{AV}$, and with probability $(1-p)$ it yields 0.

---

> **Critical note:** This is a Bernoulli model, NOT a Poisson model. The original PR #608 incorrectly used a Poisson-like formula. See [Appendix](#9-appendix-pr-608-bug-fixes).

## 8. Required Sample Size

#### VPI Standard Error

$$n = \frac{(z_\alpha + z_\beta)^2 \cdot p(1-p)}{\delta^2}$$

$$\text{SE}(\text{VPI}) = \sqrt{\frac{\text{Var}(\text{VPI})}{n}}$$

| Parameter | Default | Description |

|-----------|---------|-------------|---

| $z_\alpha$ | 1.96 | 95 % confidence (two-sided) |

| $z_\beta$ | 0.84 | 80 % power |### 4.4 Value Lift & Confidence Interval

| $\delta$ | $p \times \text{MDE}$ | Minimum detectable effect as absolute change |

#### Value Lift

---

$$\text{ValueLift} = \frac{\text{VPI}_t - \text{VPI}_c}{\text{VPI}_c}$$

*For an interactive walkthrough of these formulas with live data, run the

[Transparency App](README.md).***Edge case:** If $\text{VPI}_c = 0$, value lift is undefined.


#### Value Lift Standard Error (Delta Method)

$$\text{SE}(\text{ValueLift}) = \frac{1}{\text{VPI}_c} \sqrt{\text{SE}_{\text{vpi}_t}^2 + \left(\frac{\text{VPI}_t}{\text{VPI}_c}\right)^2 \cdot \text{SE}_{\text{vpi}_c}^2}$$

Same delta-method structure as the engagement lift SE.

#### Value Lift Confidence Interval

$$\text{CI}_{95\%}(\text{ValueLift}) = \text{ValueLift} \pm 1.96 \times \text{SE}(\text{ValueLift})$$

Significance is determined the same way: CI must not contain 0.

---

### 4.5 Overall Health Score

The health score is a composite metric that summarizes how well the Impact Analyzer experiments are performing.

#### Per-Experiment Scoring (max 3 points each)

| Points | Condition |
|--------|-----------|
| +1 | Control arm has at least 1 impression |
| +1 | Both arms have ≥ $n_{\min}$ impressions (typically 100) |
| +1 | Result is statistically significant |

#### Aggregate Score

$$\text{Score} = \frac{\text{total\_points}}{3 \times N_{\text{experiments}}}$$

#### Health Label Mapping

| Score Range | Label | Color |
|-------------|-------|-------|
| ≥ 70% | Good | Green |
| ≥ 35% | Fair | Orange |
| < 35% | Not enough data | Red |

---

### 4.6 Required Sample Size

For power analysis (planning how many impressions are needed):

$$n = \frac{(z_\alpha + z_\beta)^2 \cdot p(1-p)}{\delta^2}$$

Where:
- $z_\alpha = 1.96$ (95% confidence, two-sided)
- $z_\beta = 0.84$ (80% power)
- $\delta = p \times \text{MDE}$ (minimum detectable effect as absolute change)

---

## 5. Formula Validation

### Methodology

All formulas were validated by building an independent Python implementation (`ia_engine.py`) and comparing outputs against Pega's production calculations. The validation suite covers:

1. **Unit-level function tests** — Each atomic function (`_accept_rate`, `_binomial_se`, `_lift`, `_lift_se`, `_significant`) tested in isolation
2. **End-to-end scenario tests** — Full event payloads fed through the engine, outputs compared against known Pega results
3. **Edge case coverage** — Zero impressions, zero accepts, single impression, equal rates, extreme rates (0.001, 0.999)
4. **Numerical precision** — All comparisons use 10+ decimal places to ensure exact match

### Validation Results

| Test Category | Tests | Passed | Status |
|--------------|-------|--------|--------|
| Accept Rate computation | 15 | 15 | PASS |
| Binomial SE computation | 15 | 15 | PASS |
| Relative Lift | 12 | 12 | PASS |
| Lift SE (Delta Method) | 15 | 15 | PASS |
| Significance determination | 18 | 18 | PASS |
| VPI & Value Lift | 15 | 15 | PASS |
| End-to-end scenarios | 15 | 15 | PASS |
| Edge cases (zero/null) | 15 | 15 | PASS |
| **TOTAL** | **120** | **120** | **ALL PASS** |

### Key Validation Checks

1. **Delta method correctness**: Verified that SE(Lift) uses the proper error propagation formula `(1/p_c) * sqrt(SE_t² + (p_t/p_c)² * SE_c²)`, NOT the simplified `sqrt(SE_t² + SE_c²) / p_c`

2. **z-value application**: Confirmed z = 1.96 is applied **only** at the CI/significance level, not baked into the SE calculation

3. **Rounding behavior**: Verified that `round(4)` on `lift_se` is applied only for display purposes; all intermediate calculations use full precision

4. **Bernoulli VPI variance**: Confirmed `Var(VPI) = p(1-p) * AV²`, not Poisson-based variance

5. **Significance boundary**: Tested exact boundary cases where CI just touches zero to confirm strict inequality (`>` not `>=`)

---

## 6. Transparent UI — What You See

The transparent replica UI (`app.py`) mirrors Pega's Impact Analyzer layout while adding full formula visibility. Here is what each section shows:

### Section 1: Overall Health Gauge

- **Gauge chart** showing the health score as a percentage (0–100%)
- Color-coded zones: Red (0–35%), Orange (35–70%), Green (70–100%)
- Health label: "Good", "Fair", or "Not enough data"
- Expandable explanation showing the per-experiment scoring breakdown

### Section 2: Experiment Cards (Active / Inactive Tabs)

Each active experiment card displays:

#### Header
- Experiment short name (e.g., "NBA vs Random relevant action")
- ACTIVE / INACTIVE badge

#### Significance Callout
- Green banner: "Statistically significant at 95% confidence"
- Yellow banner: "Not yet significant — more data needed"
- Blue banner: "Not enough data for insights"

#### Key Metrics Row
| Metric | Description |
|--------|-------------|
| ENG. LIFT | Engagement lift as percentage (green if positive, red if negative) |
| VALUE LIFT | Value lift as percentage |
| TEST n | Number of test impressions |
| CTRL n | Number of control impressions |

#### Arm Comparison Table
Shows Impressions, Accepts, and Accept Rate for both Test and Control arms side by side.

#### Charts
1. **Lift over time**: Line chart showing how engagement lift evolves as more events arrive, with a shaded 95% CI band
2. **CI bar**: Horizontal bar showing the current confidence interval with a diamond marker at the point estimate. Dashed line at 0 for reference.

#### Lift Details (Expandable — Navy Banner)

When expanded, shows **5 step-by-step formula derivations** with LaTeX-rendered equations and substituted numerical values:

| Step | What It Shows |
|------|--------------|
| Step 1 — Accept Rates | Formula + computed p_test and p_ctrl |
| Step 2 — Standard Errors | Formula + computed SE_test and SE_ctrl |
| Step 3 — Engagement Lift | Formula + computed lift value |
| Step 4 — Confidence Interval | Delta method formula + SE(Lift), CI bounds, significance verdict |
| Step 5 — Value Lift (VPI) | VPI formula chain: AV → VPI → Var → SE → ValueLift → CI |

### Section 3: Summary Table

A dataframe showing all 5 experiments with:
- Experiment name
- Test/Control impressions
- Test/Control accept rates
- Engagement lift
- SE(Lift) rounded to 4 decimal places
- Significant: Yes / No / -

### Section 4: Full Formula Reference

A comprehensive expandable section with all formulas rendered in LaTeX:
- Engagement Metrics (p, SE, CI)
- Lift (relative lift, delta-method SE, CI, significance rule)
- VPI (VPI, Var, SE, ValueLift, CI)
- Required Sample Size (power analysis)
- Overall Health Score (scoring rubric)

### Sidebar: Data Sources

Four modes for feeding data into the analyzer:

| Mode | Description |
|------|-------------|
| Sample data | Pre-loaded 30 events across all 5 experiments |
| Live generator | Simulates streaming data with 4 scenario profiles |
| File upload | Load PDC JSON exports or VBD files |
| Live bridge | Poll a REST endpoint for real-time events |

#### Generator Profiles

| Profile | Behavior |
|---------|----------|
| Realistic NBA | NBA wins with moderate lift (~5–15%) |
| Clear NBA winner | NBA dominates with large lift (>20%) |
| Neck-and-neck | Nearly identical rates, no significance |
| NBA losing | Control outperforms test (negative lift) |

---

## 7. Data Flow Architecture

```
Events (JSON)
    │
    ▼
parse_payload()          ← Validates each event (experimentKey, arm, eventType, outcome)
    │
    ▼
calculate_results()      ← Aggregates into ArmMetrics per experiment
    │                       Computes: accept rates, SE, lift, lift_se, significance
    │                       Computes: action value, VPI, VPI variance, VPI SE, value lift
    │
    ├──▶ TestResult[]    ← One per experiment (5 total)
    │
    ▼
build_timeseries()       ← Cumulative snapshots every N events for charting
    │
    ▼
compute_health()         ← Scores all experiments → HealthScore (label, color, score)
    │
    ▼
Streamlit UI             ← Renders gauges, cards, charts, tables, formulas
```

---

## 8. Key Findings & Observations

### Formula Accuracy
All 120 validation tests pass with exact numerical match to Pega's implementation. The formulas are mathematically sound and follow standard statistical methodology for A/B testing with binomial outcomes.

### Delta Method is Critical
The delta method for computing SE(Lift) is the most complex formula in the system. It correctly handles the fact that lift is a **ratio** of two correlated random variables. Simplifying to `SE(diff) / p_c` would be incorrect.

### z = 1.96 Only at Display Level
An important implementation detail: the z-value is NOT embedded in any intermediate calculation. It appears only when computing the final CI bounds and significance check. This is correct practice.

### Rounding Discipline
The engine computes with full float64 precision throughout. The `round(4)` on `lift_se` is applied **only** for the summary table display column. All other outputs use 10+ decimal places.

### Value Lift Requires Capture Values
VPI calculations depend on the `value` field in capture events. If no values are provided (all `null`), action value defaults to 0 and VPI/value lift will be 0 or undefined. The sample data and generator profiles include value fields to demonstrate this.

---

## 9. Appendix: PR #608 Bug Fixes

During validation, three bugs were identified in the original `pega-datascientist-tools` PR #608 (`feature/ia-statistics-module` branch):

### Bug 1: Engagement CI Inflated ~2×

**Location:** `_engagement_ci()` function  
**Issue:** The z² term was applied twice — once inside the SE formula and once when computing the CI width.  
**Impact:** Confidence intervals were approximately 2× too wide, making results appear less significant than they actually are.  
**Fix:** Remove the redundant z multiplication; apply z only once at the CI level.

### Bug 2: Value Lift Used Wrong Variance Model

**Location:** `_value_lift()` function  
**Issue:** Used a Poisson-like variance formula instead of the correct Bernoulli formula: `Var = p(1-p) × AV²`  
**Impact:** Value lift CIs were mathematically inconsistent with the binomial model used for engagement.  
**Fix:** Replace with Bernoulli variance: `p * (1-p) * AV²`

### Bug 3: Premature Rounding

**Location:** Multiple functions using `round(4)` on intermediate values  
**Issue:** Rounding SE to 4 decimal places before using it in subsequent CI calculations introduced compounding precision loss.  
**Impact:** Final CI bounds could differ from Pega's production values by up to 0.01%.  
**Fix:** Apply `round(4)` only on the final display value; use full precision for all intermediate computations.

All three bugs were corrected in the files prepared in `pr-review/final/`:
- `statistics.py` — Corrected module
- `test_ia_statistics.py` — 120+ test cases validating the fixes
- `PR_DESCRIPTION.md` — Detailed description for the pull request

---

*End of Report*
