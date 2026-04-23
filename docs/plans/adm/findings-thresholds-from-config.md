# Move bespoke Analysis thresholds to MetricLimits.csv

**Priority:** P2
**Files:**
- `python/pdstools/adm/Analysis.py`
- `python/pdstools/resources/MetricLimits.csv`

## Problem

After the v5 cleanup, the cohort-level cutoffs in `Analysis.findings()`
(maturity buckets, low-performance band, never-used severity) read from
`MetricLimits.csv`. A handful of agent-specific heuristics still carry
inline magic numbers:

| Location | Literal | What it gates |
|---|---|---|
| `_compute_health_headline` (~L216-229) | 25 %, 55, 62 | 🩺 NEEDS ATTENTION / MIXED / HEALTHY classifier |
| `_check_average_performance` (~L772-796) | 58, 62, 70 | AUC verdict tier (low/borderline/healthy/strong) |
| `_check_response_distribution` (~L930-962) | 0.7, 0.9 | Gini skew warning vs critical |
| `_check_predictor_overview` (~L1168) | 0.5 | "predictors with >50 % missing" warning |
| `_check_predictor_overview` (~L1277/1312/1326) | 0.1, 0.10, 0.005 | Lift / control-percentage bands |

## Proposed approach

Add new rows to `MetricLimits.csv` for each:

- `HeadlineNeverUsedPercentage` (bp_max=0.10, max=0.25)
- `HeadlineAverageModelPerformance` (min=0.55, bp_min=0.62, bp_max=0.70)
- `ResponseDistributionGini` (bp_max=0.7, max=0.9)
- `PositiveResponseDistributionGini` (bp_max=0.7, max=0.9)
- `PredictorMissingPercentage` (bp_max=0.3, max=0.5)
- `EngagementLift` already exists — reuse for the 0.1 cutoff
- `ControlGroupPercentage` (min=0.005, bp_min=0.05, bp_max=0.10, max=0.20)

Then have `Analysis.py` look them up via `MetricLimits.minimum(...)` /
`MetricLimits.best_practice_max(...)`. Keeps the agent's policy in one
inspectable file and lets us retune without code changes.

## Out of scope (defensible math constants)

- `> 0.5` AUC random-baseline floor
- `== 0` count comparisons
- `pct == 100` ceilings

These will never change.
