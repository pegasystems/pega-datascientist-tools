# AUC roll-up weighting: statistical comparison of aggregation methods

Priority: P2

Files touched: `scripts/batch_healthcheck.py`,
`python/tests/healthcheck/test_batch_healthcheck.py`, possibly
`python/pdstools/utils/cdh_utils/_metrics.py` (no new library API
expected — existing functions are sufficient, see below).

## Problem

`pdstools` currently rolls up per-model AUC into an aggregate (e.g. per
`Configuration`) with `cdh_utils.weighted_average_polars("Performance",
"ResponseCount")` — a response-count-weighted average
(`Aggregates.model_summary`, `HealthCheck.qmd`, etc.).

Weighting by `ResponseCount = Positives + Negatives` implicitly treats AUC
like a proportion (e.g. a success rate), where precision scales with
total trials. AUC is not a proportion — its sampling variance depends on
*both* class counts multiplicatively, not their sum. A proposed
alternative is to weight by `Positives * Negatives` instead.

## Statistical grounding

1. **Combining independent estimates.** The classical result for pooling
   independent unbiased estimates (fixed-effect meta-analysis, Cochran
   1954, "The Combination of Estimates from Different Experiments",
   *Biometrics* 10(1)) is that the variance-minimizing weights are
   proportional to `1 / Var(estimate)` — inverse-variance weighting. Any
   other weighting scheme is a heuristic approximation to this.

2. **AUC variance formula.** Hanley & McNeil (1982, "The Meaning and Use
   of the Area under a ROC Curve", *Radiology* 143(1)) give the
   approximate variance of an AUC/Wilcoxon estimate as

   `Var(AUC) ≈ [AUC(1-AUC) + (n_pos - 1)(Q1 - AUC²) + (n_neg - 1)(Q2 - AUC²)] / (n_pos * n_neg)`

   where `Q1 = AUC / (2 - AUC)` and `Q2 = 2*AUC² / (1 + AUC)`. The
   denominator is `n_pos * n_neg`, not `n_pos + n_neg`. This is also the
   basis of the exact nonparametric variance estimator used by DeLong,
   DeLong & Clarke-Pearson (1988, "Comparing the Areas under Two or More
   Correlated Receiver Operating Characteristic Curves: A Nonparametric
   Approach", *Biometrics* 44(3), <https://doi.org/10.2307/2531595>),
   which `pdstools` already implements in
   `cdh_utils.auc_variance_delong_grouped` / `auc_ci_from_bincounts`.

3. **Implication.** Since `Var(AUC) ∝ 1 / (n_pos * n_neg)` to first order,
   weighting by `Positives * Negatives` is a much closer proxy to the
   variance-optimal inverse-variance weight than weighting by
   `Positives + Negatives`. The `ResponseCount`-weighted average is the
   right choice for pooling *proportions* (e.g. `SuccessRate`), but is a
   statistically weaker choice for pooling AUC.

4. **The rigorous option already exists in-repo.** `pdstools` already
   computes a DeLong-style AUC variance per model
   (`ADMDatamart.active_ranges` returns `AUC_ActiveRange` and
   `AUC_ActiveRange_CI_Variance` from classifier bin counts) and already
   has a function to combine independent AUC + variance estimates with
   inverse-variance weights: `cdh_utils.weighted_auc_ci_from_estimates`.
   So the "gold standard" comparison point — true inverse-variance
   weighting using DeLong variances — is a small amount of glue code away,
   not new statistical machinery.

Conclusion: there are five aggregation methods worth comparing head-to-head
on real customer data, in increasing order of statistical rigor:

| # | Method | Weight | Requires |
|---|--------|--------|----------|
| a | Naive average | 1 (equal weight per model) | `Performance`, `Configuration` |
| b | Current (response-count-weighted) | `Positives + Negatives` | same |
| c | Proposed (pos*neg-weighted) | `Positives * Negatives` | same |
| d | Positives-only-weighted | `Positives` | same |
| e | Inverse-variance (DeLong) | `1 / Var_DeLong(AUC)` | predictor/classifier bin data |

(e) is the statistically correct pooling method; (c) and (d) are evaluated
as cheap proxies that don't require classifier bin data — (d) because
`Var(AUC) ~= (Q2 - AUC^2) / n_pos` once `Negatives >> Positives` (see
below), making plain `Positives` weighting a candidate that may track (e)
even more closely than (c) under heavy class imbalance; (a) and (b) are
the naive/current baselines being challenged.

Two supplementary diagnostic variants of (b) and (c) are also reported,
restricted to models with `Positives > 0`: `AUC_Weighted_ResponseCount_PositivesOnly`
and `AUC_Weighted_PosNeg_PositivesOnly`. Zero-positive models have an
essentially undefined/uninformative AUC but can carry a large `ResponseCount`,
so they can pull (b) away from (c)/(e) even though (c)/(e) already give them
zero weight. Comparing (b) against its positives-only variant isolates how
much of the gap to (c)/(e) is explained purely by that exclusion, versus by
the `Positives * Negatives` vs. `Positives + Negatives` weighting itself.

## Proposed approach

### Phase 1 — implement the five-way comparison in `batch_healthcheck.py`

Add a new analysis step to `process_dataset()`, alongside the existing CI
maturity analysis:

1. Build a per-model frame from the latest snapshot
   (`datamart.aggregates.last()`), with `ModelID`, `Configuration`,
   `Performance`, `Positives`, `ResponseCount` (`Negatives = ResponseCount
   - Positives`). Filter to `ResponseCount > 0`.
2. Join in `datamart.active_ranges()` output (`AUC_ActiveRange`,
   `AUC_ActiveRange_CI_Variance`, `AUC_ActiveRange_CI_Available`) for
   models that have classifier bin data. This reuses the same call
   already made for CI maturity — no duplicate computation of active
   ranges.
3. Group by `Configuration` (falling back to all-models-in-one-group if
   `Configuration` isn't present) and compute, per group:
   - `AUC_Naive_Mean` — `pl.col("Performance").mean()`
   - `AUC_Weighted_ResponseCount` — `cdh_utils.weighted_performance_polars()`
     (existing / current behavior)
   - `AUC_Weighted_PosNeg` — `weighted_average_polars("Performance",
     Positives * Negatives)`
   - `AUC_Weighted_Positives` — `weighted_average_polars("Performance",
     Positives)`
   - `AUC_Weighted_InverseVariance_DeLong` — via
     `cdh_utils.weighted_auc_ci_from_estimates(auc=AUC_ActiveRange,
     variance=AUC_ActiveRange_CI_Variance, weights=1/variance)`, restricted
     to models with `AUC_ActiveRange_CI_Available`. Also report the
     resulting CI bounds and `N_Models_With_DeLong_CI`.
   Also compute the two `_PositivesOnly` diagnostic variants of (b) and
   (c) described above, restricted to models with `Positives > 0`.
4. Collect one row per `(Dataset, Configuration)` into a list (same
   collector pattern as `ci_maturity_dataset_rows`), write to
   `auc_rollup_comparison.csv` next to the other batch summary outputs at
   the end of `main()`.
5. No changes to `python/pdstools` library code are anticipated — this is
   analysis/tooling in the batch script only. If the comparison proves the
   pos*neg or inverse-variance approach is superior, a follow-up PR
   changes the library default (`Aggregates.model_summary`,
   `HealthCheck.qmd`, etc.) — out of scope for this plan.

### Phase 2 — run across the private multi-customer dataset

Run `scripts/batch_healthcheck.py` over the full private customer corpus,
inspect `auc_rollup_comparison.csv`:

- How much do (b), (c), (e) disagree from each other and from (a),
  in absolute AUC points, across configurations?
- Does (c) (cheap proxy) track (e) (rigorous) closely? If yes, (c) is a
  good practical default even without classifier bin data.
- Are there configurations where (b) meaningfully diverges from (c)/(e) —
  e.g. because a low-volume-but-balanced-classes model would be
  under-weighted by (c)/(e) relative to (b), or vice versa for
  high-volume-but-skewed-classes models?

### Phase 3 — decide and (if warranted) change the library default

Out of scope for this plan; file a follow-up plan item if the data
supports changing `weighted_performance_polars`'s default weighting.

## Open questions / assumptions made

- Grouping level: `Configuration` (matches `Aggregates.model_summary`
  convention). Not grouping by `Channel`/`Direction` as well, to keep the
  comparison table small; can be revisited.
- Only the latest snapshot per model is used (consistent with
  `aggregates.last()` elsewhere in the codebase), not a time-weighted
  roll-up across snapshots.
- The DeLong/inverse-variance metric is computed on the *active-range*
  AUC (`AUC_ActiveRange`), which is what `active_ranges()` already
  provides variance for. This is what the existing CI maturity analysis
  uses too, so it's consistent within the script — but note it's not
  exactly the same point estimate as `Performance` (datamart-reported
  AUC from the full classifier range) used for (a)/(b)/(c). Both figures
  are reported so the disagreement is visible rather than hidden.

## Conclusions from the private customer-data run

Ran `scripts/batch_healthcheck.py` across the full private customer
corpus and analyzed `auc_rollup_comparison.csv` (320 configurations, 234
with a DeLong estimate pooling >= 5 models) with
`print_auc_rollup_agreement_table` / `generate_auc_rollup_bland_altman_plot`.
Agreement with the DeLong reference, on the 0-1 AUC scale:

| Method | Bias | MAE | RMSE | Pearson r | Lin's CCC |
|---|---|---|---|---|---|
| Naive mean | -0.078 | 0.091 | 0.122 | 0.61 | 0.40 |
| **Current**: ResponseCount-weighted | -0.027 | 0.052 | 0.088 | 0.71 | 0.65 |
| **Proposed**: PosNeg-weighted | -0.006 | 0.036 | 0.059 | 0.87 | 0.86 |

Plain Pearson r is misleading for this comparison (two methods can
correlate highly while being systematically offset), so Bias/MAE/RMSE and
Lin's CCC (which penalizes offset/scale mismatch as well as weak
correlation) are the metrics that matter. **Conclusion: `Positives *
Negatives` weighting roughly halves the error against the DeLong
reference compared to the current `ResponseCount` weighting** (MAE 0.036
vs. 0.052; CCC 0.86 vs. 0.65), and is a feasible drop-in replacement that
needs no classifier/predictor bin data.

### Caveat: the DeLong reference is currently NB-only, and pooling assumptions differ by technique

NB and AGB models are architecturally different in a way that matters
for this whole analysis: **for NB, each action/treatment is a completely
separate model**, fit independently — the "pool k independent AUC
estimates" assumption behind Cochran/DeLong inverse-variance weighting
holds naturally. **For AGB, there is typically one shared model per
`Configuration` (usually per channel), with issue/group/action splits
scored from that same shared model** — so the per-`ModelID` rows being
pooled are correlated segment-level scores of one model, not independent
models. Per-row DeLong variance (computed from each row's own classifier
bin counts) captures sampling noise in that segment's outcomes, but not
the shared model-fit uncertainty common across all of that model's
segments — so even where available, it likely understates true
uncertainty for AGB.

In practice it usually isn't even available: **every AGB/GradientBoost
configuration observed in the private customer-data run had
`N_Models_With_DeLong_CI = 0`** (`*_AGB` configs, `BoostingModel_*`,
etc.). Root cause: `ADMDatamart._minMaxScoresPerModel` (which
`active_ranges()` and therefore the DeLong CI machinery depends on)
computes a model's reachable score range as the sum of each active
predictor's log-odds min/max — the Naive Bayes score formula. AGB's
score isn't a sum of independent per-predictor log-odds contributions
(tree splits interact nonlinearly), so this computation structurally
cannot produce a meaningful active range for AGB models; `AUC_ActiveRange`
and its CI end up null.

**Consequence: the agreement table and Bland-Altman plot above are
implicitly NB-only.** AGB configurations contribute nothing to the
"eligible" (DeLong-available) comparison set — not because AGB agrees or
disagrees with the DeLong reference, but because DeLong currently can't
be computed for AGB at all with `active_ranges()`. Whether `PosNeg`
weighting is a good DeLong proxy for AGB models remains unvalidated.

This does **not** affect the `Positives > 200` maturity discussion below:
that reasoning is about the precision of a single AUC estimate from its
own bin counts, which applies the same way whether the row is a
standalone NB model or a correlated segment of a shared AGB model.

A genuine fix (an AGB-aware active-range/CI computation, or a different
variance model that accounts for shared model-fit uncertainty across
segments) is out of scope for this plan — noted as a follow-up.

Restricting either weighting to models with `Positives > 0`
(`*_PositivesOnly` columns) made no measurable difference in this data —
zero-positive models are not the main driver of the gap; the weighting
scheme itself (multiplicative vs. additive combination of class counts)
is.

### The two aggregates answer different questions — keep both

`AUC_Weighted_ResponseCount` has a clean operational interpretation:
weighting by `ResponseCount` (= number of decisions) means the result is
*the average AUC experienced by a randomly picked decision/interaction*.
That's the right number for "what discriminative power did our deployed
decisioning actually run at."

`AUC_Weighted_PosNeg` (and `AUC_Weighted_InverseVariance_DeLong`) answer a
different question: *what's the lowest-noise estimate of typical model
quality*, by upweighting statistically reliable models and downweighting
models whose AUC estimate is noisy (regardless of how many decisions they
served). A model with 10,000 decisions but only 3 positives contributes
almost nothing to this number, because its AUC estimate is unreliable —
not because it's operationally unimportant.

Recommendation: report both, explicitly labeled by purpose.
`ResponseCount`-weighted for business/operational reporting ("what AUC are
our decisions running at"); `PosNeg`/DeLong-weighted for health-check-style
diagnostics, trend monitoring, and cross-configuration comparison, where a
handful of immature models shouldn't swing the number.

### Class imbalance changes which weight is theoretically optimal

The customer data has a strong, consistent class imbalance: `Negatives`
is typically ~100x `Positives`. Splitting the Hanley-McNeil variance
formula's three terms by `n_pos * n_neg` and taking `n_neg -> infinity`
with `n_pos` fixed:

`Var(AUC) -> (Q2 - AUC^2) / n_pos`

i.e. **once negatives are abundant relative to positives, the variance is
governed almost entirely by `Positives` alone; extra negatives barely
reduce it further.** This means:

- If the pos:neg ratio is roughly similar across a customer's models,
  `Positives * Negatives ~ constant * Positives^2`, which weights more
  aggressively (quadratically) than the theoretically optimal weight
  (`1/Var ~ Positives`, i.e. linear). `PosNeg` still beats `ResponseCount`
  by a wide margin empirically (see table above), but a plain
  **`Positives`-weighted** aggregate is a candidate to test as an even
  closer, and simpler, DeLong approximation under heavy imbalance.
- For model **maturity**: this partially vindicates positives-count as
  the primary signal (matching the existing crude `Positives > 200`
  heuristic in `Analysis.health_check_maturity_criteria` /
  `Aggregates.py`'s `isValid` checks), but the threshold itself is
  arbitrary. A calibrated version would derive the required `Positives`
  from a target CI half-width `w` via
  `n_pos ~ (Q2 - AUC^2) / (w / 1.96)^2`, rather than a fixed 200 for
  every model regardless of its AUC or desired precision.

### Retrofit justification for the existing `Positives > 200` threshold

`Positives > 200` is used widely across Pega (`Analysis.health_check_
maturity_criteria`, `Aggregates.py`'s `isValid` checks, this batch
script's `positives_maturity_threshold`) and would be difficult to
change. Given the class-imbalance finding above, there are two
independent statistical arguments that retrofit-justify keeping it:

1. **It buys a roughly constant AUC confidence interval, regardless of
   model quality.** Under `Negatives >> Positives`, `Var(AUC) ~= (Q2 -
   AUC^2) / n_pos`. Evaluating this at `n_pos = 200` across the AUC range
   CDH models typically produce:

   | AUC | Var(AUC) | 95% CI half-width (pts) | Full CI width (pts) |
   |---|---|---|---|
   | 0.55 | 0.00044 | 4.1 | 8.2 |
   | 0.65 | 0.00045 | 4.2 | 8.3 |
   | 0.75 | 0.00040 | 3.9 | 7.9 |
   | 0.85 | 0.00029 | 3.4 | 6.7 |

   The precision this threshold buys (~+/-4 AUC points, ~8-point full
   95% CI width) is nearly independent of how good the model actually
   is — a useful property for a single portfolio-wide threshold. `200`
   can be presented as "the point count that buys ~+/-4 AUC points of
   certainty," not an arbitrary round number.

2. **It independently matches the classical "events per variable" rule
   for stable model fitting.** Peduzzi, Concato, Kemper, Holford &
   Feinstein (1996, "A simulation study of the number of events per
   variable in logistic regression analysis", *Journal of Clinical
   Epidemiology* 49(12)) found that ~10 events per predictor are needed
   for stable coefficient/log-odds estimates in a binary classifier. A
   typical NBAD Naive Bayes model with ~15-20 active predictors gives
   `10 x ~20 = ~200` — the same number, from an entirely different
   angle (parameter stability rather than AUC estimation precision).

Caveats to state alongside this, so it isn't oversold:

- The "flat across AUC" result specifically depends on `Negatives >>
  Positives` holding. With a much less extreme imbalance (e.g. 10:1),
  the negatives term stops being negligible and 200 positives would
  actually buy *more* precision than the table above — so `200` is a
  conservative, not overly lax, threshold under less extreme imbalance.
- It's still a single fixed threshold, not adaptive to a chosen
  confidence level or a specific model's AUC/predictor count — defensible
  as "a good round number with real statistical backing," not "the
  provably optimal choice" (see the calibrated-threshold follow-up
  above).

### Assessing a prior empirical rule of thumb: `CI ≈ 3.79 / sqrt(Positives)`

A previous internal analysis (also based on DeLong estimators) arrived at
a rule of thumb `AUC CI ≈ 3.79 * 1/sqrt(Positives)`. The *functional
form* — inverse-square-root of `Positives` — is exactly what this plan
derives independently from `Var(AUC) -> (Q2 - AUC^2) / n_pos` under heavy
imbalance: that's a strong cross-check that the earlier analysis picked
up the same imbalance-driven effect. The constant `3.79` can't be
verified from the number alone, though, since it depends on choices not
recorded alongside it: full- vs. half-width, 0-1 fraction vs. 50-100
points scale, confidence level, and the AUC value assumed for `Q2 -
AUC^2` (the true constant is `z * sqrt(Q2 - AUC^2)`, which is fairly
flat but not literally constant across the 0.55-0.85 AUC range — same
caveat as the `200` threshold above).

To make this testable rather than guessed at, `_generate_ci_width_plot`
(shared by the per-dataset and pooled cross-dataset CI plots in
`batch_healthcheck.py`) now also fits a version of the power law
constrained to the theoretical slope of `-0.5`, and prints the
resulting constant on both the 0-1 and Pega points scales. On the next
full customer-corpus run, check:

1. The free-fit `slope` printed alongside the cross-dataset CI plot —
   close to `-0.5` confirms the imbalance-driven story empirically.
2. The constrained `1/sqrt(Positives)` constant — compare it to `3.79`
   on whichever scale the original rule used. Validated against
   synthetic data generated with a known slope/constant, this fit
   recovers the ground truth exactly (slope -0.500, constant recovered
   to 3 decimal places, R²=1.00), so any mismatch on real data reflects
   the real customer data, not a fitting artifact.

## Follow-ups (not implemented in this plan)

- Empirically validate `AUC_Weighted_Positives` against the DeLong
  reference on a fresh full private-customer-corpus run (the Conclusions
  section above reflects a run from before this column existed). Expect
  it to track DeLong at least as well as `AUC_Weighted_PosNeg` given the
  observed ~100x class imbalance; update the Conclusions table once
  confirmed.
- Design a CI-width- or `Positives`-derived maturity metric to replace/
  complement the `Positives > 200` heuristic in
  `Analysis.health_check_maturity_criteria` and `Aggregates.py`. This
  touches library code (not just the batch script) and needs its own
  plan doc if pursued.
- If the customer-data conclusions hold up, consider changing the
  library default (`Aggregates.model_summary`, `HealthCheck.qmd`) from
  `ResponseCount`- to `PosNeg`-weighted AUC roll-up — as a deliberate,
  separately-reviewed change, not a silent side effect of this
  investigation.
- Build an AGB-aware active-range/CI computation, so the DeLong reference
  and its agreement validation can be extended to AGB configurations,
  not just NB. [PR #947](https://github.com/pegasystems/pega-datascientist-tools/pull/947)
  fixed duplicate-row inflation in the existing calculation, but
  `active_ranges()` still derives reachable scores by summing
  predictor-bin log odds, which is specific to Naive Bayes and does not
  represent an AGB tree ensemble. The grouped DeLong helper itself can
  operate on valid classifier-bin counts, but it does not account for
  shared model-fit uncertainty across an AGB model's issue/group/action
  segments. Enabling validated AGB confidence intervals requires a
  separate design and implementation.
