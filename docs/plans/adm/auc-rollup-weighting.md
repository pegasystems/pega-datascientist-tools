# AUC roll-up weighting: statistical comparison of aggregation methods

Review various weighting techniques for roll up of AUC for both NB and AGB.
No production-library default changes are in scope; the batch run provides
diagnostic comparisons and the assumptions behind them.

## Problem

`pdstools` currently rolls up per-model AUC into an aggregate (e.g. per
`Configuration`) with `cdh_utils.weighted_average_polars("Performance",
"ResponseCount")` — a response-count-weighted average
(`Aggregates.model_summary`, `HealthCheck.qmd`, etc.).

Weighting by `ResponseCount = Positives + Negatives` treats AUC like a
proportion (e.g. a success rate), where precision is expected to scale with
total trials. There is a more direct reason to consider the product. For
model `i`, let `C_i` be its concordance score (counting tied pairs as
one-half), and let `n_pos_i` and `n_neg_i` be its class counts. Conventional
empirical AUC is `AUC_i = C_i / (n_pos_i * n_neg_i)`. Therefore:

`sum(C_i) / sum(n_pos_i * n_neg_i)`
`= sum((n_pos_i * n_neg_i) * AUC_i) / sum(n_pos_i * n_neg_i)`.

Thus `Positives * Negatives` weighting is exactly the ratio of total
concordant positive-negative pairs to total eligible positive-negative pairs:
the AUC of the pooled set of within-model pairs. This is a particularly
transparent estimand, not merely a heuristic sample-size weight. It is still
not universally variance-optimal; that depends on the target, independence,
and the class-count-dependent AUC variance.

## Statistical grounding

1. **Combining independent estimates.** For independent unbiased estimates
   of one common quantity (the fixed-effect meta-analysis setting, Cochran
   1954, "The Combination of Estimates from Different Experiments",
   *Biometrics* 10(1)), the variance-minimizing weights are proportional to
   `1 / Var(estimate)` — inverse-variance weighting. If the estimates have
   different underlying AUCs, the same calculation defines a
   precision-weighted blend, not an estimate of one shared AUC. Correlation
   between estimates also changes the optimal weighting and uncertainty.

2. **AUC variance formula.** Hanley & McNeil (1982, "The Meaning and Use
   of the Area under a ROC Curve", *Radiology* 143(1)) give the
   approximate variance of an AUC/Wilcoxon estimate as

   `Var(AUC) ≈ [AUC(1-AUC) + (n_pos - 1)(Q1 - AUC²) + (n_neg - 1)(Q2 - AUC²)] / (n_pos * n_neg)`

   where `Q1 = AUC / (2 - AUC)` and `Q2 = 2*AUC² / (1 + AUC)`. The
   denominator is `n_pos * n_neg`, but the numerator also grows with the
   class counts. After expansion, the leading terms are
   `(Q2 - AUC^2) / n_pos` and `(Q1 - AUC^2) / n_neg`; the residual
   `1 / (n_pos * n_neg)` term is not generally dominant. `pdstools`
   implements a grouped-bin DeLong-style variance estimate based on
   DeLong & Clarke-Pearson (1988, "Comparing the Areas under Two or More
   Correlated Receiver Operating Characteristic Curves: A Nonparametric
   Approach", *Biometrics* 44(3), <https://doi.org/10.2307/2531595>).
   Because the inputs are classifier-bin counts, it is an uncertainty
   estimate for the binned active-range AUC, not a perfect substitute for
   row-level scores.

3. **Pair-count interpretation.** For conventional empirical AUC in a common
   score direction, the identity above makes `Positives * Negatives` weighting
   the exact pair-pooled AUC estimand. It gives each eligible positive-negative
   pair equal influence, rather than giving each model equal influence or
   weighting each decision equally. `ResponseCount` remains a valid
   operational estimand when the desired question is the decision-volume-
   weighted average of model AUCs; it is not simply an incorrect AUC aggregate.

4. **Variance implication.** `Positives * Negatives` can be a closer proxy to
   inverse-variance weighting than `Positives + Negatives`, especially when
   both class counts vary materially. Under strong class imbalance, however,
   the `1 / n_pos` term dominates and plain `Positives` weighting may be
   closer. The pair-pooled estimand and the fixed-effect inverse-variance
   estimand are therefore interpretable but different targets.

5. **A conditional reference already exists in-repo.** `pdstools` computes
   a grouped-bin DeLong-style variance from classifier counts
   (`ADMDatamart.active_ranges`) and combines AUC plus variance estimates
   with inverse-variance weights through
   `cdh_utils.weighted_auc_ci_from_estimates`. This gives a useful
   conditional fixed-effect benchmark, not a ground-truth AUC for every
   configuration. The reference uses the active-range CI estimate, whereas
   the other methods aggregate the datamart's full-range `Performance`
   value; that scope difference is part of the observed disagreement.

Conclusion: there are five aggregation methods worth comparing head-to-head
on the available data, representing different weighting assumptions and data
requirements:

| # | Method | Weight | Requires |
|---|--------|--------|----------|
| a | Naive average | 1 (equal weight per model) | `Performance`, `Configuration` |
| b | Current (response-count-weighted) | `Positives + Negatives` | same |
| c | Pair-pooled (pos*neg-weighted) | `Positives * Negatives` | same |
| d | Positives-only-weighted | `Positives` | same |
| e | Conditional inverse-variance (DeLong-style) | `1 / Var_DeLong(AUC)` | predictor/classifier bin data |

(e) is the minimum-variance fixed-effect combination when the estimates are
independent and share one target AUC. It is used here as a conditional
benchmark, not as a universally correct pooling method. Method (c) is the
exact pair-pooled estimand under the stated conventional-AUC assumptions;
method (d) is a positive-count-weighted alternative that does not require
classifier-bin data. Method (d) may track (e) more closely than (c) when
`Negatives >> Positives` because `Var(AUC) ~= (Q2 - AUC^2) / n_pos` (see
below). Numerical agreement with (e) does not make the estimands
interchangeable.

Two supplementary diagnostic variants of (b) and (c) are also reported,
restricted to models with `Positives > 0`: `AUC_Weighted_ResponseCount_PositivesOnly`
and `AUC_Weighted_PosNeg_PositivesOnly`. Zero-positive models have an
essentially undefined/uninformative AUC but can carry a large `ResponseCount`,
so they can pull (b) away from (c)/(e) even though (c)/(e) already give them
zero weight. Comparing (b) against its positives-only variant isolates how
much of the gap to (c)/(e) is explained purely by that exclusion, versus by
the `Positives * Negatives` vs. `Positives + Negatives` weighting itself.

## Proposed approach

### Phase 1 (done) — implement the five-way comparison in `batch_healthcheck.py`

Add a new analysis step to `process_dataset()`, alongside the existing CI
maturity analysis:

1. Build a per-model frame from the latest snapshot
   (`datamart.aggregates.last()`), with `ModelID`, `Configuration`,
   `Performance`, `Positives`, `ResponseCount` (`Negatives = ResponseCount
   - Positives`). Filter to `ResponseCount > 0`.
2. Join in `datamart.active_ranges()` output (`AUC_ActiveRange`,
   `AUC_ActiveRange_CI_Variance`, `AUC_ActiveRange_CI_Available`) for
   models that have classifier bin data. The call is batched across model
   IDs, avoiding the much more expensive per-model loop used by the original
   implementation.
3. Group by `Configuration` (falling back to all-models-in-one-group if
   `Configuration` isn't present) and compute, per group:
   - `AUC_Naive_Mean` — `pl.col("Performance").mean()`
   - `AUC_Weighted_ResponseCount` — `cdh_utils.weighted_performance_polars()`
     (existing / current behavior)
   - `AUC_Weighted_PosNeg` — `weighted_average_polars("Performance",
     Positives * Negatives)`, the pair-pooled AUC estimand when
     `Performance` is conventional empirical AUC.
   - `AUC_Weighted_Positives` — `weighted_average_polars("Performance",
     Positives)`
   - `AUC_Weighted_InverseVariance_DeLong` — via
     `cdh_utils.weighted_auc_ci_from_estimates(auc=AUC_ActiveRange,
     variance=AUC_ActiveRange_CI_Variance, weights=1/variance)`, restricted
     to rows with `AUC_ActiveRange_CI_Available`, taking a configuration-
     scoped AGB interval only once. Also report the resulting CI bounds and
     `N_DeLong_Estimates`.
   Also compute the two `_PositivesOnly` diagnostic variants of (b) and
   (c) described above, restricted to models with `Positives > 0`.
4. Collect one row per `(Dataset, Configuration)` into a list (same
   collector pattern as `ci_maturity_dataset_rows`), write to
   `auc_rollup_comparison.csv` next to the other batch summary outputs at
   the end of `main()`.
5. No production-library behavior changes are anticipated — this is
   analysis/tooling in the batch script only. If the comparison proves the
   pos*neg or inverse-variance approach is superior, a follow-up PR
   changes the library default (`Aggregates.model_summary`,
   `HealthCheck.qmd`, etc.) — out of scope for this plan.

### Phase 2 (done) — run across the available multi-dataset corpus

Run `scripts/batch_healthcheck.py` over the available corpus,
inspect `auc_rollup_comparison.csv`:

- How much do (b), (c), (e) disagree from each other and from (a),
  in absolute AUC points, across configurations?
- Does (c) track (e) closely? If yes, (c) may be a practical alternative
  even without classifier-bin data.
- Are there configurations where (b) meaningfully diverges from (c)/(e) —
  e.g. because a low-volume-but-balanced-classes model would be
  under-weighted by (c)/(e) relative to (b), or vice versa for
  high-volume-but-skewed-classes models?

See "Conclusions from the available-corpus run" below for the
results.

### Decision: keep the current `ResponseCount`-weighted default for now

`PosNeg` (and plain `Positives`) weighting track the conditional
inverse-variance benchmark much more closely than the current
`ResponseCount` weighting in the available comparison (see Conclusions
below), and are feasible drop-in replacements. **We are not changing the
library default at this time** — `Aggregates.model_summary`, `HealthCheck.qmd`,
and `weighted_performance_polars` keep `ResponseCount` weighting. Reasons:

- The latest run now includes 17 AGB configurations, but that sample is
  small and the AGB interval still excludes shared model-fit uncertainty
  (see the AGB caveat below). The AGB-aware CI computation landed in
  [PR #948](https://github.com/pegasystems/pega-datascientist-tools/pull/948)
  and `compute_auc_rollup_comparison` uses its configuration-scoped estimate
  once.
- `ResponseCount`-weighted and `PosNeg`/`Positives`-weighted answer
  different questions (see "The two aggregates answer different
  questions" below) — changing the default isn't just a precision
  upgrade, it changes what the number means, which warrants a
  deliberate, separately-reviewed decision rather than a side effect of
  this investigation.

If a future change is warranted, it stays tracked as a follow-up below,
not decided here.

## Open questions / assumptions made

- Grouping level: `Configuration` (matches `Aggregates.model_summary`
  convention). Not grouping by `Channel`/`Direction` as well, to keep the
  comparison table small; can be revisited.
- Only the latest snapshot per model is used (consistent with
  `aggregates.last()` elsewhere in the codebase), not a time-weighted
  roll-up across snapshots.
- The DeLong/inverse-variance metric is computed on the *active-range*
  AUC (`AUC_ActiveRange_CI_Estimate`), which is what `active_ranges()`
  provides variance for. This is what the existing CI maturity analysis
  uses too, so it is consistent within the script — but it is not exactly
  the same point estimate as `Performance` (datamart-reported AUC from the
  full classifier range) used for (a)/(b)/(c). Both figures are reported so
  the disagreement is visible rather than hidden.
- The pair-count and positive-count identities below apply to conventional
  raw AUC in a common score direction. Pega's safe-AUC convention reflects
  values below 0.5, so these are interpretations of the current aggregates
  rather than algebraic identities for every possible input.

## Conclusions from the available-corpus run

Ran `scripts/batch_healthcheck.py` across the available corpus and analyzed
`auc_rollup_comparison.csv` (320 configurations, 234
with a conditional inverse-variance estimate combining >= 5 estimates) with
`print_auc_rollup_agreement_table` / `generate_auc_rollup_bland_altman_plot`.
Agreement with the conditional inverse-variance benchmark, on the 0-1 AUC scale:

| Method | Bias | MAE | RMSE | Pearson r | Lin's CCC |
|---|---|---|---|---|---|
| Naive mean | -0.078 | 0.091 | 0.122 | 0.61 | 0.40 |
| **Current**: ResponseCount-weighted | -0.027 | 0.052 | 0.088 | 0.71 | 0.65 |
| **Pair-pooled**: PosNeg-weighted | -0.006 | 0.036 | 0.059 | 0.87 | 0.86 |
| Positives-only-weighted | -0.014 | 0.033 | 0.057 | 0.89 | 0.86 |

Plain Pearson r is misleading for this comparison (two methods can
correlate highly while being systematically offset), so Bias/MAE/RMSE and
Lin's CCC (which penalizes offset/scale mismatch as well as weak
correlation) are the metrics that matter. **Conclusion: `Positives *
Negatives` weighting roughly halves the error against the conditional
benchmark compared to the current `ResponseCount` weighting** (MAE 0.036
vs. 0.052; CCC 0.86 vs. 0.65). In addition to this empirical agreement, it
has a clear interpretation: under conventional common-direction AUC, it is
the ratio of total concordant pairs to total positive-negative pairs. It is
therefore a feasible, interpretable pair-pooled alternative that needs no
classifier/predictor bin data. This is not evidence that `PosNeg` is the
universally correct aggregate for every operational or statistical target.

**Update from a later full-corpus run (with `AUC_Weighted_Positives`
added):** plain `Positives` weighting matches `PosNeg` weighting almost
exactly (MAE 0.033 vs. 0.036, CCC 0.86 vs. 0.86 — Positives-only is
marginally tighter on MAE/RMSE, `PosNeg` has a smaller Bias magnitude).
This confirms the imbalance-driven prediction above: since
`Negatives >> Positives` in this data, `Var(AUC) ~= (Q2 - AUC^2) / n_pos`
means the variance-optimal weight is governed almost entirely by
`Positives`, and multiplying by `Negatives` on top (as `PosNeg` does)
adds little further discriminative value between models. Either `PosNeg`
or plain `Positives` weighting is a solid, feasible-to-implement upgrade
over the current `ResponseCount` weighting. `PosNeg` has the more direct
pair-pooled interpretation; `Positives` is simpler and may approximate the
conditional inverse-variance weights more closely under stable extreme
imbalance.

**Update from the fresh full-corpus run following [PR #948](https://github.com/pegasystems/pega-datascientist-tools/pull/948)
(AGB-aware active ranges):** this run is the first where AGB
configurations contribute a DeLong estimate at all — `active_ranges()`
now reports a usable CI for 4,588 of 4,611 AGB/GradientBoost model rows
in the corpus (~99.5%), versus 0 previously. Of the 250
configurations with a DeLong estimate pooling >= 5 models (up from 234),
17 are AGB by name heuristic (`*AGB*`/`*Boosting*` in the configuration
name). Agreement with the conditional DeLong-style inverse-variance
benchmark, split out:

| Method | Scope | Bias | MAE | RMSE | Pearson r | Lin's CCC |
|---|---|---|---|---|---|---|
| Naive mean | All (n=250) | -0.082 | 0.094 | 0.125 | 0.62 | 0.40 |
| ResponseCount-weighted (current) | All (n=250) | -0.032 | 0.055 | 0.091 | 0.71 | 0.64 |
| **Pair-pooled**: PosNeg-weighted | All (n=250) | -0.009 | 0.036 | 0.060 | 0.87 | 0.86 |
| Positives-only-weighted | All (n=250) | -0.019 | 0.037 | 0.065 | 0.86 | 0.83 |
| ResponseCount-weighted (current) | NB-only (n=233) | -0.029 | 0.054 | 0.091 | 0.69 | 0.62 |
| **Pair-pooled**: PosNeg-weighted | NB-only (n=233) | -0.006 | 0.035 | 0.059 | 0.87 | 0.86 |
| ResponseCount-weighted (current) | AGB-only (n=17) | -0.071 | 0.071 | 0.087 | 0.84 | 0.64 |
| **Pair-pooled**: PosNeg-weighted | AGB-only (n=17) | -0.051 | 0.051 | 0.073 | 0.85 | 0.74 |

The NB-only figures are essentially unchanged from the earlier run
(same corpus, same 48,105-row NB CI-maturity dataset underneath). The
new information is the AGB-only row: `PosNeg` still beats
`ResponseCount` by a wide margin on AGB configurations too (CCC 0.74 vs.
0.64, smaller bias/MAE/RMSE throughout), consistent with the NB finding,
but the AGB sample is small (17 configurations) and its agreement with
the conditional benchmark is looser than NB's across every method (e.g. CCC
0.74 vs. 0.86 for `PosNeg`) — plausibly reflecting the model-fit
uncertainty caveat below, which the per-row DeLong variance still does
not capture even now that a CI is computable.

### Caveat: pooling assumptions differ by technique, and the AGB CI still doesn't capture shared model-fit uncertainty

NB and AGB models are architecturally different in a way that matters
for this analysis. **For NB, each action/treatment is typically a separate
model**, so the independent-estimate assumption is more plausible, although
overlapping evaluation data can still induce correlation. **For AGB, there
is typically one shared model per `Configuration`**, with issue/group/action
splits scored from that same model. The per-`ModelID` rows are therefore
correlated segment-level scores, not independent model fits.

In the earlier run analyzed above, this was compounded by a separate
issue: **every AGB/GradientBoost configuration then observed had
`N_Models_With_DeLong_CI = 0`**, because `ADMDatamart._minMaxScoresPerModel`
(which `active_ranges()` depended on) computed a model's reachable score
range as the sum of each active predictor's log-odds min/max — the
Naive Bayes score formula, which doesn't apply to AGB's tree ensemble.
[PR #948](https://github.com/pegasystems/pega-datascientist-tools/pull/948)
fixed this by deriving AGB active ranges from occupied classifier bins
instead, and pooling classifier-bin counts by `Configuration` so that
segments sharing one fitted ensemble get a single grouped DeLong interval.
The fresh full-corpus run above confirms the fix works at scale (~99.5%
of AGB rows now get a usable CI), which is what makes the AGB-only
agreement row above possible for the first time.

What PR #948 does **not** resolve is the shared model-fit uncertainty
issue: the pooled AGB interval (`AUC_ActiveRange_CI_Estimate`) quantifies
validation-sample uncertainty conditional on the exported fitted
ensemble, not model-fit uncertainty, which can't be identified from a
single datamart export (`AUC_ActiveRange_CI_IncludesModelFitUncertainty`
is explicitly `False`). So even with a computable CI, the AGB DeLong
reference likely still understates true uncertainty relative to NB's,
which may partly explain why every method's agreement with the DeLong
reference is looser on AGB than on NB in the table above.

This does **not** affect the `Positives > 200` maturity discussion below:
that reasoning is about the precision of a single AUC estimate from its
own bin counts, which applies the same way whether the row is a
standalone NB model or a correlated segment of a shared AGB model.

Restricting either weighting to models with `Positives > 0`
(`*_PositivesOnly` columns) made no measurable difference in this data —
zero-positive models are not the main driver of the gap; the weighting
scheme itself (multiplicative vs. additive combination of class counts)
is.

### The two aggregates answer different questions

`AUC_Weighted_ResponseCount` has a clean operational interpretation:
weighting by `ResponseCount` (= number of decisions) means the result is
the decision-volume-weighted average of the model AUCs. That's the right
number for "what discriminative power did our deployed decisioning actually
run at"; it should not be confused with a row-level AUC recomputed after
pooling all observations.

`AUC_Weighted_PosNeg` answers a different question: what is the AUC when all
eligible within-model positive-negative pairs are pooled, so each pair has
equal influence? A model contributes in proportion to the number of pairs it
provides. This is a natural statistical estimand, but it can differ from the
decision-volume-weighted operational quantity. The conditional
inverse-variance benchmark answers yet another question: how should
independent estimates of one common AUC be combined when their uncertainty is
known? A model with 10,000 decisions but only 3 positives contributes few
positive-negative pairs and little inverse-variance information — not because
it is operationally unimportant.

Observation: the two numbers serve different purposes and neither
supersedes the other. `ResponseCount`-weighted maps to business/operational
reporting ("what AUC are our decisions running at"); `PosNeg`/DeLong-weighted
maps to health-check-style diagnostics, trend monitoring, and
cross-configuration comparison, where a handful of immature models
shouldn't swing the number. Whether/how to surface both in practice is
left for a future, separately-scoped decision.

### Class imbalance changes which proxy is useful

The observed data has a strong, consistent class imbalance: `Negatives`
is typically ~100x `Positives`. Splitting the Hanley-McNeil variance
formula's three terms by `n_pos * n_neg` and taking `n_neg -> infinity`
with `n_pos` fixed:

`Var(AUC) -> (Q2 - AUC^2) / n_pos`

i.e. **once negatives are abundant relative to positives, the variance is
governed almost entirely by `Positives` alone; extra negatives barely
reduce it further.** This means:

- If the pos:neg ratio is roughly similar across a deployment's models,
  `Positives * Negatives ~ constant * Positives^2`, which weights more
  aggressively (quadratically) than the fixed-effect approximation to the
  inverse-variance weight (`1/Var ~ Positives`, i.e. linear, when AUC is
  roughly comparable across models). `PosNeg` still beats `ResponseCount`
  by a wide margin empirically (see table above), while retaining the
  especially clear pair-pooled interpretation. A plain
  **`Positives`-weighted** aggregate is a candidate to test as an even
  closer, and simpler, DeLong-style approximation under heavy imbalance.
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

### Simple CI-vs-Positives formula

Empirically, on this corpus (`ci_maturity_model_level.csv`, 48,105
NaiveBayes model rows with a usable CI):

`CI_Width ≈ 2.52 / sqrt(Positives)` (0-1 AUC scale), i.e.
`CI_Width ≈ 252 / sqrt(Positives)` on Pega's 50-100 points scale.

## Suggested follow-up experiments

- **Common-target simulation:** generate independent model estimates with one
  known AUC and vary positive/negative counts. Compare mean squared error and
  confidence-interval coverage for inverse-variance, `PosNeg`, `Positives`,
  and `ResponseCount` weights.
- **Heterogeneous-target simulation:** vary the true AUC by model and score
  each method against its intended estimand: equal-model mean,
  response-weighted mean, pair-pooled AUC, and precision-weighted mean. This
  separates numerical agreement from correctness for a stated target.
- **Correlated-segment simulation:** add a shared model-fit component to AGB
  segment estimates, then compare naive inverse-variance intervals with
  configuration-level pooling and cluster/bootstrap intervals. Measure
  coverage and effective sample size.
- **Scope-sensitivity analysis:** quantify the gap between full-range
  `Performance`, segment-level active-range AUC, and configuration-scoped
  active-range estimates before attributing differences to weighting.

## Follow-ups (not implemented in this plan)

- Design a CI-width- or `Positives`-derived maturity metric to replace/
  complement the `Positives > 200` heuristic in
  `Analysis.health_check_maturity_criteria` and `Aggregates.py`. This
  touches library code (not just the batch script) and needs its own
  plan doc if pursued.
- Revisit the library default (`Aggregates.model_summary`,
  `HealthCheck.qmd`) if a future need arises — see "Decision: keep the
  current `ResponseCount`-weighted default for now" above. Not changing
  it as part of this investigation.
- **Resolved:** an AGB-aware active-range/CI computation was needed so
  the conditional inverse-variance benchmark and its agreement validation could extend to AGB
  configurations, not just NB. [PR #947](https://github.com/pegasystems/pega-datascientist-tools/pull/947)
  fixed duplicate-row inflation in the existing calculation, but
  `active_ranges()` still derived reachable scores by summing
  predictor-bin log odds, which is specific to Naive Bayes and does not
  represent an AGB tree ensemble. The grouped DeLong helper itself could
  operate on valid classifier-bin counts, but it did not account for
  shared model-fit uncertainty across an AGB model's issue/group/action
  segments.

  [PR #948](https://github.com/pegasystems/pega-datascientist-tools/pull/948)
  (merged) addressed this properly: it derives AGB active ranges from
  occupied classifier bins instead of the NB log-odds reconstruction,
  and pools classifier-bin counts by `Configuration` so segments sharing
  one fitted ensemble get a single grouped DeLong interval instead of
  independent-looking per-segment ones. It adds
  `AUC_ActiveRange_CI_Estimate` (the pooled interval's center),
  `AUC_ActiveRange_CI_Scope`, and
  `AUC_ActiveRange_CI_IncludesModelFitUncertainty` (explicitly `False` —
  the interval quantifies validation-sample uncertainty conditional on
  the exported fitted ensemble, not model-fit uncertainty, which can't be
  identified from a single datamart export), while `AUC_ActiveRange`
  keeps returning the segment-level AUC for diagnostics.

  `compute_auc_rollup_comparison` was updated to match: it now detects
  already-pooled AGB rows via `AUC_ActiveRange_CI_Scope ==
  "configuration"` and uses `AUC_ActiveRange_CI_Estimate`/its variance
  directly per `Configuration`, instead of re-pooling the same
  configuration-level variance once per segment row (which would have
  double-counted the same evidence and understated the resulting
  uncertainty). The CSV now reports the number of distinct
  `N_DeLong_Estimates`, rather than counting repeated AGB segment rows.
  Verified against a synthetic AGB configuration built from the sample
  dataset.

  **Resolved:** a fresh full-corpus run was performed after PR #948
  landed. AGB configurations are now represented in the empirical
  Bland-Altman/agreement results (see the AGB-only row in the updated
  agreement table above) — the pos*neg/positives weighting advantage
  over `ResponseCount` observed for NB also holds for AGB in this run,
  though the AGB sample is small (17 configurations) and its overall
  agreement with the conditional benchmark is looser than NB's, consistent
  with the still-unresolved shared model-fit-uncertainty caveat.
