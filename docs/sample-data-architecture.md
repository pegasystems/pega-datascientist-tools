# Synthetic Sample Data Architecture

**Updated:** November 11, 2026 — Maintainer guide to the intent behind
`python/pdstools/sample_data/`. User-facing behaviour is described in
`python/docs/source/sample_data.rst`; read the source for signatures.

The generator deliberately plants **stories**: patterns that make the ADM
Health Check, Model Report, Prediction Studio, Interaction History and Impact
Analyzer outputs worth reading. Each story exists for a downstream plot or
narrative. Before tuning a constant, find its row below, keep the guarding
test meaningful and re-read the affected report text.

The code is the source of truth. The original v3.2 design document and JSON
specification are historical inputs and are not kept in the repository.

---

## Layers

```
spec (immutable catalog)  →  world (noise-free truth per channel-day)  →  views
_spec, _actions, _treatments,   _world                                   _prediction, _adm, _binning,
_predictors, _configurations,                                            _ih, _impact
_stories, _calendar, _random                                             → _demo.demo_data()
```

- Views never depend on each other; they each sample noise around the same
  world, which is why they tell consistent stories.
- Each entity draws from its own SHA-256-derived stream (`_random.py`), so
  adding catalog entries does not perturb existing data.
- The public entry points are `demo_data()` and the `from_mock_data()`
  constructors on `ADMDatamart`, `Prediction`, `IH` and `ImpactAnalyzer`.

## Story index

Test paths are relative to `python/tests/`.

| Story | Demonstrates | Implemented in | Guarded by |
|---|---|---|---|
| Email feed dip in weeks 17–18 with negative lift, then a three-week recovery | Prediction lift and performance trends, success-rate trend | `_stories.email_feed_dip`, `_world._channel_days` | `sample_data/test_sample_world.py::test_email_lift_is_negative_only_during_feed_dip`, `test_sample_spec.py::test_exact_story_timings_and_effects` |
| Positive-only weeks (Email NB week 3, Mobile NB week 8, Web AGB week 20), then a burst of late negatives | Success-rate anomalies, response-loop guidance | `_stories.*_positive_only`, `_world._NEGATIVE_BURST` | `test_sample_world.py::test_positive_only_days_have_recovery_burst`, `test_sample_adm.py::test_positive_only_weeks` |
| Mobile learns from a 1.2% to about a 2% success rate, and an AGB challenger joins mid-window | Model maturity, NB versus AGB comparison | `_world._mobile`, `_MOBILE_START_CTR`, `_MOBILE_CHALLENGER` | `test_sample_world.py::test_mobile_learns_and_challenger_helps`, `test_sample_adm.py::test_mobile_models_learn_from_zero`, `test_sample_ih.py::test_mobile_challenger_mixes_techniques` |
| Three shared predictors lead both Mobile configurations | NB/AGB predictor comparison | `_binning._MOBILE_SHARED`, `_MOBILE_SHARED_AGB_BOOST` | `test_sample_binning.py::test_mobile_shared_predictors_lead` |
| Web and Email weekend dips | Weekly seasonality in trends | `_world._WEEKEND_CTR`, `_prediction._WEEKEND_AUC` | `test_sample_world.py::test_weekend_dips_only_on_web_and_email` |
| Per-channel lift targets; correlated Test/Control arms so the 1% control still shows lift | Prediction overview Lift equals (Test − Control) / Control | `_world._TARGET_LIFT`, `_prediction.ARM_CORRELATION`, `MODEL_CONTROL_SHARE` | `test_sample_prediction.py::test_mock_data_lift_and_validity`, `test_lift_is_test_over_control_and_ignores_nba` |
| Call Center is tiny, has no NBA row and has deliberately invalid arms | Struck-through prediction row, excluded-channel callout | `_prediction._NO_NBA`, `_ih._CALL_CENTER_SHARE` | `test_sample_adm.py::test_call_center_is_not_a_valid_channel`, `test_sample_impact.py::test_call_center_arms_stay_sparse` |
| Specialized Web Loans AGB model: exclusive routing and a deny-by-default predictor allowlist | Specialized-model setup, protected-field policy | `_configurations` `Web_Loans_Click_Through_Rate_GB`, `_spec` validation | `test_sample_spec.py::test_configured_predictor_counts_and_loans_subset`, `test_exclusive_routing_and_context_lifetimes`, `test_reject_baseline_excluded_loans_predictor` |
| Actions launch and retire during the window; new actions get a decaying novelty boost | New/stopped actions plot, response concentration | `_actions` start/stop days, `_adm._NOVELTY_BOOST` | `test_sample_spec.py::test_exact_lifecycle_and_channel_memberships`, `test_sample_adm.py::test_stopped_models_keep_their_last_counts` (no dedicated novelty test) |
| 12 stuck, 12 weak and 4 overfitting models | Bubble chart outliers, model quality sections | `_adm.seeded_cases`, `_SEEDED_CASES` | `test_sample_adm.py::test_seeded_quality_cases` |
| Weaker Web Risk and Email Service contexts | A few orange cells in the performance treemap | `_adm._LOW_SIGNAL_PLATEAU` | `test_sample_adm.py::test_health_check_performance_treemap_has_a_few_orange_contexts` |
| Naming defects: zero-width space in a name, no-break space in a group, empty groups | Model Naming Review; whitespace variants collapse in the hierarchy | `_adm.adm_models` (`ZERO_WIDTH_SPACE`, `NO_BREAK_SPACE`) | `test_sample_adm.py::test_naming_review`, `test_naming_cases_use_expected_characters`, `reports/test_healthcheck_sample.py::test_action_hierarchy_collapses_whitespace_group_variants` |
| Sales and Service carry most volume; Retention and Risk carry less | Response counts spread across issues | Action-channel `volume_weight` in `_actions` | `test_sample_spec.py::test_action_volume_weights_by_channel_and_issue`, `test_sample_adm.py::test_health_check_response_counts_are_spread_across_issues` |
| Customer predictors interleave with IH and Param predictors in importance | Predictor importance and category plots | `_binning._FAMILY_BOOST`, `_AGB_CUSTOMER_IMPORTANCE_BOOST` | `test_sample_binning.py::test_customer_predictors_are_mixed_into_each_health_check_importance_plot` |
| `Param.CallToActionStyle` is 100% missing; `Param.CreativeFormat` is partially missing | Missing-predictor diagnostics | `_predictors` distributions | `test_sample_binning.py::test_fully_missing_parameter_is_inactive_and_reconciles`, `test_partially_missing_parameter` |
| `Customer.ChurnRiskModelScore` external score, never in Web Loans | "External Model" predictor category | `_predictors`, `_configurations` | `test_sample_binning.py::test_external_churn_score_*`, `test_sample_spec.py::test_external_churn_score_is_excluded_from_the_loans_policy` |
| Mature Mobile NB `SavingsReview` model with about 30 active predictors, including the week-8 positive-only marker | Default `ModelReport.qmd` render and the docs Model Report example | `_adm.default_report_model`, `_DEFAULT_REPORT_ACTIVE` | `test_sample_adm.py::test_default_report_model`, `test_default_model_report_uses_the_generator_default_model` |
| Conversion-Test converts better than Conversion-Control | Conversion Reporting notebook | `_ih._CONVERSION` | `test_sample_ih.py::test_conversions_favour_test_group` |
| Five Impact Analyzer experiments, about 1% each, sharing the NBA test arm; assigned per customer per calendar month | Impact Analyzer comparisons | `_ih.IA_GROUPS`, `IA_TARGET_FRACTION`, `_impact` | `test_sample_impact.py::test_reconciles_exactly_with_ih`, `test_sample_ih.py::test_impact_groups_are_shared_and_stable`, `test_shifted_impact_groups_follow_actual_calendar_months` |

## Invariants

- Shifting the end date by whole weeks changes only the timestamps in ADM and
  Prediction views (`test_sample_demo.py::test_adm_and_prediction_views_shift_only_their_timestamps`).
  IH and Impact Analyzer data may change, because experiment assignment follows calendar months.
- Impact Analyzer mock counts reconcile exactly with IH for matching arguments.
- Predictor bin totals reconcile with model snapshots, and stored Performance is
  the AUC of the emitted bins.
- Model IDs are stable UUIDs. The default Model Report hardcodes the
  generator's `default_report_model` ID in `ModelReport.qmd`, the docs example
  generator and the shell example; a test keeps them in sync.
- Windows are at least 56 days, end on a Sunday and preserve weekdays.
- Default `demo_data()` stays under 15 s and 1 GiB; prediction-only generation
  stays under 1 s.

## When changing a story

1. Update the code and the guarding test in the same change, using exact
   values where possible.
2. Update this table and, if users can see the change, `sample_data.rst`.
3. Re-render the default Health Check and check that the callouts and
   notebook narrative still match the plots.
