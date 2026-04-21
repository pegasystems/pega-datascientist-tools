# Changelog

All notable changes to pdstools are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(Pre-release notes accumulate here until the next tag.)

## [5.0.0] — TBD

Major release. **Breaking changes** — see
[`docs/migration-v4-to-v5.md`](docs/migration-v4-to-v5.md) for the upgrade
guide.

### Removed

- `ADMTrees(...)` polymorphic factory shim. Use `ADMTreesModel.from_file`,
  `ADMTreesModel.from_url`, `ADMTreesModel.from_dict`, or
  `MultiTrees.from_datamart` instead.
- `ADMTreesModel(file)` deprecated polymorphic constructor. Use the
  explicit `from_*` classmethods.
- `ADMTreesModel.parse_split_values` /
  `ADMTreesModel.parse_split_values_with_spaces` deprecated helpers.
  Use `ADMTreesModel.parse_split` instead.
- `pdstools.adm.ADMTrees` re-export module (file).  Import from
  `pdstools.adm.trees` directly.
- `pdstools.app.decision_analyzer.da_streamlit_utils.get_current_index`
  back-compat re-export.
- Private `_read_client_credential_file` removed from
  `pdstools.pega_io.__all__`. Still importable for now but no longer a
  documented entry point.

### Changed

- (To be populated by the v5 cleanup PRs.)
- Decision Analyzer plot helpers: renamed `propensityTH` / `priorityTH`
  keyword arguments to `propensity_th` / `priority_th` on
  `DecisionAnalyzer.get_offer_quality`,
  `pdstools.decision_analyzer.plots.offer_quality_piecharts`, and
  `pdstools.decision_analyzer.plots.offer_quality_single_pie`. See the
  migration guide for before/after examples.
- `pdstools.decision_analyzer.plots` is now a package (split into private
  topical submodules `_sensitivity`, `_winloss`, `_optionality`,
  `_funnel`, `_distribution`, `_components`, `_trend`, `_offer_quality`).
  The public surface (`Plot` class, free helper functions) is unchanged
  apart from the `propensityTH`/`priorityTH` rename above.
- Streamlit apps audited and aligned with `AGENTS.md` "Streamlit apps"
  rules: every page now calls `standard_page_config(...)` before any other
  Streamlit output (Decision Analyzer pages 2–10 + both Health Check
  sub-pages were missing it). Removed deprecated `use_container_width=True`
  / `width="stretch"` arguments (now Streamlit defaults). The single
  remaining `components.html` usage (Decision Analyzer "Single Decision"
  tree-grid) is now documented as the only place without a Streamlit-native
  alternative.
- `DecisionAnalyzer.__init__` keyword arguments are now keyword-only
  (`level`, `sample_size`, `mandatory_expr`, `additional_columns`,
  `num_samples`). Positional usage now raises `TypeError`.
- `DecisionAnalyzer.from_explainability_extract` /
  `from_decision_analyzer` now have explicit keyword-only parameters
  instead of `**kwargs`, matching `__init__`.
- `DecisionAnalyzer.get_available_fields_for_filtering` parameter
  renamed `categoricalOnly` → `categorical_only` (now keyword-only).
- `DecisionAnalyzer.cleanup_raw_data` made private:
  `_cleanup_raw_data`. It was always called internally from `__init__`.
### Added

- `pdstools.valuefinder.__init__` now re-exports `ValueFinder` (was empty).
- `CHANGELOG.md` (this file).
- `docs/migration-v4-to-v5.md`.

---

## [4.7.1] and earlier

See git history (`git log --oneline V4.7.1..HEAD`) for pre-CHANGELOG
release notes. Highlights from the 4.x line are summarised in the
migration guide where they matter for v5 users.
