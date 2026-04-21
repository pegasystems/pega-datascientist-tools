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

- `DecisionAnalyzer` reorganised into a namespace facade (matching the
  `ADMDatamart` pattern): 16 aggregation methods moved to
  `da.aggregates.*` and 12 scoring / ranking / lever methods moved to
  `da.scoring.*`. Data-loading classmethods, properties and lifecycle
  helpers stay on the top-level class. See
  [`docs/migration-v4-to-v5.md`](docs/migration-v4-to-v5.md#namespace-reorganisation)
  for the complete before/after table.
- (To be populated by the v5 cleanup PRs.)
- Decision Analyzer Streamlit pages: converted bare
  `from da_streamlit_utils import ...` imports to absolute
  `from pdstools.app.decision_analyzer.da_streamlit_utils import ...`,
  matching the Health Check and Impact Analyzer apps. This removes the
  dependency on `streamlit run`'s `sys.path` mutation and unblocks
  testing DA pages with `streamlit.testing.v1.AppTest` (#724).
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
- `Reports.excel_report` `predictor_binning` is now keyword-only. Any
  `print()` calls inside the Reports class have been replaced with
  `logging` — redirect the `pdstools.adm.Reports` logger if you relied
  on stdout output.
- `S3Data` renamed for snake_case compliance: constructor param
  `bucketName` → `bucket_name`; methods `getS3Files` → `get_files`,
  `getDatamartData` → `get_datamart_data`, `get_ADMDatamart` →
  `get_adm_datamart`. The `datamart_folder` parameter is now
  keyword-only on both methods. `DATAMART_TABLE_PREFIXES` is now a
  module-level dict for user extension.
- `pdstools.pega_io.read_ds_export` no longer accepts arbitrary
  `**reading_opts`. Only `infer_schema_length`, `separator`, and
  `ignore_errors` are supported, all keyword-only.
- `pdstools.pega_io.read_multi_zip` parameters `add_original_file_name`
  and `verbose` are now keyword-only. Dead `zip_type` parameter removed
  (only gzip was ever implemented).
- `pdstools.pega_io.read_dataflow_output` dead `extension` and
  `compression` parameters removed (only the defaults were ever
  implemented). `ADMDatamart.from_dataflow_export` and
  `ValueFinder.from_dataflow_export` drop the matching pass-through
  kwargs.
- `pdstools.pega_io.get_latest_file` now raises `ValueError` when
  passed an unknown `target` instead of returning the string sentinel
  `"Target not found"`. Returning `None` still signals "no matching
  file present in the directory".
- `Anonymization` constructor: default `temporary_path` is now `None`
  (lazily creates a fresh `tempfile.mkdtemp` directory on first use)
  instead of the hard-coded `/tmp/anonymisation`. No directory is
  created at construction time. Parameter
  `skip_columns_with_prefix` accepts a tuple too.
- `Anonymization.min_max` parameter `range` renamed to `value_range`
  (shadowed the builtin).

### Removed

- `pdstools.pega_io.File.import_file` (public helper). It was always
  internal to `read_ds_export`; use `read_ds_export` instead.
### Added

- `pdstools.valuefinder.__init__` now re-exports `ValueFinder` (was empty).
- `CHANGELOG.md` (this file).
- `docs/migration-v4-to-v5.md`.
- AppTest-based smoke tests for all three Streamlit apps under
  `python/tests/streamlit_apps/` — covers Decision Analyzer (Home + 10
  sub-pages), Health Check (Home + 3 sub-pages) and Impact Analyzer
  (Home + 3 sub-pages). Uses seeded fixtures so pages render the
  populated branch, not the upload-prompt branch.
- Unit tests for `cdh_utils.get_latest_pdstools_version` covering
  happy path, network failure and malformed-response paths.

---

## [4.7.1] and earlier

See git history (`git log --oneline V4.7.1..HEAD`) for pre-CHANGELOG
release notes. Highlights from the 4.x line are summarised in the
migration guide where they matter for v5 users.
