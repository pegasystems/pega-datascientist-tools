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

### Added

- Python 3.14 support (#416). CI now runs against 3.10–3.14.
- `ADMDatamart.from_s3` is now implemented (was a no-op stub). Downloads
  model + predictor data from S3 to a temporary directory and delegates
  to `from_ds_export`. `boto3` is gated via `MissingDependenciesException`
  and lives in the existing `pega_io` extras group (#702).
- `ADMDatamart.plot.over_time` (and aggregates equivalent) now accepts a
  list of column names for `by`, in addition to the existing single-string
  form — enables grouping by Channel × Direction etc. in Health Check /
  Model Reports (#700).
- `ImpactAnalyzer.from_excel` classmethod reads PDC Excel exports
  (`.xlsx`) using polars' built-in calamine engine (no extra deps). The
  Streamlit app and `--data-path` CLI flag accept `.xlsx` uploads with
  auto-detection (#685).
- `pega_io.read_data` now recognises `.xlsx` / `.xls` and owns the
  `fastexcel` optional-dependency shim. `ImpactAnalyzer.from_excel`
  delegates the file load to `read_data` (#694).
- Decision Analyzer auto-detects mandatory actions (priority ≥ 5M) when
  no `mandatory_expr` is passed, and flags them visually in the global
  win/loss distribution pie and on the Global Sensitivity page (#698).
- Health Check `--full-embed` / `--no-full-embed` CLI flag, propagated to
  the Streamlit Reports page as a new "Embed JS/CSS for offline viewing"
  toggle in the Advanced expander. Defaults preserve the previous
  behaviour (#688).
- `pdstools.valuefinder.__init__` now re-exports `ValueFinder` (was empty).
- `CHANGELOG.md` (this file).
- `docs/migration-v4-to-v5.md`.
- AppTest-based smoke tests for all three Streamlit apps under
  `python/tests/streamlit_apps/` — covers Decision Analyzer (Home + 10
  sub-pages), Health Check (Home + 3 sub-pages) and Impact Analyzer
  (Home + 3 sub-pages). Uses seeded fixtures so pages render the
  populated branch, not the upload-prompt branch.
- Unit tests for `cdh_utils.get_latest_pdstools_version` covering the
  happy path, network failure and malformed-response paths.

### Changed

- `DecisionAnalyzer` reorganised into a namespace facade (matching the
  `ADMDatamart` pattern): 16 aggregation methods moved to
  `da.aggregates.*` and 12 scoring / ranking / lever methods moved to
  `da.scoring.*`. Data-loading classmethods, properties and lifecycle
  helpers stay on the top-level class. See
  [`docs/migration-v4-to-v5.md`](docs/migration-v4-to-v5.md#namespace-reorganisation)
  for the complete before/after table.
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
  apart from the `propensityTH` / `priorityTH` rename above.
- Decision Analyzer app routes through programmatic `st.navigation()`
  instead of Streamlit's auto-discovered `pages/` directory. Pages that
  require loaded data (Overview, Action Distribution, Action Funnel,
  Global Sensitivity, Win/Loss, Optionality, Offer Quality, Thresholding,
  Arbitration Distribution, Single Decision) are hidden from the sidebar
  until data is uploaded — the Home page lists them as "Upload data above
  to unlock these analysis pages" so users still know what's available.
  Hidden pages remain reachable by URL; existing `ensure_data()` guards
  apply unchanged. `standard_page_config()` is now idempotent so per-page
  `st.set_page_config` calls don't conflict with the entry-script call.
- Decision Analyzer Streamlit pages: converted bare
  `from da_streamlit_utils import ...` imports to absolute
  `from pdstools.app.decision_analyzer.da_streamlit_utils import ...`,
  matching the Health Check and Impact Analyzer apps. This removes the
  dependency on `streamlit run`'s `sys.path` mutation and unblocks
  testing DA pages with `streamlit.testing.v1.AppTest` (#724).
- Streamlit apps audited and aligned with `AGENTS.md` "Streamlit apps"
  rules: every page now calls `standard_page_config(...)` before any other
  Streamlit output (Decision Analyzer pages 2–10 + both Health Check
  sub-pages were missing it). Removed deprecated `use_container_width=True`
  / `width="stretch"` arguments (now Streamlit defaults). The single
  remaining `components.html` usage (Decision Analyzer "Single Decision"
  tree-grid) is now documented as the only place without a Streamlit-native
  alternative.
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
- Internal split of
  `pdstools.infinity.resources.prediction_studio.v24_2.prediction` and
  `prediction_studio` into multi-module packages. Public imports unchanged
  (#732, #735 also splits `champion_challenger`).
- `Infinity(...)` / `AsyncInfinity(...)` now require keyword-only
  `base_url` and `auth` (plus optional `application_name`, `verify`,
  `pega_version`, `timeout`). The catch-all `*args, **kwargs` signature
  is gone; unknown kwargs are rejected. Direct `Infinity()` with no args
  now raises `TypeError` instead of `RuntimeError`. The
  `from_basic_auth` / `from_client_credentials` /
  `from_client_id_and_secret` classmethods are unchanged (#714).
- Explanations: `sort_by`, `display_by`, `descending`, `missing`,
  `remaining`, `include_numeric_single_bin` are now explicit keyword-only
  parameters with literal defaults across `Plots.contributions`,
  `Plots.plot_contributions_for_overall`,
  `Plots.plot_contributions_by_context`,
  `Aggregate.get_predictor_contributions`,
  `Aggregate.get_predictor_value_contributions`, and `Reports.generate`.
  New `SortBy` / `DisplayBy` `Literal` type aliases (#714).
- Explanations now validates raw input parquet schemas up front in
  `Preprocess.generate` and raises a descriptive `ValueError` for
  malformed / partially-written files instead of surfacing as a cryptic
  DuckDB SQL error mid-aggregation. New `pdstools.explanations.Schema`
  module (#741).
- `pdstools.adm.Reports` shared options (`title`, `subtitle`,
  `disclaimer`, `output_dir`, `output_type`, `qmd_file`, `full_embed`,
  `keep_temp_files`) are now passed via `**options: Unpack[ReportOptions]`
  (PEP 692 `TypedDict`). Existing keyword call sites
  (`dm.generate.health_check(title=..., output_type='pdf')`) continue to
  work unchanged and now get IDE autocomplete + static type checking.
  Unknown option keys raise `TypeError` (#653).
- `verbose=` parameters that only gated internal noise (subprocess
  output, schema warnings, decoded values, raw `os.listdir`) have been
  dropped on `cdh_utils._apply_schema_types`,
  `report_utils.{get_quarto_with_version, get_pandoc_with_version, run_quarto}`,
  `Reports.{model_reports, health_check}`,
  `pega_io.File.{read_ds_export, read_zipped_file, get_latest_file}`,
  `ADMTrees.{get_agb_models, __new__, get_multi_trees}` and the inner
  `_decode_trees` helpers. Such output now goes through
  `logger.debug` / `logger.info`. The `verbose=` parameter is preserved
  on call sites that drive genuine user-facing progress (S3,
  `read_multi_zip`, `Anonymization`). `Reports.health_check` no longer
  couples `verbose` to cache / error-reporting paths (#647).
- `BinAggregator` constructor parameter renamed `dm=` → `datamart=`,
  matching every other sub-namespace on `ADMDatamart`. No deprecated
  alias — the class is always constructed via
  `ADMDatamart.bin_aggregator` (#675).
- `pdstools.infinity.internal._base_client._infer_version` now logs
  warnings via the module logger (with `exc_info`) instead of `print()`
  (#736).
- Optional-dependency handling tightened: `MissingDependenciesException`
  is now raised consistently across pdstools instead of bare
  `ImportError`s, and the Quarto `standalone` flag is now wired through
  to report generation (#639, closes #436).
- Reorganised `python/tests/` into per-module subfolders mirroring
  `python/pdstools/` (`adm/`, `decision_analyzer/`, `impact_analyzer/`,
  `infinity/`, `pega_io/`, `ih/`, `valuefinder/`, `healthcheck/`,
  `explanations/`, `utils/`). Test paths in CI invocations have been
  updated; `--ignore=` flag values have changed (e.g.
  `python/tests/test_healthcheck.py` →
  `python/tests/healthcheck/test_healthcheck.py`).

### Removed

- `ADMTrees(...)` polymorphic factory shim. Use `ADMTreesModel.from_file`,
  `ADMTreesModel.from_url`, `ADMTreesModel.from_dict`, or
  `MultiTrees.from_datamart` instead.
- `ADMTreesModel(file)` deprecated polymorphic constructor. Use the
  explicit `from_*` classmethods.
- `ADMTreesModel.parse_split_values` /
  `ADMTreesModel.parse_split_values_with_spaces` deprecated helpers.
  Use `ADMTreesModel.parse_split` instead.
- `pdstools.adm.ADMTrees` re-export module (file). Import from
  `pdstools.adm.trees` directly.
- `pdstools.app.decision_analyzer.da_streamlit_utils.get_current_index`
  back-compat re-export.
- Private `_read_client_credential_file` removed from
  `pdstools.pega_io.__all__`. Still importable for now but no longer a
  documented entry point.
- `pdstools.pega_io.File.import_file` (public helper). It was always
  internal to `read_ds_export`; use `read_ds_export` instead.

### Fixed

- Better diagnostic information when Health Check report generation
  hits a `PermissionDenied` from Quarto on Windows; Infinity API
  client now supports Pega `'25.1`; Decision Analyzer no longer
  crashes with `ColumnNotFoundError: "Stage Group"` when the dataset
  only contains `Stage` — `__init__` now validates `level` against
  `available_levels` and falls back when needed (#642, closes #439,
  #409, #599).
- `pega_io.File.find_latest_files` no longer raises a silent
  `re.PatternError` on every call (`\d.{0,15}*GMT` had a stray
  repeat). Surfaced while narrowing `except Exception:` blocks across
  the codebase to specific exception types, which also restores
  meaningful error reporting for previously-swallowed bugs (#650).
- Health Check `report_utils` now inlines CSS in CDN-mode HTML output,
  fixing reports that rendered unstyled when opened without network
  access (#679).
- Health Check hides trend plots when only one snapshot is present
  (instead of rendering a misleading single-point line) (#682).
- Decision Analyzer `get_thresholding_data` now works with
  `num_samples > 1` (was incorrectly locked to a single sample) (#683).
- Decision Analyzer Streamlit app: more robust file uploads — META-INF
  entries from zipped exports are filtered out, partial state is
  cleared on upload errors, and error messages are clearer (#687).
- ADM `plot.over_time` and other category-coloured charts now use a
  global category → colour map so legends stay consistent across
  plots within a single report (#689).
- `ADMTreesModel` decoder dispatch converted from chained `try/except`
  to explicit `if/elif`, and its "warn once" path is now thread-safe
  (#690).
- Decision Analyzer `prio_factor_boxplots` no longer mutates
  `self.sample_size` as a side effect of plotting (#691).
- Decision Analyzer column schema now distinguishes Model Propensity
  vs Final Propensity display names so plots that reference one don't
  silently pick up the other (#699).

### Performance

- Lazy-import `plotly` across `adm`, `explanations`, `valuefinder`,
  `ih`, `impactanalyzer` and the Decision Analyzer Overview page so
  importing pdstools (or rendering a cold page) no longer pays the
  plotly import cost up front (#676, #686, #692).
- `BinAggregator.accumulate_num_binnings` is now batched, removing a
  per-binning Python-loop hot spot (#747).
- `IH.get_sequences` rewritten for fewer scans / collects (#745).
- `utils.overlap_matrix` and `overlap_lists_polars` vectorised over
  polars expressions, replacing per-row Python work (#746).
- Per-row `map_elements` lookups across the codebase replaced with
  `pl.Expr.replace_strict` (#652).

---

## [4.7.1] and earlier

See git history (`git log --oneline V4.7.1..HEAD`) for pre-CHANGELOG
release notes. Highlights from the 4.x line are summarised in the
migration guide where they matter for v5 users.
