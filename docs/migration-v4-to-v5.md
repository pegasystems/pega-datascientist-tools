# Migration guide: pdstools v4 → v5

v5 is the first release after a major cleanup pass. The changes are
mostly mechanical — deprecated APIs are dropped, types are tightened,
fat modules are split, and large analyzer classes are reorganised into
namespace facades. For most users the upgrade is a handful of import
path or keyword updates.

This guide lists every breaking change in v5 and the minimum edit needed
to get your code working again. If you hit something missing from this
guide, please file an issue.

---

## Removed APIs

### `ADMTrees(...)` factory function

The polymorphic factory was deprecated in v4.x and is removed in v5.

```python
# Before (v4.x):
from pdstools.adm.ADMTrees import ADMTrees
model = ADMTrees("model.json")
multi  = ADMTrees(datamart_df)

# After (v5):
from pdstools.adm.trees import ADMTreesModel, MultiTrees
model = ADMTreesModel.from_file("model.json")
multi = MultiTrees.from_datamart(datamart_df)
```

### `ADMTreesModel(file=...)` constructor

Same story — explicit constructors only.

```python
# Before:
ADMTreesModel("model.json")
ADMTreesModel.from_dict(my_dict)   # also supported

# After (v5):
ADMTreesModel.from_file("model.json")
ADMTreesModel.from_dict(my_dict)
```

### `parse_split_values` / `parse_split_values_with_spaces`

Replaced by the typed module-level `parse_split` helper, which returns a
`Split` instance with `variable`, `operator`, and `value` fields.

```python
# Before:
model.parse_split_values("Color < 5")
model.parse_split_values_with_spaces("Color < 5")

# After (v5):
from pdstools.adm.trees import parse_split
parse_split("Color < 5")  # → Split(variable="Color", operator="<", value=5.0)
```

### `pdstools.adm.ADMTrees` module

The single-file re-export module was removed. The package layout is now:

```python
# Before:
from pdstools.adm.ADMTrees import ADMTreesModel, MultiTrees

# After (v5):
from pdstools.adm.trees import ADMTreesModel, MultiTrees
```

### Streamlit-internal re-exports

`get_current_index` is no longer re-exported from
`pdstools.app.decision_analyzer.da_streamlit_utils` for the legacy
import path. Import it directly from its defining module.

### Private API name removals from `__all__`

`pdstools.pega_io._read_client_credential_file` is no longer in
`pega_io.__all__`. The function still exists (leading underscore = private)
but is no longer a documented entry point.

---

## Subpackage layout fixes

### `pdstools.valuefinder`

Now exports `ValueFinder`. Previously you had to import from the deeper
path:

```python
# Before:
from pdstools.valuefinder.ValueFinder import ValueFinder

# After (v5, both still work; first one is preferred):
from pdstools.valuefinder import ValueFinder
from pdstools import ValueFinder            # also works (top-level export)
```

### `pdstools.decision_analyzer`

Stops re-exporting internal modules. The following now require explicit
imports:

```python
# Before:
from pdstools.decision_analyzer import plots, utils, da_streamlit_utils

# After (v5):
from pdstools.decision_analyzer import plots               # ← still works (in __all__)
from pdstools.decision_analyzer import utils               # ← still works
from pdstools.app.decision_analyzer import da_streamlit_utils
```

(The `da_streamlit_utils` re-export was Streamlit-only and only available
when `streamlit` was already imported — fragile, removed.)

---

## Tightened keyword arguments

### `ADMTreesModel.from_file`, `from_url`, `from_dict`, `from_datamart_blob`

Previously accepted `**kwargs` that were silently dropped (only
`context_keys=` was actually consumed). Now all parameters are explicit:

```python
# Before — extra kwargs were no-ops:
ADMTreesModel.from_file("model.json", whatever=42)

# After (v5) — only documented parameters are accepted:
ADMTreesModel.from_file("model.json", context_keys=["Issue", "Group"])
```

### `ADMTreesModel.__init__`

The constructor is now pure: it accepts already-parsed `trees` and
`model` data only. All file/URL/blob loading must go through the
`from_*` classmethods (see above). Tests and advanced callers that
already hold a parsed booster list can still call `ADMTreesModel(trees=...,
model=...)` directly.

### `Infinity(...)` / `AsyncInfinity(...)`

Previously `def __init__(self, *args, **kwargs)`. Now uses explicit
parameters matching the documented constructor signature.

### Explanations: explicit filter parameters

The `**filter_kwargs` catch-all on the Explanations `Plots`,
`Aggregate`, and `Reports` public methods has been replaced with
explicit, keyword-only parameters: `sort_by`, `display_by`,
`descending`, `missing`, `remaining`, `include_numeric_single_bin`.
Affected methods: `Plots.contributions`,
`Plots.plot_contributions_for_overall`,
`Plots.plot_contributions_by_context`,
`Aggregate.get_predictor_contributions`,
`Aggregate.get_predictor_value_contributions`, and `Reports.generate`.
Unknown kwargs now raise `TypeError`. New `SortBy` / `DisplayBy`
`Literal` aliases are exported alongside.

```python
# Before (v4.x):
plots.contributions(**{"sort_by": "value", "typo_arg": True})  # silently dropped

# After (v5):
plots.contributions(sort_by="value")  # unknown kwargs raise TypeError
```

### `DecisionAnalyzer.__init__` and `from_*` classmethods

Previously accepted positional arguments and `**kwargs`. Now all
configuration is keyword-only and the `from_*` classmethods spell out
each parameter explicitly (no more silent `**kwargs` drop-through).

```python
# Before (v4.x):
da = DecisionAnalyzer(raw_data, "Stage Group", 50_000)
da = DecisionAnalyzer.from_decision_analyzer(
    "data.parquet", sample_size=10_000, foo="ignored-silently"
)

# After (v5):
da = DecisionAnalyzer(raw_data, level="Stage Group", sample_size=50_000)
da = DecisionAnalyzer.from_decision_analyzer(
    "data.parquet", sample_size=10_000  # unknown kwargs now raise TypeError
)
```

### `DecisionAnalyzer.get_available_fields_for_filtering`

Parameter renamed and made keyword-only.

```python
# Before:
da.get_available_fields_for_filtering(categoricalOnly=True)

# After (v5):
da.get_available_fields_for_filtering(categorical_only=True)
```

### `DecisionAnalyzer.cleanup_raw_data` is now private

It was only ever called from `__init__` — no public callers in the
ecosystem. Renamed to `_cleanup_raw_data`. If you were calling it
externally, build a new `DecisionAnalyzer` instead.

### Decision Analyzer: `propensityTH` / `priorityTH` → `propensity_th` / `priority_th`

The `propensityTH` and `priorityTH` keyword arguments have been renamed
to snake_case across the Decision Analyzer API. Affected callables:
`DecisionAnalyzer.get_offer_quality(...)`,
`pdstools.decision_analyzer.plots.offer_quality_piecharts(...)`, and
`pdstools.decision_analyzer.plots.offer_quality_single_pie(...)`.

```python
# Before:
da.get_offer_quality(propensityTH=0.5, priorityTH=50)
offer_quality_piecharts(df, propensityTH=0.5, AvailableNBADStages=stages)

# After (v5):
da.get_offer_quality(propensity_th=0.5, priority_th=50)
offer_quality_piecharts(df, propensity_th=0.5, AvailableNBADStages=stages)
```

### `Reports.excel_report(predictor_binning=...)` is keyword-only

```python
# Before:
dm.generate.excel_report("Tables.xlsx", True)

# After (v5):
dm.generate.excel_report("Tables.xlsx", predictor_binning=True)
```

`Reports` methods also no longer `print()` status / warning messages;
they log through `logging.getLogger("pdstools.adm.Reports")`. Configure
that logger (or the root logger at `INFO`/`WARNING`) if you relied on
the stdout output — e.g. the "Data exported to …" message and the
Excel size / row-limit warnings.

### `S3Data` — snake_case method and parameter names

```python
# Before (v4.x):
from pdstools.pega_io import S3Data
s3 = S3Data(bucketName="my-bucket", temp_dir="./s3")
files = await s3.getS3Files(prefix="data/")
await s3.getDatamartData("modelSnapshot", "datamart", True)
dm = await s3.get_ADMDatamart("datamart", True)

# After (v5):
from pdstools.pega_io import S3Data
s3 = S3Data(bucket_name="my-bucket", temp_dir="./s3")
files = await s3.get_files(prefix="data/")
await s3.get_datamart_data("modelSnapshot", datamart_folder="datamart", verbose=True)
dm = await s3.get_adm_datamart(datamart_folder="datamart", verbose=True)
```

`datamart_folder` and `verbose` are now keyword-only.

### `read_ds_export` — explicit reading options

```python
# Before (v4.x):
read_ds_export("export.csv", infer_schema_length=200000, sep=";")

# After (v5):
read_ds_export("export.csv", infer_schema_length=200000, separator=";")
```

The function no longer accepts arbitrary `**reading_opts`. The
supported, keyword-only options are `infer_schema_length`,
`separator`, and `ignore_errors`. The legacy alias `sep=` is gone —
use `separator=`.

### `read_multi_zip` — dead `zip_type` parameter removed

```python
# Before (v4.x):
read_multi_zip(files, zip_type="gzip", add_original_file_name=True, verbose=True)

# After (v5):
read_multi_zip(files, add_original_file_name=True, verbose=True)
```

Only gzip was ever implemented. The remaining `add_original_file_name`
and `verbose` flags are keyword-only.

### `read_dataflow_output` / `from_dataflow_export` — dead params removed

```python
# Before (v4.x):
read_dataflow_output(files, "cache", extension="json", compression="gzip")
ADMDatamart.from_dataflow_export(m, p, extension="json", compression="gzip")
ValueFinder.from_dataflow_export(f, extension="json", compression="gzip")

# After (v5):
read_dataflow_output(files, "cache")
ADMDatamart.from_dataflow_export(m, p)
ValueFinder.from_dataflow_export(f)
```

Only the defaults were ever accepted (`Literal["json"]` /
`Literal["gzip"]`), so no caller was passing anything meaningful.

### `get_latest_file` — raises on unknown target

```python
# Before (v4.x):
get_latest_file(path, target="typo")   # returned the string "Target not found"

# After (v5):
get_latest_file(path, target="typo")   # raises ValueError
```

Valid targets remain `"model_data"`, `"predictor_data"`,
`"prediction_data"`, `"value_finder"`. Returning `None` still means
"target is valid but no matching file exists in the directory".

### `pdstools.pega_io.File.import_file` removed

It was always an internal helper used by `read_ds_export` for
Pega-specific schema handling. Use `read_ds_export` directly; it
covers every code path the old `import_file` did.

### `Anonymization` — no more hard-coded `/tmp`

```python
# Before (v4.x):
Anonymization(path_to_files="*.json")
# → silently wrote chunks to /tmp/anonymisation, created in __init__

# After (v5):
Anonymization(path_to_files="*.json")
# → lazily creates a fresh tempfile.mkdtemp() dir when you call .anonymize()

Anonymization(path_to_files="*.json", temporary_path="/my/cache")
# → still works, directory is created on first use (not in __init__)
```

`Anonymization.__init__` is now pure — no filesystem side effects.
`Anonymization.min_max` parameter `range` was renamed to
`value_range` (it was shadowing the Python builtin).

---

## Namespace reorganisation

### `DecisionAnalyzer` — methods moved to sub-namespaces

`DecisionAnalyzer` grew past 40 public methods and was refactored into a
thin orchestrator with two sub-namespaces, matching the "Namespace facade
for large analyzer classes" pattern already used by `ADMDatamart` (see
`AGENTS.md`). Data-loading classmethods, properties and core state stay
on the top-level class; analysis methods moved to `da.aggregates` and
`da.scoring`.

**Mechanical upgrade**: prefix the method call with the new namespace.
The signatures and return shapes are unchanged.

#### Aggregation methods — now under `da.aggregates`

| v4 call                                            | v5 call                                                       |
| -------------------------------------------------- | ------------------------------------------------------------- |
| `da.aggregate_remaining_per_stage(...)`            | `da.aggregates.aggregate_remaining_per_stage(...)`            |
| `da.get_distribution_data(...)`                    | `da.aggregates.get_distribution_data(...)`                    |
| `da.get_funnel_data(...)`                          | `da.aggregates.get_funnel_data(...)`                          |
| `da.get_decisions_without_actions_data(...)`       | `da.aggregates.get_decisions_without_actions_data(...)`       |
| `da.get_funnel_summary(...)`                       | `da.aggregates.get_funnel_summary(...)`                       |
| `da.get_optionality_data(...)`                     | `da.aggregates.get_optionality_data(...)`                     |
| `da.get_optionality_funnel(...)`                   | `da.aggregates.get_optionality_funnel(...)`                   |
| `da.get_action_variation_data(...)`                | `da.aggregates.get_action_variation_data(...)`                |
| `da.get_offer_variability_stats(...)`              | `da.aggregates.get_offer_variability_stats(...)`              |
| `da.get_offer_quality(...)`                        | `da.aggregates.get_offer_quality(...)`                        |
| `da.filtered_action_counts(...)`                   | `da.aggregates.filtered_action_counts(...)`                   |
| `da.get_trend_data(...)`                           | `da.aggregates.get_trend_data(...)`                           |
| `da.get_filter_component_data(...)`                | `da.aggregates.get_filter_component_data(...)`                |
| `da.get_component_action_impact(...)`              | `da.aggregates.get_component_action_impact(...)`              |
| `da.get_component_drilldown(...)`                  | `da.aggregates.get_component_drilldown(...)`                  |
| `da.get_ab_test_results(...)`                      | `da.aggregates.get_ab_test_results(...)`                      |

#### Scoring / ranking / lever methods — now under `da.scoring`

| v4 call                                            | v5 call                                                       |
| -------------------------------------------------- | ------------------------------------------------------------- |
| `da.re_rank(...)`                                  | `da.scoring.re_rank(...)`                                     |
| `da.get_selected_group_rank_boundaries(...)`       | `da.scoring.get_selected_group_rank_boundaries(...)`          |
| `da.get_sensitivity(...)`                          | `da.scoring.get_sensitivity(...)`                             |
| `da.get_thresholding_data(...)`                    | `da.scoring.get_thresholding_data(...)`                       |
| `da.priority_component_distribution(...)`          | `da.scoring.priority_component_distribution(...)`             |
| `da.all_components_distribution(...)`              | `da.scoring.all_components_distribution(...)`                 |
| `da.get_win_loss_distribution_data(...)`           | `da.scoring.get_win_loss_distribution_data(...)`              |
| `da.get_winning_or_losing_interactions(...)`       | `da.scoring.get_winning_or_losing_interactions(...)`          |
| `da.get_win_loss_counts(...)`                      | `da.scoring.get_win_loss_counts(...)`                         |
| `da.get_win_loss_distributions(...)`               | `da.scoring.get_win_loss_distributions(...)`                  |
| `da.get_win_distribution_data(...)`                | `da.scoring.get_win_distribution_data(...)`                   |
| `da.find_lever_value(...)`                         | `da.scoring.find_lever_value(...)`                            |

Private helpers `_remaining_at_stage`, `_winning_from` and `_losing_to`
also moved to `da.scoring` — if you were calling them (outside pdstools),
prefix accordingly.

#### What stayed on `DecisionAnalyzer`

Data-loading (`from_explainability_extract`, `from_decision_analyzer`,
`__init__`), properties (`decision_data`, `sample`,
`preaggregated_filter_view`, `preaggregated_remaining_view`,
`arbitration_stage`, `mandatory_actions`, `color_mappings`,
`stages_from_arbitration_down`, `stages_with_propensity`,
`propensity_validation_warning`, `overview_stats`, `available_levels`,
`num_sample_interactions`, `stage_to_group_mapping`), and lifecycle /
filter helpers (`set_level`, `filtered`,
`get_available_fields_for_filtering`, `get_possible_scope_values`,
`get_possible_stage_values`). `da.plot` — already introduced in v4 —
is unchanged.

---

## Things that did NOT change

- `pdstools.ADMDatamart`, `pdstools.IH`, `pdstools.Prediction`,
  `pdstools.ImpactAnalyzer`, `pdstools.ValueFinder`, `pdstools.Infinity`,
  `pdstools.AsyncInfinity` — all still importable from the top level.
- All `from_<source>` classmethods on the top-level analysis classes.
- All public plot methods (`return_df` parameter remains opt-in).
- The Streamlit apps (`pdstools-app health-check`, etc.) — same UX.
- Polars version requirements — unchanged.
- Python version: same minimum (3.10+).
