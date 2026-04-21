# Migration guide: pdstools v4 → v5

This guide lists every breaking change in v5 and the minimum edit needed
to get your code working again. **Items are added incrementally as v5
PRs land** — don't treat this file as final until v5 is tagged.

If you hit something missing from this guide, please file an issue.

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

### `Explanations.filter_*`

Previously `**filter_kwargs`. Now uses explicit parameters per
`docs/plans/explanations/typed-filter-kwargs.md`.

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

---

## Behaviour changes (no API change, may affect output)

(To be populated by v5 PRs as they land.)

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
