# AGBExplained: new article notebook + 9 library plot methods

**Priority:** P1

**Files touched:**
- `python/pdstools/adm/trees/_model.py` — 9 new plot methods
- `python/pdstools/adm/trees/_nodes.py` — add `sampleCount` to `Node`
- `python/pdstools/utils/datasets.py` — swap `sample_trees()` to new sample file
- `data/agb/ModelExportWithSampleCount.json` — new anonymized 100-tree sample (produced from one-off script)
- `examples/articles/AGBExplained.ipynb` — create new article notebook
- `examples/adm/AGBModelVisualisation.ipynb` — **delete** (all content ported to AGBExplained)
- `python/tests/adm/test_trees.py` — tests for all 9 new methods

## Problem

`ADMTreesModel` has no library methods for analysing training dynamics,
feature importance by gain, or predictor role (depth vs coverage).  The
only AGB tutorial notebook (`AGBModelVisualisation.ipynb`) is a demo, not
a mathematical explainer, and it duplicates patterns better served by a
companion to `examples/articles/ADMExplained.ipynb`.

The current toy sample (`_974a7f9c-…txt` via `from_url`) has no
`sampleCount` data on tree nodes — the replacement 100-tree export does,
enabling three streaming-timeline plot methods that reveal how the model
was trained over time.

## Proposed approach

### ✅ Phase 0 — Anonymized sample file (DONE)

Produce `data/agb/ModelExportWithSampleCount.json` from the richer
100-tree export using a **one-off anonymization script** (not checked in).

Action/name set values, client-specific `Customer.*` attribute names, and
proprietary score predictor names were anonymized to generic tokens
(`Action_01`, `Name_01`, `Customer.Attr01`, `Customer.Scores.ExtModelScore1`,
etc.).  Standard Pega namespaces (`IH.*`, `py*`) and top-level metadata
(AUC, training stats, model version) were kept as-is.
The one-off anonymization script was **not** checked in.

Update `datasets.sample_trees()` to load from `data/agb/ModelExportWithSampleCount.json`
via `ADMTreesModel.from_file`.  The old remote `.txt` URL is replaced.
The original un-anonymized export is **never checked in**.

### ✅ Phase 0.5 — Add `sampleCount` to `Node` (DONE)

In `python/pdstools/adm/trees/_nodes.py`:
- Add `sampleCount: int | None = None` to the `Node` dataclass
- Update `_iter_nodes` to pass `tree.get("sampleCount")` when yielding each node

Backward-compatible: the old sample and any user-supplied model without
`sampleCount` keys → all `Node.sampleCount` are `None`; no existing code
is affected.

### ⏳ Phase 1 — 9 new plot methods in `_model.py` (TODO)

All follow the `show: bool = True` / `return_df: bool = False` pattern
with `@overload` stubs (see `plot_contribution_per_tree` as the template).
Lazy-import plotly inside each method body; raise
`MissingDependenciesException(["plotly"], "AGB", deps_group="adm")`.
Place all 9 after `plot_contribution_per_tree` (~line 1200), before
private helpers.

**Methods 1–6 — no `sampleCount` required:**

1. **`plot_gain_per_tree(show=True, return_df=False)`** (Fig 2a)
   - Data: `tree_stats` — `gains.list.sum()` as `total_gain`
   - Chart: bar of `total_gain` per `treeID`; secondary y-axis line for root `score`
   - `return_df` columns: `treeID`, `total_gain`, `score`

2. **`plot_cumulative_gain_share(show=True, return_df=False)`** (Fig 2c)
   - Data: cumsum of per-tree total gain divided by grand total
   - Chart: line (S-curve); annotate 50 % crossover tree
   - `return_df` columns: `treeID`, `cumulative_gain_share`

3. **`plot_feature_importance_by_gain(top_n=15, show=True, return_df=False)`** (Fig 3)
   - Data: `gains_per_split.group_by("predictor").agg(pl.sum("gains"))`, `head(top_n)`
   - Namespace: first dot-segment of predictor name
   - Chart: horizontal bar, color by namespace
   - `return_df` columns: `predictor`, `total_gain`, `namespace`

4. **`plot_early_vs_late_gain(show=True, return_df=False)`** (Fig 3b)
   - Data: fresh tree walk — per predictor sum gains in first-quartile trees vs last-quartile trees
   - Chart: scatter, log-log axes, x=early_gain, y=late_gain, size=total_gain, color=namespace; y=x diagonal reference line
   - `return_df` columns: `predictor`, `early_gain`, `late_gain`, `total_gain`, `namespace`

5. **`plot_gain_by_namespace(show=True, return_df=False)`** (Fig 6)
   - Data: `gains_per_split` with namespace from first dot-segment; `group_by` namespace, sum, normalise
   - Chart: horizontal bar sorted descending
   - `return_df` columns: `namespace`, `total_gain`, `gain_share`

6. **`plot_feature_role_map(show=True, return_df=False)`** (Fig 4)
   - Data: fresh tree walk via `_iter_nodes` — per predictor: `mean_depth`, `tree_coverage` (distinct trees used), `total_gain`, `namespace`
   - Chart: bubble scatter — x=mean_depth, y=tree_coverage, size=total_gain, color=namespace; hover shows predictor name
   - `return_df` columns: `predictor`, `mean_depth`, `tree_coverage`, `total_gain`, `namespace`

**Methods 7–9 — require `sampleCount`:**

Methods 7–9 must raise a clear `ValueError("sampleCount data not available …")` when
all `Node.sampleCount` values are `None` (i.e. the model was loaded from a
file without this field).

7. **`plot_training_stream_timeline(show=True, return_df=False)`** (Fig 2d)
   - Data: root `sampleCount` per tree (= cumulative responses seen when tree was built)
   - Chart: line, x=treeID, y=root_sample_count
   - `return_df` columns: `treeID`, `root_sample_count`

8. **`plot_inter_tree_gaps(show=True, return_df=False)`** (Fig 2e)
   - Data: `root_sampleCount[i] − root_sampleCount[i−1]` (responses between consecutive tree additions)
   - Chart: bar showing training cadence / activity bursts
   - `return_df` columns: `treeID`, `sample_gap`

9. **`plot_gain_decay_dual_lens(show=True, return_df=False)`** (Fig 2b)
   - Data: per tree — `total_gain`, `treeID`, `node_age = totalCount − root_sampleCount`
   - Chart: dual x-axis line: left axis = treeID, right axis = node_age; y = total_gain
   - `return_df` columns: `treeID`, `node_age`, `total_gain`

**Namespace extraction** (methods 3–6, 8, 9): first dot-segment of the
predictor name, e.g. `IH.Mobile.Inbound.Clicked.pxLastOutcomeTime.DaysSince`
→ `IH`.  Predictors with no dot (e.g. `pyTreatment`) → use the full name.

### ⏳ Phase 2 — Update `datasets.sample_trees()` (TODO)

Change `sample_trees()` in `python/pdstools/utils/datasets.py` from `from_url(remote_url)` to
`ADMTreesModel.from_file(_DATA_DIR / "ModelExportWithSampleCount.json")`.
Keep the `warnings.catch_warnings` + `RuntimeError` wrapper pattern.

### ⏳ Phase 3 — `examples/articles/AGBExplained.ipynb` (TODO)

Style: same as `ADMExplained.ipynb` — markdown-heavy, LaTeX formulas,
code verifies the math, `great_tables.GT` tables, cells **not pre-run**.
All cells use `datasets.sample_trees()` as the single data source.

**Sections:**

1. **The Building Block: A Single Decision Tree**
   - Node anatomy: `split`, `gain`, `score`, left/right children
   - Show raw dict for tree 0; explain JSON fields
   - `Trees.predictors` — name→type mapping (one cell, confirms symbolic vs numeric)
   - Gain formula (information gain / log-loss reduction)
   - `Trees.plot_tree(0)` — visualise

2. **The Ensemble: How Trees Build on Each Other**
   - Gradient boosting intuition; additive formula $\hat{y} = \sum_k f_k(x)$
   - Gain decay: early trees capture main signal, later trees refine
   - `Trees.tree_stats` — GT table
   - `Trees.plot_gain_per_tree()` — NEW
   - `Trees.plot_cumulative_gain_share()` — NEW
   - Sub-section: *Training stream* (uses `sampleCount` data)
     - `Trees.plot_training_stream_timeline()` — NEW
     - `Trees.plot_inter_tree_gaps()` — NEW
     - `Trees.plot_gain_decay_dual_lens()` — NEW

3. **Scoring a Customer**
   - Define a concrete `x` dict from predictor values in the sample model
   - `Trees.plot_tree(0, highlighted=x)` — traversal path
   - `Trees.get_all_visited_nodes(x)` — leaf-score table
   - Raw score sum + sigmoid derivation $p = 1/(1+e^{-s})$
   - `Trees.score(x)` — confirm match
   - `Trees.plot_contribution_per_tree(x)` — running propensity

4. **Feature Importance: Which Predictors Drive the Model?**
   - Gain as importance metric (XGBoost-style gain formula)
   - `Trees.grouped_gains_per_split` — summary table
   - `Trees.plot_splits_per_variable(subset=[…])` — existing box-plot
   - `Trees.plot_feature_importance_by_gain()` — NEW
   - `Trees.plot_gain_by_namespace()` — NEW
   - Namespace explanation: `IH.*` = interaction history, `Customer.*` = attributes, `py*` = policy/context
   - `Trees.plot_early_vs_late_gain()` — NEW; "early learner vs late refiner"
   - `Trees.plot_feature_role_map()` — NEW; "router vs refiner" (shallow+wide vs deep+narrow)
   - Inline cell (not a library method): gain by split depth

5. **Model Health at a Glance**
   - `Trees.metrics` — GT table grouped by category
   - Key signals explained: `score_decay_ratio`, `top_predictor_gain_share`, `predictor_gain_entropy`, `mean_gain_first/last_half`, `number_of_stump_trees`

### ⏳ Phase 4 — Delete `AGBModelVisualisation.ipynb` (TODO)

`git rm examples/adm/AGBModelVisualisation.ipynb`

All content from the old notebook is covered in `AGBExplained.ipynb`:

| Old notebook section | Where in AGBExplained |
|---|---|
| `Trees.metrics` / `metric_descriptions()` | Section 5 |
| `Trees.model[k]` raw dict / node anatomy | Section 1 |
| `Trees.predictors` | Section 1 |
| `Trees.tree_stats` | Section 2 |
| `Trees.splits_per_tree[k]` / `gains_per_tree[k]` | Subsumed by `tree_stats` GT table (Section 2) |
| `Trees.grouped_gains_per_split` | Section 4 |
| `Trees.plot_splits_per_variable(subset=[…])` | Section 4 |
| `Trees.plot_tree(k)` | Section 1 |
| `Trees.plot_tree(k, highlighted=x)` | Section 3 |
| `Trees.get_all_visited_nodes(x)` | Section 3 |
| Sigmoid derivation + `Trees.score(x)` | Section 3 |
| `Trees.plot_contribution_per_tree(x)` | Section 3 |
| `sampleX` random helper | Replaced by concrete `x` dict (better for didactics) |
| PNG/PDF export note | Deliberately omitted — low value one-liner |

## Verification

```bash
uv run pytest python/tests/adm/ -k "trees or agb" -q
uv run ruff check python/pdstools/adm/trees/
uv run ruff format --check python/pdstools/adm/trees/
```

For each of the 9 new methods verify:
- `return_df=True` → `pl.DataFrame` with documented columns
- `show=False` → `plotly.graph_objects.Figure`
- Methods 7–9: `ValueError` when `sampleCount` absent (loaded from old-style file)

## Scope

- **In**: 9 library methods, 1 new article notebook, 1 redirect, 1 anonymized sample, `Node.sampleCount`
- **Out**: Fig 5 (parent→child interaction graph; needs graph-layout library, low-confidence interpretation), Fig 4b gain-by-depth as a library method (3-line inline cell in the notebook suffices)
- `return_df` added only to the 9 new methods; existing methods not refactored
- Anonymization script is one-off / local only — never checked in
