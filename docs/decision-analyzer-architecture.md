# Decision Analyzer Architecture

**Updated:** July 8, 2026 — Guide for making changes to the Decision Analysis Tool.

For method signatures, column definitions, and page docstrings, read the source
directly. This document captures the non-obvious design decisions and patterns.

---

## How Data Flows

```
CLI (cli.py)
  --data-path, --sample, --temp-dir  →  PDSTOOLS_* env vars
                    ↓
Home.py
  Reads env vars or accepts file upload
  Optional pre-ingestion sampling (--sample flag)
  Creates DecisionAnalyzer instance → st.session_state.decision_data
                    ↓
DecisionAnalyzer (in session_state, shared across all pages)
  decision_data   ← full cleaned LazyFrame
  sample          ← 10k interaction subset (cached_property)
  preaggregated_* ← one-time aggregation views (cached_property)
  plot            ← Plot accessor wrapping plots.py functions
                    ↓
Analysis Pages (3–10)
  1. ensure_data() guard + CLI filter banner in sidebar
  2. Page-specific sidebar controls (stage, scope, sliders)
  3. contextual_filters() — divider + "Filters" + Channel/Direction
  4. Call DA methods with additional_filters from session state
  5. Render via st.plotly_chart()
```

**CLI indirection:** Flags become env vars because Streamlit spawns a subprocess;
the only way to pass config is through the environment. Helpers in
`streamlit_utils.py` (`get_data_path()`, `get_sample_limit()`, `get_filter_specs()`)
read them back.

---

## Two Data Formats (v1 vs v2)

Auto-detected by column presence (see `determine_extract_type()` in `utils.py`):

| | v1 ("explainability_extract") | v2 ("decision_analyzer") |
|---|---|---|
| Source | Explainability Extract | Action Analysis / EEV2 |
| Stages | Synthetic: Arbitration + Output only | Real pipeline stages from data |
| Detection | Missing `Stage_pyName` / `Stage_pyStageGroup` | Has both stage columns |
| Scope | Arbitration-level analysis only | Full pipeline analysis |

v1 derives synthetic stages from ranking: rank-1 actions get "Output", others get
"Arbitration". This is why some pages behave differently for v1 data.

---

## Key Design Decisions

### Interaction-Level Sampling

The `sample` cached property selects ~10k unique interactions using hash-based
deterministic sampling. All rows within selected interactions are kept (preserving
the action-per-interaction structure).

```python
# Compares hash directly against scaled UInt64 threshold.
# Do NOT use hash % small_N — it collapses to few buckets for some ID patterns.
hash_threshold = int((2**64 - 1) * sample_rate)
df.filter(pl.col("Interaction ID").hash() < hash_threshold)
```

**Gotcha:** With 10k interactions, the sample can still contain millions of rows
(e.g., 400+ actions per interaction in some datasets). Plot functions must handle
this gracefully.

### Four-Layer Caching

1. **`@cached_property`** on DecisionAnalyzer — invalidated by `set_level()` which
   deletes cached attributes. Add new cached properties to the invalidation list in
   `set_level()`.
2. **`@st.cache_data`** on Streamlit wrapper functions — uses `polars_lazyframe_hashing`
   for Polars types. Used in `da_streamlit_utils.py`.
3. **Manual cache dicts** — `_thresholding_cache`, `_sensitivity_cache` for
   parameterized results.
4. **Pre-aggregation** — `preaggregated_filter_view` and `preaggregated_remaining_view`
   collect the full dataset once, then re-wrap as lazy for downstream queries.

### Contextual Filters (Sidebar)

`contextual_filters()` in `da_streamlit_utils.py` renders a universal filter section
(divider + caption + Channel/Direction selector) at the bottom of every analysis
page's sidebar. Session state keys:

- `page_channel_filter` — UI display value ("Any" or "Channel/Direction")
- `page_channel_expr` — Polars filter expression or None

Pages collect the active expressions via `collect_page_filters()` (also in
`da_streamlit_utils.py`) and pass the resulting list to
`DecisionAnalyzer.filtered(filters)`. The library deliberately does **not**
read `st.session_state` itself — keeping that translation in the app layer
means the analyser stays usable from notebooks and scripts. To add a new
contextual filter (e.g. Issue, Group), add a selector to
`contextual_filters()`, store the expression in a `page_{name}_expr` key,
and append that key to the tuple inside `collect_page_filters()`.

### Ranking

Actions are ranked per interaction by: `is_mandatory` desc → `Priority` desc →
`Stage Order` desc → alphabetic tiebreaker (Issue, Group, Action). The `Rank`
column (1 = winner) is derived in `cleanup_raw_data()`.

### Stable Colors and Optionality Breakdowns

`DecisionAnalyzer.color_mappings` owns colors for all pages and notebook plots.
The ten actions present in the most distinct decisions in the full dataset
receive distinct colors (alphabetical tie-breaking); other actions remain
individually represented in gray. Page filters and sampling do not recolor
them. Stage and component colors follow pipeline/filter families; components
used across multiple families are gray.

Issue/Group coloring of optionality changes only the composition of each bar.
A decision with two actions from one group and one from another stays in the
three-action bin, contributing two-thirds and one-third of a decision to the
respective colors. Fractional counts sum back to the original histogram.
Zero-action decisions remain a separate category, and the propensity line is
computed from the original, unsplit histogram.

### Column Schema

`column_schema.py` defines a mapping from Pega internal names to display names
with type annotations. Multiple aliases resolve to the same display name (e.g.,
`pxInteractionID` / `InteractionID` / `Interaction ID` all → "Interaction ID").
Critical columns that must be present: Interaction ID, Issue, Group, Action.

### Scope Hierarchy

`Issue → Group → Action` is the core dimension hierarchy. "Scope" controls which
level pages analyze. `get_possible_scope_values()` returns available levels based
on what columns exist. Page 6 (Win/Loss) uses this hierarchy for comparison group
definition.

### Business Lever What-if Simulation

The What-if: Business Levers page uses `da.scoring.lever_simulation()` to replace
levers for an Issue, Group, or Action. Its -50% to +50% slider maps to absolute
replacement values of 0.50 to 1.50, not multipliers of existing levers. Reset
removes the override; 0% replaces it with 1.0.

Both Before and After use reconstructed PVCL ranks on the same sampled,
population-filtered arbitration candidates. Target selectors only identify
which levers change; they do not remove competing offers. Mandatory precedence
and existing tie-breaking are retained. Counts and shares use only decisions
with rankable arbitration candidates. Decisions without such candidates are
excluded from both scenarios, with no synthetic placeholder offers.

The sidebar's "Define winning: rank in top N" control retains the top N
highest-ranked offers per interaction (default 1, up to `da.max_win_rank`).
Pass `win_rank=N` to the scoring API for the same behavior in a notebook.
Business Levers, Thresholding, Global Sensitivity, and Win/Loss use the shared
`win_rank_selector()` helper. It persists `win_rank` independently of the
`_win_rank` widget, so the selected value survives visits to pages without a
Top-N control and is restored on the next page that has one.
Selected-target percentages are decision coverage: a decision counts once when
at least one selected offer is in the top N. A separate "Analysis granularity"
control selects Issue, Group, or Action for both the delta chart and table.
Business Levers and Thresholding share this setting through
`simulation_granularity_selector()`, independently of the widget's lifetime.
Pass `by="Issue"` or `by="Group"` to the scoring API for the same aggregation.
Each decision counts once per fully qualified issue, group, or action, even if
multiple offers reach Top-N; percentages across rows may overlap. Changing
granularity neither resets the adjustment nor changes the selected-target summary.

The method returns an action-level distribution and a one-row selected-scope
summary, including the number of distinct re-evaluated arbitration decisions
and measured scoring/aggregation time. These appear in one line beneath the
table. The re-evaluated count is also the offer-share denominator. The UI shows
absolute share differences as plain numbers under "Change" (percentage points,
not relative percent changes).
Timing excludes plot construction and rendering.

The page passes the distribution to `da.plot.lever_delta()`, which shows
positive and negative winning-share changes as horizontal bars, with fully
qualified labels and Before/After counts in hover details. The plot supports
`return_df=True`. The source analyzer is not mutated. See the
[Business Levers page](../python/pdstools/app/decision_analyzer/pages/13_Business_Levers.py)
and [scoring API](../python/pdstools/decision_analyzer/_scoring.py) for the
implementation and its limitations.

### Threshold What-if Simulation

Thresholding uses `da.scoring.threshold_simulation()` and the same
`show_simulation_comparison()` renderer as Business Levers: identical bordered
delta/chart sections, one-decimal table formatting, plain numeric changes,
and an evaluation count/time footer.

Minimum propensity and recorded-priority thresholds apply globally. The After
scenario removes offers below either threshold, then re-ranks the survivors
into Top-N. Both scenarios use reconstructed PVCL priority and a fixed
population of original rankable arbitration decisions. Decisions left without
a qualifying offer remain in the denominator and contribute real lost wins,
not synthetic placeholder offers. The shared scoring implementation preserves
these baseline rows even when no offers survive.

The existing candidate-volume metrics, threshold sliders, and clickable
histograms remain available. They describe filtering before Top-N selection;
the delta comparison describes the resulting winning mix.

### Aggregate Summaries and Row Cohorts

Decision Analyzer exposes two related but distinct API shapes:

- **Aggregate summary methods** answer chart/table questions compactly. Examples:
   `da.aggregates.get_funnel_data(...)`,
   `da.aggregates.get_decisions_without_actions_data(...)`, and
   `da.aggregates.get_filter_component_data(...)`.
- **Row-producing cohort methods** return the decision rows behind a cohort and
   must include `Interaction ID`. Examples:
   `da.aggregates.remaining_at_stage(...)`,
   `da.aggregates.filtered_at_stage(...)`, and
   `da.aggregates.filtered_by_component(...)`.

When a user-facing aggregate may need downstream handoff, keep the cohort logic
in a row-producing method first, then build the summary from that row set or add
an explicit row counterpart. Do not make downstream consumers reconstruct a
cohort from an aggregate table, and do not add one-off `*_interactions()` wrapper
methods for every summary.

The generic handoff APIs live on `DecisionAnalyzer`:

```python
ids = da.get_interaction_ids("aggregates.filtered_at_stage", "Engagement Policies")
count = ids.height
```

`get_interaction_ids` accepts a public dotted method path, calls that
row-producing method with the remaining arguments, and projects unique
`Interaction ID` values. Call `.height` (or `len(...)`) on the returned frame
when you need the cohort size. `Interaction ID` is the boundary: Decision
Analyzer must not resolve it to customer IDs, subject IDs, decision dates,
accounts, households, or profile attributes. That join belongs in the downstream
application that owns the identity model and retention rules.

Canonical aggregate-to-row mappings:

| Summary question | Aggregate method | Row-producing method for IDs |
| --- | --- | --- |
| Distribution at a stage | `da.aggregates.get_distribution_data(...)` | `aggregates.remaining_at_stage` with the same stage and cell filters |
| Funnel available actions | `da.aggregates.get_funnel_data(...)[0]` | `aggregates.available_at_stage` |
| Funnel passing actions | `da.aggregates.get_funnel_data(...)[1]` | `aggregates.passing_at_stage` |
| Funnel filtered actions | `da.aggregates.get_funnel_data(...)[2]` | `aggregates.filtered_at_stage` |
| Decisions left without actions | `da.aggregates.get_decisions_without_actions_data(...)` | `aggregates.without_actions_at_stage` |
| Top filter components | `da.aggregates.get_filter_component_data(...)` | `aggregates.filtered_by_component` |

Rules of thumb:

- Put aggregate and row-cohort methods in `_aggregates.py` under
   `da.aggregates`, not directly on `DecisionAnalyzer`.
- Use full `decision_data` for exact row cohorts. Sample-backed scoring methods
   can stay sample-backed for performance, but they should not pretend to provide
   exact downstream cohorts.
- If an aggregate method collapses away row identity, document or add its
   row-producing counterpart before exposing it as a downstream handoff path.
- Keep aggregate outputs compact. They should not carry subject/customer/date
   details or embedded interaction-ID lists unless the method is explicitly a row
   cohort.

---

## Patterns to Follow When Adding/Modifying Pages

### New Analysis Page Checklist

1. Create `pages/N_Page_Name.py` with a docstring at the top (rendered as page intro)
2. Call `ensure_data()` first (guards + CLI banner)
3. Set `st.session_state["sidebar"] = st.sidebar`
4. Add page-specific controls in the sidebar block
5. Call `contextual_filters()` as the **last** sidebar item
6. Compute `filtered_data = da.filtered(collect_page_filters())` for filtered access to the sample
7. Check for empty results when channel filter is active (standard pattern in all pages)
8. Use `additional_filters=channel_filter` when calling DA query methods

### Adding a New Contextual Filter

1. Add selector function in `da_streamlit_utils.py` (follow `channel_direction_selector()` pattern)
2. Add it inside `contextual_filters()` after the channel selector
3. Store selection in `page_{name}_filter` / `page_{name}_expr` session state keys
4. Append the new `page_{name}_expr` key to the tuple in `collect_page_filters()`
5. Update empty-data checks on all pages

### Adding a New DA Method

1. Choose the right namespace first: `da.aggregates` for summary and row-cohort
   queries, `da.scoring` for sample-backed scoring/reranking questions,
   `da.plot` for figures, and top-level `DecisionAnalyzer` only for shared
   lifecycle or compatibility surface.
2. Add an `additional_filters` parameter when callers need reusable filtering.
3. Use `apply_filter(self.da.decision_data, additional_filters)` for exact
   full-data row cohorts and `apply_filter(self.da.sample, additional_filters)`
   for explicitly sample-backed methods.
4. If a summary identifies a cohort that downstream may need, add or reuse a
   row-producing method that returns rows with `Interaction ID`, then let callers
   use `get_interaction_ids("aggregates.method_name", ...)` and call `.height` on
   the result for the cohort size.
5. If it needs stage filtering, reuse the aggregate namespace stage helpers such
   as `remaining_at_stage(...)`, `available_at_stage(...)`, or
   `filtered_at_stage(...)`.
6. If expensive, consider adding to one of the cache layers.
7. Add corresponding plot function in `plots.py` if visualization is needed.
8. Wrap in `da_streamlit_utils.py` with `@st.cache_data` if called from Streamlit.

---

## Known Gotchas

1. **Stage level switch invalidates everything.** `set_level()` clears all cached
   properties. New cached properties must be added to the invalidation list there.

2. **Pre-aggregation is expensive.** `preaggregated_filter_view` collects the full
   dataset. It runs once per data load but can take seconds on large data.

3. **Color consistency from full data.** `color_mappings` assigns colors from the
   unsampled dataset so colors stay consistent across pages. Unused categories
   still claim color slots.

4. **Two sampling layers.** CLI `--sample` reduces data before DA sees it. DA's
   `sample_size` then sub-samples further. Both are interaction-level and
   hash-based.

5. **v1 has only two stages.** Many pages assume multiple stages exist. Guard
   against this when the stage list is short.

6. **Widget state across pages.** Streamlit loses widget state on page navigation.
   `_persist_widget_value()` in `da_streamlit_utils.py` works around this by
   copying widget keys to stable session state keys.

7. **Mandatory actions skew analysis.** Priority 4999999+ bypasses normal ranking.
   `is_mandatory` column flags these; some analyses should exclude or separate them.
