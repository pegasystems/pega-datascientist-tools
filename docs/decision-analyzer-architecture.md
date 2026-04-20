# Decision Analyzer Architecture

**Updated:** April 1, 2026 — Guide for making changes to the Decision Analysis Tool.

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

Applied via `DecisionAnalyzer.filtered_sample` property, which reads session state
and filters the sample. To add new contextual filters (e.g., Issue, Group), extend
`contextual_filters()`, add corresponding session state keys, and update
`filtered_sample` to collect all filter expressions.

### Ranking

Actions are ranked per interaction by: `is_mandatory` desc → `Priority` desc →
`Stage Order` desc → alphabetic tiebreaker (Issue, Group, Action). The `Rank`
column (1 = winner) is derived in `cleanup_raw_data()`.

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

---

## Patterns to Follow When Adding/Modifying Pages

### New Analysis Page Checklist

1. Create `pages/N_Page_Name.py` with a docstring at the top (rendered as page intro)
2. Call `ensure_data()` first (guards + CLI banner)
3. Set `st.session_state["sidebar"] = st.sidebar`
4. Add page-specific controls in the sidebar block
5. Call `contextual_filters()` as the **last** sidebar item
6. Read `filtered_sample` and `page_channel_expr` from session state
7. Check for empty results when channel filter is active (standard pattern in all pages)
8. Use `additional_filters=channel_filter` when calling DA query methods

### Adding a New Contextual Filter

1. Add selector function in `da_streamlit_utils.py` (follow `channel_direction_selector()` pattern)
2. Add it inside `contextual_filters()` after the channel selector
3. Store selection in `page_{name}_filter` / `page_{name}_expr` session state keys
4. Update `DecisionAnalyzer.filtered_sample` to read the new expression
5. Update empty-data checks on all pages

### Adding a New DA Method

1. Add method to `DecisionAnalyzer` class with `additional_filters` parameter
2. Use `apply_filter(self.sample, additional_filters)` for filtered sample access
3. If expensive, consider adding to one of the cache layers
4. If it needs stage filtering, use `_remaining_at_stage(stage, additional_filters)`
5. Add corresponding plot function in `plots.py` if visualization needed
6. Wrap in `da_streamlit_utils.py` with `@st.cache_data` if called from Streamlit

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
