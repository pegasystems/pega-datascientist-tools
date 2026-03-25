# Decision Analyzer — Backlog

Active work items for the Decision Analysis Tool.

## Quick Reference

**Blockers (fix before release):**
- None currently

**High Impact (fix soon):**
- [P1] `num_samples > 1` breaks thresholding — affects data quality, users cannot get representative distributions
- [P1] Overview page slow first load — 10-30 second wait creates poor UX
- [P1] Mandatory actions skew analysis — sensitivity and win/loss metrics misleading

**Priority levels:**
- **[P1]** High — Critical functionality, poor UX, or blocking issues
- **[P2]** Medium — Important improvements, moderate UX impact
- **[P3]** Low — Nice-to-have, minor issues, future enhancements

**Total items:** 45+ across core library, app architecture, pages, and plots

**Recently Completed (Mar 2026):**
- ✅ Color consistency across all plots (color_mappings property)
- ✅ Channel/Direction filter expansion to 7+ pages
- ✅ Component drilldown granularity selector (scope parameter)
- ✅ Streamlit width="stretch" deprecation fix
- ✅ Page naming consistency (Arbitration Distribution)
- ✅ Action Funnel redesign — three tabs (Passing, Filtered, Decisions w/o Actions), funnel summary table, unified component analysis
- ✅ Design principle: decisions shown as % of total, actions as avg/decision, "Interactions" → "Decisions" throughout UI
- ✅ Offer Variation section moved from Optionality to Offer Quality page

---

## Core Library

### High Priority

- [ ] **[P1] Fix `num_samples > 1`** — Pre-aggregation sampling ([DecisionAnalyzer.py:569](../python/pdstools/decision_analyzer/DecisionAnalyzer.py#L569)) is locked at 1 because >1 breaks thresholding. **Consequence:** Users cannot get representative distributions across groups; sampling only provides one data point per group regardless of group size, potentially missing important variation patterns in small-volume groups. **Root cause:** With multiple samples per group, `.explode()` in `getThresholdingData` ([line 1236](../python/pdstools/decision_analyzer/DecisionAnalyzer.py#L1236)) creates multiple rows per aggregated group, biasing quantile calculations toward groups with more underlying interactions. **Solutions:** (A) Weight samples properly in thresholding, (B) Calculate quantiles at group level before exploding, (C) Use min/max columns instead of samples for distributions, or (D) Document the limitation and keep `num_samples=1`.

### Medium Priority

- [ ] **[P2] Raw IH data support** — Test with raw Interaction History format (similar to EE v1). **Consequence:** Users with only IH data (not EE exports) cannot use the tool, limiting adoption in organizations that haven't migrated to EE. IH may lack columns needed for certain analyses (e.g., component details, treatment info). **Action:** Identify missing columns, test with sample IH data, decide whether to support as first-class format or document limitations and required preprocessing.

- [ ] **[P2] Refactor `get_offer_quality`** — Uses manual stage loop instead of delegating to `aggregate_remaining_per_stage`. **Consequence:** Code duplication increases maintenance burden; inconsistent with other aggregation methods; harder to optimize or extend to new use cases. **Action:** Rewrite to use `aggregate_remaining_per_stage`, verify results match current implementation, update tests.

- [ ] **[P2] Win rank flexibility** — `get_win_loss_distribution_data` has fixed rank parameter and hardcoded `max_value=10`. **Consequence:** Users cannot analyze ranks beyond 10, even when data has higher ranks; UI must make multiple calls to get different ranks; performance impact from redundant processing; artificial limit doesn't match data. **Action:** Return all ranks in single call, let UI filter as needed; compute `max_value` from actual max rank in data (`df.select(pl.col("Rank").max())`).

- [ ] **[P1] Handle Mandatory Actions** — Mandatory actions use special arbitration priority (4999999+) that bypasses normal ranking. **Consequence:** Sensitivity analyses are skewed because mandatory actions never change rank regardless of lever adjustments; win/loss distributions are misleading (mandatory wins look like normal wins); users cannot distinguish mandatory from competitive arbitration outcomes; lever recommendations may be wrong. **Status:** `DecisionAnalyzer` already accepts a `mandatory_expr` parameter to tag issue/group/action combinations as mandatory at init time, and ranking sorts by `is_mandatory` field before Priority. **Remaining work:** (1) Auto-detect from priority value in data (instead of requiring explicit expression), (2) Flag/annotate mandatory actions in all UI visualizations with distinct color/marker, (3) Exclude from sensitivity analysis or show separately, (4) Consider separate treatment in component distributions and win/loss analyses. See [Pega docs: NBA Strategy Arbitration](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/nba-strategy-arbitration.html).

- [ ] **[P2] Distinct propensity display names** — `column_schema.py` maps model propensity and (final) propensity to same display name. **Consequence:** Users cannot distinguish between raw model output and arbitration-adjusted propensity in UI; confusing for lever analysis where the distinction matters (model propensity is input, final propensity is after lever adjustments); charts and tables ambiguous. **Action:** Use "Model Propensity" for raw model output vs "Final Propensity" for arbitration-adjusted value, update `column_schema.py` and all PVCL references.

### Low Priority

- [ ] **[P3] Pre-aggregated data visualization** — The Infinity Action Analysis feature exports pre-computed results (sensitivity, funnel data) rather than raw interactions. **Consequence:** Missed opportunity for faster load times with pre-aggregated data; users must run full pipeline even when Infinity has already done the work. **Action:** Add data loader branch that detects pre-aggregated format and bypasses `DecisionAnalyzer` aggregation, directly feeds plots module.

- [ ] **[P3] Treatment support** — Currently not supported; Treatment column commented out in `ExplainabilityExtract` schema. **Consequence:** Users analyzing treatment-level data (creative variants, channel placements) see duplicated actions with different propensities; all current metrics are wrong for treatment-enabled data (actions counted multiple times, propensity values ambiguous). **Scope:** Affects ~0.6% of data rows; 284 actions with treatments in sample data; each action has 1-16 treatments with varying propensities. **Action:** See detailed analysis below in "Treatment Support Analysis" section. Key decisions needed: (1) aggregation strategy (max propensity vs best rank vs weighted average), (2) UX approach (automatic aggregation vs user choice vs auto-detect prompt), (3) which metrics need treatment-specific handling.

- [ ] **[P3] Customer-level aggregates** — Per-customer statistics and trends. **Consequence:** Cannot analyze customer-level patterns (e.g., "do high-value customers see different action mixes?"); limited to interaction-level view. **Status:** Postponed; current sample data not representative of real customer distributions. **Action:** Wait for representative multi-customer dataset, then design customer aggregation methods.

- [ ] **[P3] AB test: include IA properties** — `getABTestResults` doesn't include Impact Analyzer properties when populated in data. **Consequence:** Users running A/B tests cannot see treatment effects on business outcomes (CTR, conversion, revenue) that IA would provide; limited to arbitration metrics only. **Action:** Extend method to join IA properties when available, add to output dataframe.

- [ ] **[P3] Optimize thresholding quantiles** — Current implementation is verbose and potentially slow on large datasets. **Consequence:** Slight performance overhead in thresholding analysis; code harder to maintain. **Action:** Profile performance, refactor to use vectorized polars operations if measurably faster.

- [ ] **[P3] Stratified sampling option** — Current `sample` property is pure random sampling by interaction ID. **Consequence:** Small channels/directions may be underrepresented or missing entirely in samples; analysis may miss important segments. **Action:** Add optional stratified sampling mode (e.g., `sample_stratified_by=["Channel", "Direction"]`) that ensures proportional representation across specified dimensions.

- [ ] **[P3] Scale up counts in UI after sampling** — When sampling is applied, displayed counts are actual sample counts, not estimates of true volumes. **Consequence:** Users see artificially low numbers that don't reflect reality; can't estimate production impact from sample analysis. **Status:** Metadata infrastructure now in place to track sample fraction. **Action:** Add UI toggle to show "Estimated Full Volume" by multiplying counts by inverse of sample fraction; clearly indicate estimates vs actuals.

- [ ] **[P3] Streaming pre-aggregation** — `getPreaggregatedFilterView` calls `.collect()` on full dataset before aggregation. **Consequence:** Memory pressure on GB-scale datasets; potential OOM crashes; slower than necessary. **Action:** Investigate polars streaming mode or chunked processing to aggregate without full materialization; benchmark memory and speed improvements.

---

## Streamlit App — Architecture

- [ ] **[P1] Custom state-aware sidebar navigation with disabled pages** — Streamlit's default multipage nav hides unavailable pages, which makes capabilities and prerequisites unclear (especially for EEv1 vs EEv2). **Consequence:** Users cannot see the full app surface area, do not understand why features are missing, and may hit confusing behavior when trying to access pages directly by URL. **Action (mini plan):** (1) Define a central page registry (single source of truth) in `da_streamlit_utils.py` with page id, label, route, prerequisites, and availability predicate (e.g., v1/v2 compatibility, data loaded, workflow requirements). (2) Disable the default Streamlit multipage navigation and render a custom sidebar menu from that registry, always showing all pages; render unavailable pages as greyed-out text with disabled/non-clickable controls. (3) Attach concise reason messages to each disabled page (e.g., "Requires Action Analysis / EEV2 data" or "Load data on Home first") and show the same reason in page-level guards. (4) Add a shared `ensure_page_access(page_id)` guard and call it at the top of every page to enforce identical access rules and block direct URL access with a clear warning + `st.stop()`. (5) Add tests for navigation state and guard enforcement (EEv1 vs EEv2, no data loaded, direct URL attempt), and document the page availability matrix in Decision Analyzer docs.

- [ ] **[P2] Improve file upload robustness** — Apply lessons from Impact Analyzer file upload improvements. **Consequence:** Users dragging unzipped Pega export folders get confusing errors about MANIFEST.mf files; failed uploads show stale success messages from previously loaded data; unclear error messages when wrong file format is uploaded. **Current state:** DA filters `__MACOSX` files but not `MANIFEST.mf` or other META-INF files; doesn't clear session state when new upload attempted; relies solely on file extension for format detection. **Action:** (1) Filter out META-INF files in `handle_file_upload()` similar to [IA Home.py:134-142](../python/pdstools/app/impact_analyzer/Home.py#L134-L142); (2) Clear `decision_data` from session state when new upload is attempted (line 69); (3) Consider content-based format detection in addition to extension-based; (4) Improve error messages to explain what format is expected and why upload failed.

- [ ] **[P2] Dynamic stage UI** — Mostly data-driven, but some hardcoded stages remain. **Consequence:** Tool breaks or shows wrong/empty data when stage names differ from expected values (e.g., customer has "OutputStage" instead of "Output", or no "Final" stage); limits portability across Pega versions and configurations. **Locations:** [8_Optionality_Analysis.py:97-100](../python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py#L97-L100) hardcodes `stage="Output"` for action variation; [plots.py:439](../python/pdstools/decision_analyzer/plots.py#L439) filters `"Final"` stage without checking existence. **Action:** Get all stage names from `DecisionAnalyzer.stages` property, use last stage instead of hardcoded "Output", check stage existence before filtering.

- [ ] **[P3] HC data import alignment** — Health Check uses different `import_datamart()` pattern with different labels than Decision Analyzer's data loader. **Consequence:** Maintenance burden (parallel implementations of similar functionality); confusing code patterns for contributors; harder to share improvements between tools; inconsistent UX (different file type labels, different error messages). **Action:** Evaluate consolidating to common pattern, possibly using `data_read_utils` as shared foundation.

- [ ] **[P3] Promote `data_read_utils` to pdstools core** — The data-reading facilities in `decision_analyzer/data_read_utils.py` (multi-format ingestion, schema detection, lazy scanning) are generic and could serve all apps. **Consequence:** Each app reimplements file reading (duplication in ADM Health Check, Value Finder, Prediction Studio); inconsistent UX across tools; missed optimization opportunities; harder to add new formats. **Action:** Extract to `pdstools/pega_io` or top-level utility module, potentially replacing legacy `readDSExport` family; refactor existing apps to use shared implementation; ensure backward compatibility.

---

## Streamlit Pages

### Page 2 — Global Data Filters

- [ ] **[P2] Robustness against heavy filtering** — Aggressive filters (e.g., selecting only one stage, filtering out major channels) can leave downstream pages with empty or invalid data. **Consequence:** Pages crash with cryptic polars errors; users lose work and trust in tool; unclear which filter combination caused the problem. **Action:** Add data validation after filter application; show warning if filters eliminate >95% of data or result in <100 interactions; provide "reset filters" escape hatch on error pages.

- [ ] **[P3] Apply filters by default when set** — Currently filters are defined but user must manually apply them. **Consequence:** Users may forget to apply filters and analyze wrong data; extra click required; not intuitive. **Action:** Add toggle "Auto-apply filters" (default on) that applies filters immediately when changed; maintain manual "Apply" button for users who want to batch changes.

### Page 3 — Overview

- [ ] **[P1] Speed up first-use page load** — First render computes all stage aggregations, metrics, and charts upfront. **Consequence:** 10-30 second wait on large datasets (appears frozen); poor first impression; users may think app crashed; unnecessary computation for tabs user never opens. **Action:** Lazy-load stage-specific metrics only when tabs are opened; show loading spinner with progress indicator; consider background computation with "still loading" indicators; prioritize visible content first.

- [ ] **[P2] Add Value Finder-style pie chart at arbitration** — Users migrating from Value Finder expect this visualization showing win/loss proportions. **Consequence:** Less intuitive for existing VF users; harder to see arbitration impact at a glance; missing familiar visual reference. **Action:** Add pie chart showing % of interactions where top action wins vs loses at arbitration stage, matching VF design; place prominently near arbitration stage metrics.

### Page 4 — Action Distribution

- [ ] **[P2] Dynamic stages from data** — Some stage references may still be hardcoded. **Consequence:** Charts break or show empty data when expected stage names differ. **Action:** Audit page for any hardcoded stage names, replace with data-driven stage selection using `DecisionAnalyzer.stages` property.

- [ ] **[P2] Top-K limiter for bar charts** — Bar charts with >50 actions are unreadable; x-axis labels overlap. **Consequence:** Cluttered visualizations; can't read action names; poor UX for large action catalogs. **Action:** Add "Show top N" control (default 20, max 50) with sorting options (by volume, by propensity, by win rate); clearly indicate "Showing top 20 of 150 actions" with option to download full data.

- [ ] **[P3] Multi-stage selection** — Currently shows one stage at a time. **Consequence:** Users cannot compare distributions across stages side-by-side; must screenshot or remember values; harder to see how actions drop off through pipeline. **Action:** Add multi-select for stages; show side-by-side bar charts or small multiples layout; ensure colors consistent across stages.

### Page 5 — Action Funnel

- [ ] **[P2] "Decisions w/o Actions" view** — The Pega product's funnel analysis includes a "Decisions w/o Actions" view that shows, per stage, the number of decisions ending in zero actions. The app's funnel currently shows "Average Actions per Interaction" and "Reach" (% of interactions with ≥1 action) but lacks a dedicated view counting interactions where *all* actions have been filtered out by a given stage. **Consequence:** Users cannot quickly see how many decisions result in no actions at all — a critical indicator of over-filtering. The "Reach" metric is related (100% − Reach ≈ % with no actions) but is a percentage buried in hover text, not an explicit count per stage. **Action:** Add a "Decisions w/o Actions" tab or toggle in the funnel page. For each stage, compute the number of unique interactions with zero `REMAINING` records at that stage. Display as a bar chart showing how the zero-action count grows through stages. Implementation: use `aggregate_remaining_per_stage` grouped by `Interaction ID`, then count interactions whose action count is 0 at each stage. This helps users pinpoint exactly which stages cause the most decisions to lose all their actions.

- [ ] **[P2] Restore funnel-shaped view for Passing Actions** — The classic funnel visualisation (narrowing bars showing how actions pass through each stage) is missing. The current bar chart shows average actions per decision per stage but doesn't convey the funnel shape intuitively. **Action:** Redesign the Passing Actions tab to use a proper funnel chart (or waterfall) so the narrowing across stages is visually obvious.

- [ ] **[P2] Rethink Filter Impact Details table** — The summary table currently shows avg actions per decision and % decisions with actions. Consider whether absolute counts (Available/Passing/Filtered actions and raw decision count per stage) are more useful or whether a mixed display (both absolute and relative) serves users better. **Action:** User research or A/B test; decide on final column set.

- [ ] **[P2] Toggle between Passing and Filtered perspective** — Currently shows only actions passing through stages (remaining after filters). **Consequence:** Hard to see filter impact; users can't visualize what was removed; unclear which filters are most restrictive. **Action:** Add toggle "Show: Remaining / Filtered Out / Both" to switch between perspectives; use contrasting colors for passed vs filtered; show filter-specific impact.

- [ ] **[P2] Better coloring for filter components** — All filters use similar colors regardless of type. **Consequence:** Hard to distinguish engagement filters from eligibility filters from suitability filters; cognitive load to parse charts. **Action:** Use consistent color scheme by filter category (e.g., blue for eligibility, orange for suitability, green for engagement); add legend; consider icons for common filter types.

- [ ] **[P2] Align funnel filter view with product (show post-stage counts)** — The filter view in the funnel is shifted with respect to the Pega product: it currently shows incoming counts rather than the results after each stage. **Consequence:** Mismatch with the product’s funnel confuses users who cross-reference; harder to compare tool output against Pega’s built-in analysis; “remaining” numbers don’t match expectations when reading stage-by-stage. **Action:** Shift displayed counts so each stage shows results *after* that stage’s filters have been applied, aligning with the product’s convention; verify against product screenshots; update labels to make the perspective unambiguous.

- [ ] **[P3] Move top-N control from sidebar to section** — Top-N action selector is in sidebar, far from affected charts. **Consequence:** Poor UX (control separated from effect); unclear what top-N applies to; sidebar cluttered. **Action:** Move control to inline position above affected charts; make scope clear with label "Show top N actions by final volume".

- [ ] **[P3] Component overlap analysis** — Cannot see which actions are affected by multiple filters simultaneously. **Consequence:** Users don't know if filters are redundant or complementary; can't optimize filter strategy; unclear if filters compound or overlap. **Action:** Add UpSet plot or Sankey diagram showing filter overlap patterns; show "removed by multiple filters" vs "removed by single filter" breakdown.

- [ ] **[P3] Component impact over time** — Filter effectiveness may vary over time (seasonal, campaign-related). **Consequence:** Static view doesn't reveal temporal patterns; can't identify trending issues; miss opportunities to adjust filters based on changing behavior. **Action:** Add time-series trend chart showing filter removal rates over time; allow date range selection; flag significant changes.

### Page 6 — Global Sensitivity

- [ ] **[P2] Infer top-X from data** — Currently uses user-specified top-X value, but optimal value depends on data. **Consequence:** Users don't know what value to pick; may choose too high (noisy results) or too low (miss important actions); inconsistent across datasets. **Action:** Auto-compute from data: use max rank per channel for Final records as default; show data-driven suggestion; explain reasoning in help text.

- [ ] **[P2] Win rank upper bound from data** — Max rank hardcoded in some places (see Core Library item). **Consequence:** Analysis artificially limited; can't see sensitivity for lower-ranked actions even when they exist in data. **Action:** Use `df.select(pl.col("Rank").max())` to set upper bound dynamically; remove hardcoded limits.

### Page 7 — Win/Loss Analysis

- [ ] **[P2] Verify numbers across visualizations** — Bar charts and box plots sometimes show different counts for same data. **Consequence:** Users lose confidence in tool; unclear which number is correct; potential bugs in aggregation logic. **Action:** Audit all metrics computations; ensure bar charts and box plots use identical data sources; add data validation checks; show counts in both views for cross-reference.

- [x] **[DONE] Consistent colors across paired charts** — ✅ Completed Mar 2026: Win/loss charts now use consistent color_mappings. Remaining: Ensure minimum contrast ratio 4.5:1 for accessibility; consider patterns/shapes for colorblind users.

- [x] **[PARTIAL] Handle many channels gracefully** — ✅ Completed Mar 2026: Channel/Direction filter added to 7+ pages. Remaining: Consider faceted plots with scrolling for very large channel lists; grouping low-volume channels as "Other".

- [ ] **[P3] Generalize rank usage across pages** — Rank selection UI differs between Win/Loss, Sensitivity, and other pages. **Consequence:** Inconsistent UX; users must relearn controls on each page; different defaults may confuse. **Action:** Create shared rank selector component in `da_streamlit_utils.py`; standardize on data-driven default; consistent placement and labeling.

### Page 8 — Optionality Analysis

- [ ] **[P2] Overlay propensity + priority on optionality plot** — Currently shows only counts; component distributions not visible. **Consequence:** Users can't see how propensity/priority correlate with optionality levels; miss insights about whether high-propensity actions have more options; unclear if low optionality is due to poor propensity or filtering. **Action:** Add dual-axis or faceted view showing average propensity and priority alongside optionality distribution; use color gradient for propensity; add toggle to switch between metrics.

- [x] **[DONE] Consistent stage color scheme** — ✅ Completed Mar 2026: Stage colors now consistent via color_mappings property across all plots.

### Page 9 — Offer Quality Analysis

- [ ] **[P2] Generalize stage naming** — May have hardcoded stage references or assumptions about stage sequence. **Consequence:** Breaks with non-standard stage configurations; not portable across Pega versions. **Action:** Audit for hardcoded stages; use `DecisionAnalyzer.stages` property; make stage selection data-driven.

- [ ] **[P2] Session state cleanup** — Page may store intermediate results in session state unnecessarily. **Consequence:** Session state bloat; stale data persists across page navigation; harder to debug; potential memory leaks. **Action:** Review what's stored in `st.session_state`; remove anything that can be recomputed cheaply; use `@st.cache_data` for expensive computations instead of manual caching.

- [ ] **[P2] Move logic to DecisionAnalyzer class** — Business logic for offer quality calculations may be in Streamlit page code. **Consequence:** Can't reuse logic in other contexts (CLI, reports, tests); harder to test; violates separation of concerns. **Action:** Extract offer quality methods to `DecisionAnalyzer` class; page should only handle UI presentation.

### Page 10 — Thresholding Analysis

- [ ] **[P2] Move inline plots to `plots` module** — Three plotly charts ([lines 134-174, 202-209](../python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py#L134-L174)) defined inline in page code. **Consequence:** Code duplication if plots reused elsewhere; page file bloated; harder to maintain consistent styling; can't test plots independently. **Action:** Extract to reusable functions in `plots.py` (e.g., `plot_threshold_distribution()`, `plot_threshold_impact()`); page should only configure and display.

- [ ] **[P1] Depends on `num_samples = 1`** — Page uses `.explode()` on samples throughout (lines 31-32, 155, 187-188). **Consequence:** Will break if `num_samples > 1` is ever enabled (see Core Library High Priority item); blocks improvement to sampling strategy. **Action:** Coordinate with `num_samples` fix; update page to handle multiple samples correctly (aggregate before exploding, or use min/max columns).

### Page 11 — Arbitration Component Distribution

- [ ] **[P2] EE v1: include all propensity-like properties** — Explainability Extract v1 has multiple propensity flavors (e.g., model propensity, adjusted propensity, final propensity) whereas v2 only has one. **Consequence:** V1 users miss detailed propensity breakdown; can't analyze lever impact on different propensity stages; reduced value for V1 datasets. **Action:** Detect V1 vs V2 format; for V1, show component distributions for all propensity columns; add tabs or facets to compare model vs adjusted vs final; ensure column_schema properly maps V1 propensity columns.

- [ ] **[P3] Finish proposition distribution side-by-side view** — Partially implemented feature for comparing component distributions across propositions. **Consequence:** Users can only view one proposition at a time; can't compare propensity distributions for similar actions; harder to identify patterns. **Action:** Complete implementation of side-by-side view; allow multi-select of propositions; show overlapping violin/box plots with transparency; add statistical comparison (e.g., KS test for distribution difference).

### Hidden Page — Business Lever Analysis *(needs full rework)*

> **Status:** Page moved to `_stashed/Business_Lever_Analysis.py` as of the `feat/external-user-readiness` branch. Not ready for external users. Before re-enabling, the page needs a thorough redesign.
>
> **Purpose:** Support what-if analysis of lever changes — let users simulate the effect of adjusting business levers on win distributions and, ultimately, automatically find the right lever values to reach specific goals (target win rates, volume targets, etc.).
>
> **Why hidden:** The UX is confusing (unclear terminology, non-intuitive controls), the code bypasses the standard data/session-state patterns used by other pages, and the lever-finder feature is too rough for production use.
>
> **To re-enable:** Address the items below, rename the file back to a numbered page (e.g. `11_Business_Lever_Analysis.py`), and renumber About accordingly.

- [ ] **[P1] Redesign UX** — Current interface is confusing. **Consequence:** Users don't understand what levers do or how to use the page; unclear terminology ("weight" vs "lever" vs "priority"); non-intuitive controls; can't interpret results. **Action:** User research on terminology; redesign with side-by-side before/after distributions; add explanatory text and examples; show clear cause-and-effect; use consistent terms from Pega docs.

- [ ] **[P1] Generalize to action sets** — Currently works on single action only. **Consequence:** Limited utility; users want to adjust levers for groups of actions (e.g., all retention offers); must repeat process for each action; misses cross-action effects. **Action:** Allow multi-select or action group selection; show aggregate impact across selected actions; handle interactions between actions.

- [ ] **[P1] Refactor lever calculation to DecisionAnalyzer** — Lever logic is embedded in Streamlit page. **Consequence:** Can't reuse in other contexts; hard to test; violates architecture; inconsistent with other pages. **Action:** Create `simulate_lever_changes()` method in `DecisionAnalyzer` class; page should only handle UI.

- [ ] **[P2] Use aggregated/sampled data** — Currently uses raw `decision_data`, bypassing normal data flow. **Consequence:** Slow on large datasets; ignores user-selected sample/filters; different behavior from other pages; potential memory issues. **Action:** Use `DecisionAnalyzer.filtered_data` property that respects sampling and filters; consistent with architecture.

- [ ] **[P2] Move plots to plots module** — Charts defined inline. **Consequence:** Code duplication with Thresholding Analysis (similar distribution plots); can't maintain consistent styling; harder to test. **Action:** Extract to `plots.py`, share visualization code with Thresholding Analysis page.

- [ ] **[P3] Start target win ratio at > 0** — Min value for target slider is 0. **Consequence:** Confusing (0% win rate is meaningless); solver may fail or behave unexpectedly with zero target. **Action:** Set min to 1% or 5%; add validation.

- [ ] **[P3] Improve lever-finder algorithm** — Current optimization algorithm is brittle. **Consequence:** May not converge; gives nonsensical lever values; no feedback on why it failed; users lose trust. **Action:** Add constraints (lever values must be realistic); improve solver convergence; show optimization progress; provide fallback if no solution found; explain failure modes.

---

## Plots (`plots.py`)

- [ ] **[P2] Plotly box sizing** — Multiple box plots ignore size constraints and render too large or too small. **Consequence:** Charts overflow containers on small screens; inconsistent sizing across plots; poor mobile experience; layout breaks. **Action:** Adopt the solution from ADM Datamart Plots (explicit height/width parameters with responsive scaling); audit all box plot calls; test on various screen sizes.

- [ ] **[P3] Legend suppression** — `showlegend=False` parameter not working in some polar/bar charts; legends appear despite being disabled. **Consequence:** Cluttered visualizations when legend is redundant; wastes screen space; inconsistent behavior across chart types. **Action:** Investigate Plotly version-specific workarounds; may need `layout.showlegend=False` instead; test with current Plotly version; document any version-dependent behavior.

- [ ] **[P2] Hover info in stacked histograms** — Stacked bar charts only show individual segment value on hover, not total bar height. **Consequence:** Users must do mental math to get totals; reduced data clarity; poor UX compared to industry standard behavior. **Action:** Add hover template showing both individual segment value and total bar height; format as "Component: X (Y% of total Z)"; ensure works across all stacked charts.

- [ ] **[P3] Pie chart stage limit** — Temporary hardcoded cap at 5 stages to prevent overcrowding. **Consequence:** Tool fails or shows incomplete data with >5 stages; arbitrary limit doesn't match data; unclear to user why stages are missing. **Action:** Make dynamic based on actual stage count; for >5 stages, use donut chart or horizontal bar chart instead; add "show all stages" expansion option; or group low-volume stages as "Other".

---

## Data Size & Performance

- [ ] **[P2] Data size warning in UI** — No feedback when loading large files; app becomes unresponsive without explanation. **Consequence:** Users load multi-GB files without realizing impact; app freezes or crashes; poor experience with out-of-memory errors; users lose work; no guidance on appropriate data size. **Thresholds:** Warn at >500MB or >5M rows (expect slow performance); strongly discourage >2GB (may fail); show estimated memory usage and expected load time. **Action:** Add size check in data loader with clear warning message; suggest sampling for large files; show progress bar during load; add "memory mode" setting for large datasets (streaming vs full load).

---

## Treatment Support Analysis

**Status:** Under investigation. Treatment column is currently commented out in `ExplainabilityExtract` schema (column_schema.py:202). Before enabling, need to resolve aggregation strategy and user experience questions.

### Background

EE v2 exports contain a granularity hierarchy: **Issue → Group → Action → Treatment**

- Not all customers use treatments (~0.6% of export rows in sample data)
- When present: ~3 treatments per action on average (range: 1-16)
- Current code assumes action-level granularity: `scope_hierarchy = ["Issue", "Group", "Action"]` (DecisionAnalyzer.py:907)

### The Core Problem

When treatment data exists, **the same action appears multiple times per interaction** with different treatments. Each treatment has its own propensity and priority values. This creates duplication when analyzing at the action level.

**Example from sample data:**
- Action: "SavingsNudgeMoneyThor" in single interaction appears twice:
  - Treatment A: propensity 0.979, priority 0.979
  - Treatment B: propensity 0.564, priority 0.564

For action-level analysis (e.g., "which actions win at arbitration?"), we need to either:
1. **Aggregate treatments to actions** — collapse multiple treatment rows into one action row
2. **Support treatment-level granularity** — extend all analyses to work at treatment level

### Data Behavior (from 81M row sample)

**Treatment prevalence:**
- 470K rows (0.6%) have treatment values; 81M (99.4%) do not
- 284 actions have treatments; 820 unique treatments total
- At Output (Final) stage: 235 actions appear with multiple treatments (~0.4% duplication)

**Treatment scoring:**
- Treatments appear AFTER action-level propensity scoring
- Pipeline stages: TreatmentJoins (2600) → TreatmentAvailability (2900) → TreatmentPlacements (4900)
- Each treatment gets its own `FinalPropensity` and `Priority` (Priority = FinalPropensity)
- For same action, propensities vary significantly by treatment:
  - Example: "OnlineStatements" (16 treatments) — propensities range 0.41 to 0.59
  - Each treatment represents a different creative/channel variant

**Arbitration implications:**
- Multiple treatments for same action compete in arbitration separately
- Each treatment-action pair has its own rank and win/loss record
- When aggregating to action level, unclear which treatment's metrics to use

### Open Questions

#### 1. Aggregation Strategy for Action-Level Analysis

When rolling up treatments to actions, which treatment "represents" the action?

**Option A: Max Propensity**
- Use treatment with highest `FinalPropensity`
- Rationale: Most likely to convert
- Issue: May not be the one selected by arbitration if other factors matter

**Option B: Max Priority**
- Use treatment with highest `Priority`
- Rationale: Most likely to rank highly
- Issue: Priority = Propensity in this data, so equivalent to Option A

**Option C: Best Rank**
- Use treatment with best (lowest) rank after arbitration
- Rationale: Reflects actual selection logic
- Issue: Requires computing rank first; rank is stage-specific

**Option D: Weighted Average**
- Aggregate propensity/priority by some weighting (e.g., by treatment occurrence frequency)
- Rationale: Represents "typical" treatment performance
- Issue: Unclear what weight to use; loses information about best-case

#### 2. Aggregation Scope

Which metrics need aggregation, and how?

| Metric | Current Type | Aggregation Strategy |
|--------|--------------|---------------------|
| `FinalPropensity` | Float | Max? Mean? Take from selected treatment? |
| `Priority` | Float | Max (makes sense for ranking) |
| `Value` | Float | Max? Mean? Take from selected treatment? |
| `ContextWeight` | Float | Same as Value? |
| `Weight` (Levers) | Float | Same as Value? |
| `Rank` | Int | Min (best rank) |

#### 3. User Experience

Should treatment support be:

**Option A: Automatic Aggregation**
- Always aggregate treatments to actions at data load
- Pro: Simpler UX; action-level view matches user mental model
- Con: Loses treatment-level detail; user can't analyze treatment performance

**Option B: User Choice**
- Add UI toggle: "Action-level" vs "Treatment-level" analysis
- Pro: Preserves full data; users who care about treatments can analyze them
- Con: More complex UX; need to handle both modes throughout app

**Option C: Auto-detect + Explicit Choice**
- If treatments present, prompt user at data load: "Aggregate to actions or analyze treatments?"
- Store choice in `DecisionAnalyzer` instance
- Pro: Explicit about the decision; users understand what they're analyzing
- Con: Extra interaction; may confuse users unfamiliar with treatments

### Recommended Investigation Steps

Before implementing, need to:

1. **Validate arbitration behavior**
   - When multiple treatments for same action compete, which wins?
   - Does the highest-ranked treatment always have highest propensity?
   - Test: For actions with multiple treatments, check if rank correlates with propensity

2. **Understand treatment selection in Pega**
   - Is treatment selection deterministic (same action always picks same treatment)?
   - Or does it vary by context (same action can pick different treatments for different customers)?
   - Check Pega docs on treatment arbitration logic

3. **Test aggregation strategies**
   - Implement both "max propensity" and "best rank" aggregation
   - Compare results against raw treatment-level data
   - Validate that action-level win/loss metrics make sense

4. **Check v1 vs v2 differences**
   - Does EE v1 have treatments? (Need to test with v1 data)
   - Do v1 and v2 handle treatments differently?

5. **User research**
   - Do users expect to see treatments in the tool, or just actions?
   - What questions do users have about treatment performance?
   - Would treatment-level analysis add value, or just complexity?

### Implementation Considerations

**If aggregating at data load:**
- Add treatment aggregation step in `cleanup_raw_data()` when Treatment column is present
- Use max propensity or best rank to select representative treatment
- Drop Treatment column after aggregation so rest of code works unchanged

**If supporting both granularities:**
- Add `granularity` parameter to `DecisionAnalyzer.__init__`: `"action"` (default) or `"treatment"`
- Extend `scope_hierarchy` to optionally include `"Treatment"`: `["Issue", "Group", "Action", "Treatment"]`
- Update all groupby operations: `get_win_distribution_data`, `aggregate_remaining_per_stage`, `get_component_impact_detailed`, etc.
- Add UI control to switch between action/treatment views
- Handle case where treatment data doesn't exist (fall back to action level)

**Testing needs:**
- Test with data containing treatments (sample_99k_1.parquet)
- Test with data without treatments (need to find sample)
- Test with v1 data (if v1 supports treatments)
- Verify win/loss, sensitivity, and ranking metrics are correct at both granularities

### Related TODO Items

This work relates to:
- **Distinct propensity display names** — Need to clarify model propensity vs final propensity, especially for treatments
- **Handle Mandatory Actions** — Mandatory actions may or may not have treatments; ensure aggregation doesn't break mandatory logic
