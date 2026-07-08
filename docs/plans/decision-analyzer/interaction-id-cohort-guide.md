# Interaction ID cohort guide for downstream users

**Priority:** P2
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`, downstream pen-portrait workflows.

## Purpose

Decision Analyzer exposes two kinds of answers:

1. Aggregate methods that summarize behavior, counts, distributions, or comparisons.
2. Interaction ID methods that return the exact decision cohort behind a question.

Downstream applications should use `Interaction ID` as the handoff key. Decision Analyzer should not resolve those IDs to customer, subject, or date attributes. The upstream application owns that mapping because it knows which customer identity, time window, and data-retention rules apply.

The generic API architecture for aggregate summaries, row-producing cohorts,
and interaction-ID projection is documented in
[`docs/decision-analyzer-architecture.md`](../../decision-analyzer-architecture.md).
This guide applies that pattern to downstream cohort analysis and pen-portrait
workflows.

## Aggregate-first workflow

Use aggregate methods to decide where to investigate. These methods answer questions such as:

- How many actions were filtered at each stage?
- Which actions win or lose arbitration?
- Which two cohorts overlap in a head-to-head comparison?
- Which priority component or stage explains a shift?

Examples:

```python
stage_summary = da.aggregates.filtered_actions_per_stage()
head_to_head = da.aggregates.head_to_head_at_stage(
    pl.col("Group") == "Retention",
    pl.col("Group") == "Sales",
    stage="Arbitration",
)
```

Aggregate results are intentionally compact. They should not carry customer or subject details.

## Interaction ID workflow

When a downstream workflow needs the exact decisions behind an aggregate, define the cohort as rows first. From that same row-producing method, ask for the interaction IDs, and call `.height` on the returned frame when you need the cohort size.

```python
output_ids = da.get_interaction_ids("aggregates.remaining_at_stage", "Output")
output_count = output_ids.height
```

`get_interaction_ids()` forwards all positional and keyword arguments to the named public method and projects unique `Interaction ID` values from the returned Polars frame. For a cohort size, call `.height` (or `len(...)`) on that frame.

```python
web_output_ids = da.get_interaction_ids(
    "aggregates.remaining_at_stage",
    "Output",
    pl.col("Channel") == "Web",
)
```

Use this pattern for public row-producing methods that return `Interaction ID`. If a method returns an aggregate summary without `Interaction ID`, `get_interaction_ids()` raises an error instead of guessing.

Aggregate methods can still exist for user-facing summaries, but they should not be the only place where a cohort is defined. If downstream users need both the summary and the exact decisions behind it, keep the cohort logic in a row-producing method and build the summary from that row set.

Common aggregate-to-row mappings:

| Summary question | Aggregate method | Row-producing method for IDs |
| --- | --- | --- |
| Distribution at a stage | `da.aggregates.get_distribution_data(...)` | `aggregates.remaining_at_stage` with the same stage and cell filters |
| Funnel available actions | `da.aggregates.get_funnel_data(...)[0]` | `aggregates.available_at_stage` |
| Funnel passing actions | `da.aggregates.get_funnel_data(...)[1]` | `aggregates.passing_at_stage` |
| Funnel filtered actions | `da.aggregates.get_funnel_data(...)[2]` | `aggregates.filtered_at_stage` |
| Decisions left without actions | `da.aggregates.get_decisions_without_actions_data(...)` | `aggregates.without_actions_at_stage` |
| Top filter components | `da.aggregates.get_filter_component_data(...)` | `aggregates.filtered_by_component` |

For example, after identifying a funnel stage with many filtered actions:

```python
filtered_ids = da.get_interaction_ids(
    "aggregates.filtered_at_stage",
    "Eligibility",
    pl.col("Issue") == "Sales",
)
filtered_count = filtered_ids.height
```

After identifying a high-impact filter component:

```python
component_ids = da.get_interaction_ids(
    "aggregates.filtered_by_component",
    "EligibilityRule",
    stage="Eligibility",
)
```

## Set-derived row cohorts

Some cohorts require set logic before they become rows. Those should still be exposed as row-producing methods first, and then projected with `get_interaction_ids()`.

`dropped_at_stage()` is one example: it returns rows for interactions that were present at one stage and absent at the next stage. The interaction IDs are still requested through the generic projector.

```python
dropped_rows = da.aggregates.dropped_at_stage(
    "Contact Policies and final Action processing"
)

dropped_ids = da.get_interaction_ids(
    "aggregates.dropped_at_stage",
    "Contact Policies and final Action processing",
)
dropped_count = dropped_ids.height
```

Use specific row-producing methods only when the cohort logic is distinct. Do not add convenience `*_interactions()` wrappers.

## Downstream resolution boundary

Decision Analyzer returns only `Interaction ID` for cohort handoff. A downstream application can join those IDs to its own data model to resolve:

- customer ID
- subject ID
- decision date or date range
- account or household keys
- application-specific profile attributes

That join should happen outside pdstools.

## What the downstream pen-portrait app should do

The pen-portrait app should treat Decision Analyzer as the owner of decision
cohort definitions, not as the owner of customer identity or profile enrichment.
The clean contract is:

1. Use aggregate methods to let the user find an interesting cohort, such as a
    high-filtering component, a funnel stage with heavy drop-off, or a sparse
    action distribution cell.
2. Convert that selected summary cell into the matching public row-producing
    method path under `aggregates.*`, passing the same stage, scope, component,
    and filter arguments that define the selected cohort.
3. Call `get_interaction_ids(...)` when the downstream app needs the handoff
    key, and call `.height` on the returned frame when it needs the exact cohort
    size for display or validation.
4. Store the cohort recipe with the output: method path, positional arguments,
    keyword arguments, user-selected filters, generated count, and generation
    timestamp. That makes the pen portrait auditable and reproducible.
5. Join the returned `Interaction ID` values to downstream-owned identity,
    profile, and time-window data outside pdstools.

The pen-portrait app should not reconstruct cohorts by reversing aggregate
tables. Aggregate tables are summaries and may collapse away row identity. It
should also not ask Decision Analyzer for subject IDs, customer IDs, dates,
accounts, households, or profile attributes. Those fields depend on the
downstream data model, retention policy, and enrichment rules, so they belong in
the downstream app.

For example, a selected filter component can be represented as a stable cohort
recipe:

```python
cohort_method = "aggregates.filtered_by_component"
cohort_args = (component_name,)
cohort_kwargs = {"stage": stage, "additional_filters": selected_filters}

interaction_ids = da.get_interaction_ids(
     cohort_method,
     *cohort_args,
     **cohort_kwargs,
)
interaction_count = interaction_ids.height
```

The downstream app can then enrich `interaction_ids` using its own data and
render the pen portrait from that enriched dataset.

## Design rules

- Prefer aggregate methods for summary analysis.
- Prefer `get_interaction_ids("aggregates.method_name", ...)` for direct row-cohort handoff.
- Call `.height` on the returned frame when an aggregate needs the count for the same cohort.
- Add named row-producing methods for distinct set logic, then use `get_interaction_ids()` to project their IDs.
- Do not add convenience `*_interactions()` wrappers.
- Do not add `include_subject_id`, `include_customer_id`, or date-resolution arguments to Decision Analyzer cohort APIs.
- Do not expose `source` switches for exact cohort APIs; interaction cohorts use the full normalized decision data.
