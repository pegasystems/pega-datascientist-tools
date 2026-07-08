# Interaction ID cohort guide for downstream users

**Priority:** P2
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`, downstream pen-portrait workflows.

## Purpose

Decision Analyzer exposes two kinds of answers:

1. Aggregate methods that summarize behavior, counts, distributions, or comparisons.
2. Interaction ID methods that return the exact decision cohort behind a question.

Downstream applications should use `Interaction ID` as the handoff key. Decision Analyzer should not resolve those IDs to customer, subject, or date attributes. The upstream application owns that mapping because it knows which customer identity, time window, and data-retention rules apply.

## Aggregate-first workflow

Use aggregate methods to decide where to investigate. These methods answer questions such as:

- How many actions were filtered at each stage?
- Which actions win or lose arbitration?
- Which two cohorts overlap in a head-to-head comparison?
- Which priority component or stage explains a shift?

Examples:

```python
stage_summary = da.filtered_actions_per_stage()
head_to_head = da.head_to_head_at_stage(
    pl.col("Group") == "Retention",
    pl.col("Group") == "Sales",
    stage="Arbitration",
)
```

Aggregate results are intentionally compact. They should not carry customer or subject details.

## Interaction ID workflow

When a downstream workflow needs the exact decisions behind an aggregate, ask for interaction IDs from the row-producing method that defines that cohort.

```python
output_ids = da.get_interaction_ids("remaining_at_stage", "Output")
```

`get_interaction_ids()` forwards all positional and keyword arguments to the named public method, then projects unique `Interaction ID` values from the returned Polars frame.

```python
web_output_ids = da.get_interaction_ids(
    "remaining_at_stage",
    "Output",
    pl.col("Channel") == "Web",
)
```

Use this pattern for public row-producing methods that return `Interaction ID`. If a method returns an aggregate summary without `Interaction ID`, `get_interaction_ids()` raises an error instead of guessing.

## Set-derived row cohorts

Some cohorts require set logic before they become rows. Those should still be exposed as row-producing methods first, and then projected with `get_interaction_ids()`.

`dropped_at_stage()` is one example: it returns rows for interactions that were present at one stage and absent at the next stage. The interaction IDs are still requested through the generic projector.

```python
dropped_rows = da.dropped_at_stage(
    "Contact Policies and final Action processing"
)

dropped_ids = da.get_interaction_ids(
    "dropped_at_stage",
    "Contact Policies and final Action processing",
)
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

## Design rules

- Prefer aggregate methods for summary analysis.
- Prefer `get_interaction_ids("method_name", ...)` for direct row-cohort handoff.
- Add named row-producing methods for distinct set logic, then use `get_interaction_ids()` to project their IDs.
- Do not add convenience `*_interactions()` wrappers.
- Do not add `include_subject_id`, `include_customer_id`, or date-resolution arguments to Decision Analyzer cohort APIs.
- Do not expose `source` switches for exact cohort APIs; interaction cohorts use the full normalized decision data.
