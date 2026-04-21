# Decision Analysis Tool — TODO

This directory holds one Markdown file per open follow-up item for the
Decision Analysis Tool (`python/pdstools/decision_analyzer/` and
`python/pdstools/app/decision_analyzer/`).

## Why one file per item?

Same reason as `docs/plans/adm/`: parallel PRs on a single TODO file
create recurring merge conflicts on the Open/Done sections. One file per
item eliminates that entirely.

## File format

Filename: `<short-slug>.md`. Contents — concise, aim for under one screen:

```markdown
# <one-line title>

**Priority:** P1 / P2 / P3
**Touches:** `<files or areas>`
<optional: **Depends on:** <other-slug>>

<problem statement>

## Approach

<concrete proposal>

<any cross-refs / issue links>
```

## Listing open items

```
ls docs/plans/decision-analyzer/
```

Find all P1 items:

```
grep -l 'Priority:** P1' docs/plans/decision-analyzer/*.md
```

## Marking done

Just `git rm docs/plans/decision-analyzer/<slug>.md` in the PR that
resolves it. The deletion is the audit trail.

## Navigation by area

| Slug prefix | Area |
|---|---|
| *(no prefix)* | Core library (`decision_analyzer.py`, schema, utils) |
| `custom-sidebar`, `add-issue-filter`, `add-cascading-group-filter`, `improve-file-upload`, `dynamic-stage-ui`, `promote-data-read-utils` | Streamlit app architecture |
| `page2-*` | Page 2 — Overview |
| `page3-*` | Page 3 — Action Distribution |
| `page4-*` | Page 4 — Action Funnel |
| `page5-*` | Page 5 — Global Sensitivity |
| `page6-*` | Page 6 — Win/Loss Analysis |
| `page7-*` | Page 7 — Optionality Analysis |
| `page8-*` | Page 8 — Offer Quality Analysis |
| `page9-*` | Page 9 — Thresholding Analysis |
| `page10-*` | Page 10 — Arbitration Component Distribution |
| `business-lever-*` | Hidden — Business Lever Analysis (stashed, needs rework) |
| `plots-*` | `plots.py` |
| `data-size-warning` | Performance |
