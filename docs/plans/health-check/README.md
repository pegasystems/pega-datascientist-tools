# ADM Health Check — TODO

This directory holds one Markdown file per open follow-up item for the
ADM Health Check report (`reports/HealthCheck.qmd`, `reports/ModelReport.qmd`)
and the supporting `python/pdstools/adm/` library code.

## File format

Filename: `<short-slug>.md`. Contents — concise, aim for under one screen:

```markdown
# <one-line title>

**Priority:** P1 / P2 / P3
**Touches:** `<files or areas>`

<problem statement>

## Approach

<concrete proposal>

<GitHub issue links if relevant>
```

## Listing open items

```
ls docs/plans/health-check/
```

Find P1 items:

```
grep -l 'Priority:** P1' docs/plans/health-check/*.md
```

## Marking done

`git rm docs/plans/health-check/<slug>.md` in the PR that resolves it.

## Navigation by area

| Slug | Area |
|---|---|
| `improve-pega-plotly-template` | Plots & Plotly template (#600) |
| `consistent-active-model-filtering` | Quarto reports (#594) |
| `move-report-plot-helpers` | `reports/ModelReport.qmd`, `adm/Plots/` |
