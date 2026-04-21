# Impact Analyzer — TODO

This directory holds one Markdown file per open follow-up item for the
Impact Analyzer tool (`python/pdstools/impact_analyzer/` and
`python/pdstools/app/impact_analyzer/`).

## File format

Filename: `<short-slug>.md`. Contents — concise, aim for under one screen:

```markdown
# <one-line title>

**Priority:** P1 / P2 / P3
**Touches:** `<files or areas>`

<problem statement>

## Approach

<concrete proposal>
```

## Listing open items

```
ls docs/plans/impact-analyzer/
```

## Marking done

`git rm docs/plans/impact-analyzer/<slug>.md` in the PR that resolves it.
