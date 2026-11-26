# ADM module — TODO

This directory holds one Markdown file per open follow-up item for the
`python/pdstools/adm/` module.

## Why one file per item?

Conflicts. When every PR edits the same `adm-TODO.md`, parallel branches
collide on the Open / Done sections and need rebasing. With one file per
item:

- Adding an item = creating a file (no shared state).
- Resolving an item = `git rm` the file (visible in the PR diff).
- Re-prioritising an item = editing one isolated file.

Two PRs only ever conflict if they're literally working on the same item.

## File format

Filename: `<short-slug>.md` (e.g. `lazy-plotly.md`, `binagg-rename.md`).

Contents — concise; aim for under one screen:

```markdown
# <one-line title>

**Priority:** P1 / P2 / P3
**Touches:** `python/pdstools/adm/<file>.py`, ...

<problem statement — what's wrong / inconsistent / missing>

<proposed approach — concrete enough that someone else can pick it up>

<any cross-refs: AGENTS.md sections, related issues, prior PRs>
```

## Listing open items

```
ls docs/plans/adm/
```

## Marking done

Just `git rm docs/plans/adm/<slug>.md` in the PR that resolves it.
The PR description / commit message captures the *what* and *why*; the
removal of the file is the audit trail. No "Done" section to maintain.

## Cross-references

- [Library architecture and data](../../../AGENTS.md#library-architecture-and-data)
  — namespace facades and shared I/O boundaries.
- [Plots, reports, and notebooks](../../../AGENTS.md#plots-reports-and-notebooks)
  — figure and `return_df` contracts.
- [Git, collaboration, and follow-up work](../../../AGENTS.md#git-collaboration-and-follow-up-work)
  — when to file an issue or a plan entry.
- `docs/plans/decision-analyzer/` — DA module backlog (per-item layout).
