# Actionable recommendations per finding in `Analysis.findings()`

**Priority:** P2
**Files touched:** `python/pdstools/adm/Analysis.py`
**Tracks:** PR #623 follow-up (issue #8).

## Problem

`Finding.detail` today describes *what* the issue is. The real-data
sweep across 8 production CDH deployments highlighted that triagers
read the same set of findings repeatedly and would benefit from a
`recommendation` field that names the *next concrete step* — not just
"check the response loop" but "open the OutcomeMapping rule for
configuration X and verify the positive label".

## Proposed approach

Add an optional `recommendation: str = ""` field on `Finding`. Populate
it for the highest-frequency check outcomes first:

| Finding                                            | Recommended action                                                                  |
| -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `data_quality` invalid Channel/Direction           | "Audit upstream metadata; filter or fix in the model export."                       |
| `channel` N channels have zero responses           | "Confirm channels are configured in NBAD strategy; remove if intentionally retired."|
| `configuration` … has no positives                 | "Open OutcomeMapping rule X; verify the positive outcome label is firing."          |
| `model` X% never used                              | "Trim never-used actions or activate them in production."                           |
| `model` mature but under-performing                | "Add behavioural / IH predictors to the configuration."                             |
| `taxonomy` Total treatments below minimum          | "Define at least 3 treatments per channel for arbitration to be meaningful."        |
| `taxonomy` Total actions above typical             | "Consolidate dormant actions; review propensity for retiring."                      |

Surface `recommendation` in `__str__` only when `verbose=True` (or as a
trailing dim line) so the default rendering stays scannable.

## Notes

- Keep recommendations short (≤ 1 sentence each); long-form guidance
  belongs in docs, not in a finding.
- Avoid client-specific terminology; phrasings should make sense for
  any Pega CDH deployment.
- Add an exact-value test per recommendation string so they don't
  silently drift in subsequent edits.
