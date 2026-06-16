# Sample AGB dataset — deeper anonymization

**Priority:** P2
**Files touched:** `python/pdstools/datasets/` (sample data file + generation script)

## Problem

`datasets.sample_trees()` currently mirrors the original model very closely:
the predictor count, response volumes, action names, and ensemble size are
essentially unchanged from the source.  Anyone familiar with the original
model could recognise it from these fingerprints.

## ✅ Completed approach (2026-06-16)

Script: `scripts/anonymize_agb_sample.py`

- **Ensemble size** — kept trees 0–82 (83 trees); dropped last 17 (all had
  near-zero gain, no meaningful contribution). Tree count changed from 100 → 83.
- **Predictor renames** — four field names that could identify the source:
  - `Customer.Scores.Cisegmpricesen` → `Customer.Scores.ExtModelScore0`
  - `IH.Millenet.Inbound.` → `IH.Web.Inbound.` (channel brand name removed)
  - `Customer.RetailTrxDays` → `Customer.AccountAgeDays`
  - `Param.ClNewAbandonCnt1d` → `Param.SessionAbandonCnt1d`
  All renames applied in-place in split strings across all kept trees.
- **Response volumes** — scaled by ×0.71: pos 79,130 → 56,182;
  neg 14,973,407 → 10,631,119. `successRate` recomputed from scaled counts.
- **Float precision** — all `score` / `gain` values rounded to 5 decimal
  places; no impact on any plot or metric.
- **File size** — compact JSON (no whitespace); 2.8 MB → 0.95 MB (−66%).

All statistical properties preserved: gain-decay S-curve, namespace
breakdown (IH / Customer / py*), early-vs-late learners, training timeline
shape, feature-role-map clusters. All 13 `Trees.plot.*` methods verified.

## Cross-refs

- Notebook: `examples/articles/AGBExplained.ipynb`
- Plan: `docs/plans/adm/agb-explained-notebook.md`
