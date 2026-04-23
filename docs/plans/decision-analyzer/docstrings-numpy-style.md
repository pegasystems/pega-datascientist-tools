# Convert Google-style docstrings to numpy-style

**Priority:** P3
**Files:**
- `python/pdstools/decision_analyzer/_aggregates.py` (1 block:
  `action_variation`)
- `python/pdstools/decision_analyzer/plots/_distribution.py` (1 block:
  `plot_action_variation`)
- `python/pdstools/decision_analyzer/data_read_utils.py` (2 blocks)
- `python/pdstools/decision_analyzer/utils.py` (2 blocks)
- `python/pdstools/app/decision_analyzer/da_streamlit_utils.py` (1
  block)
- `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`
  (1 block)

## Problem

DecisionAnalyzer code mixes Google-style docstrings (`Args:` /
`Returns:`) with the numpy-style used by the rest of `pdstools`. This
renders inconsistently in Sphinx and signals the wrong convention to
reviewers.

## Approach

Convert each block to numpy-style per the convention now codified in
[AGENTS.md → Docstring style: numpy](../../../AGENTS.md). The Streamlit
helper modules don't ship as part of the public API but the convention
applies for consistency.
