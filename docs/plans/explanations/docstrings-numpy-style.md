# Convert Google-style docstrings to numpy-style

**Priority:** P3
**Files:**
- `python/pdstools/explanations/Aggregate.py` (3 blocks: `top_predictor_contributions`, `top_predictor_value_contributions`, `_check_aggregates_folder`)
- `python/pdstools/explanations/Plots.py` (2 blocks: `predictor_contributions`, `predictor_value_contributions`)
- `python/pdstools/explanations/FilterWidget.py` (1 block: `set_selected_context`)

## Problem

The explanations module uses Google-style docstrings (`Args:` / `Returns:`
/ `Raises:`) while the rest of `pdstools` uses numpy-style. This causes
inconsistent rendering in Sphinx and gives reviewers a false signal
about the expected style.

## Approach

Convert each block to numpy-style per the convention now codified in
[AGENTS.md → Docstring style: numpy](../../../AGENTS.md). Verify the
docs build (`cd python/docs && make html`) renders the converted
sections cleanly.
