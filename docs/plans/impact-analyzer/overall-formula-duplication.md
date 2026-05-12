# Overall Summary — lift formula rendered twice

**Priority:** P3
**Touches:** `python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py`

The lift LaTeX formula renders inside every active experiment card (in
`_render_experiment_card`) and again in the "Full formula reference"
expander at the bottom of the page. With 5 active experiments the same
formula appears 6 times.

## Approach

- Keep the per-card rendering — it is the transparency feature users rely
  on.
- Drop the lift entry from the bottom "Full formula reference" expander,
  or restrict that expander to the *other* formulas (CI band, EWMA,
  significance test) that are not already shown per card.
