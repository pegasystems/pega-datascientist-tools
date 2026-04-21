# Plotly box sizing

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/plots.py`

Box plots ignore size constraints, leading to layout overflow in narrow Streamlit containers.

## Approach

Adopt explicit height/width with responsive scaling (e.g. use `height` param with a sensible default; allow override). Test at various container widths.
