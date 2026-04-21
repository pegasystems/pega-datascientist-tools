# Lazy plotly imports in `Plots.py`

**Priority:** P2
**Touches:** `python/pdstools/adm/Plots.py`

Plotly is currently imported at module level inside a `try/except
ImportError` (`adm/Plots.py:35-40`). This violates the AGENTS.md "lazy
imports inside the method" rule for optional deps, and means importing
`ADMDatamart` pulls plotly even when no plot is rendered.

## Approach

Move the imports inside each plot method, mirroring the pattern in
`pdstools/utils/local_model_utils.py`. Or have `Plots` properly leverage
the existing `LazyNamespace` base class for deferred resolution.

Type annotations referencing `plotly.graph_objects.Figure` should move
under `if TYPE_CHECKING:` (the file should already have
`from __future__ import annotations`, otherwise add it as task hygiene).
