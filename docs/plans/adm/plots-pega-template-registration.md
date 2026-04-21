# `adm/Plots/ (package)` relies on cross-module side effect for "pega" template

Priority: P2

Files touched: `python/pdstools/adm/Plots/ (package)`,
possibly `python/pdstools/utils/pega_template.py`,
`python/pdstools/__init__.py`.

## Problem

Several methods in `adm/Plots/ (package)` (e.g. `bubble_chart`, `over_time`)
pass `template="pega"` to Plotly without importing
`pdstools.utils.pega_template`. Those methods only work because some
*other* module — historically `prediction/Plots.py` — imports
`pega_template` at module load time, which runs the
`pio.templates["pega"] = ...` registration as a side effect.

This was exposed during the prediction gold-standard refactor when
`prediction/Plots.py` moved its plotly imports inside individual
methods. Without the cross-module side effect, calling
`ADMDatamart.plot.bubble_chart()` raises:

```
ValueError: The first argument to the plotly.graph_objs.layout.Template
constructor must be a dict or an instance of
:class:`plotly.graph_objs.layout.Template`
```

Because the prediction refactor still triggered the side effect via a
deliberate module-level shim, the regression was masked — but the
underlying coupling remains brittle.

## Proposed approach

Two reasonable options, in order of preference:

1. **Lazy import in each adm/Plots/ (package) method** that uses
   `template="pega"` — `from ..utils import pega_template  # noqa: F401`
   alongside the existing inline `import plotly.express as px`. This
   matches the gold-standard pattern (each method owns its imports)
   and removes the cross-module coupling entirely.
2. **Register templates in `pdstools/__init__.py`** behind a guarded
   `try: from .utils import pega_template`. Centralises the
   side effect. Less surgical but ensures any pdstools consumer
   gets the templates regardless of which submodules they touch.

After fixing this, the `_pega_template` shim in
`prediction/Plots.py` (added 2025-XX as a workaround) can be removed.
