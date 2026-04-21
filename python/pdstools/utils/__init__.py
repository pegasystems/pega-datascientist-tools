"""Internal utility modules for pdstools.

Submodules (``cdh_utils``, ``datasets``, ``streamlit_utils``,
``show_versions``, ``polars_ext`` …) are imported on demand via their
fully-qualified path. Nothing is re-exported at the package level — the
``utils`` namespace is intentionally kept empty so that helpers stay
namespaced and don't leak into the public API surface.
"""

__all__: list[str] = []
