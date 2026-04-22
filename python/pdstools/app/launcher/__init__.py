"""Cross-app launcher hosting Health Check, Decision Analyzer, and
Impact Analyzer in a single Streamlit process.

Each tool keeps its own per-section sidebar group via
``st.navigation``'s sectioned dict API; URL paths are flat-namespaced
(``/hc_*``, ``/da_*``, ``/ia_*``) to avoid collisions between tools
that share page names. Standalone tool launches keep their original
URLs intact so existing bookmarks don't break.
"""
