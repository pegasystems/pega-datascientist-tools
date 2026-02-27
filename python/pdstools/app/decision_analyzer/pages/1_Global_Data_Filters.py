# python/pdstools/app/decision_analyzer/pages/1_Global_Data_Filters.py
import io
import json

import polars as pl
import streamlit as st

from da_streamlit_utils import (
    deserialize_filters,
    get_data_filters,
    reset_filter_state,
    serialize_filters,
    show_filtered_counts,
    ensure_data,
)
from pdstools.decision_analyzer.utils import get_first_level_stats

"""# Global Data Filters"""

"""Here you can filter the decision data for analysis. You don't need to, if
you don't, then **all** data will be used.

In the drop down below, select the columns you wish to filter (e.g. Group, Issue, Channel etc).
For each column, a new configuration screen will be added in which you can specify
what values you want to use in the Decision Analyzer.
"""

ensure_data()

# ── Handle pending filter reset (before any widgets render) ──────────
if st.session_state.pop("_needs_filter_reset", False):
    reset_filter_state("global")
    st.session_state["filters"] = []
    st.session_state.decision_data.resetGlobalDataFilters()
    st.cache_data.clear()

# ── Uploaded filter file ─────────────────────────────────────────────
expr_list: list[pl.Expr] = []
with st.container(border=True):
    """
    You can (optionally) save and re-apply filters you defined earlier:
    """
    uploaded_file = st.file_uploader(
        "Use the same data filters you used in previous sessions:",
        type=["json"],
    )
    if uploaded_file:
        imported_filters = json.load(uploaded_file)
        for val in imported_filters.values():
            str_io = io.StringIO(json.dumps(val))
            expr_list.append(pl.Expr.deserialize(str_io, format="json"))

# ── Build filters from widgets ───────────────────────────────────────
widget_filters = get_data_filters(
    st.session_state.decision_data.unfiltered_raw_decision_data,
    columns=st.session_state.decision_data.getAvailableFieldsForFiltering(),
    queries=expr_list,
    filter_type="global",
)

# ── Determine active filters ────────────────────────────────────────
# Widget-derived filters are authoritative when present. When empty
# (e.g. after page navigation destroyed widget state), fall back to
# the persistent JSON store so filters survive navigation.  As a last
# resort, reuse the live filter list from session state.
if widget_filters:
    active_filters = widget_filters
    st.session_state["_applied_filters_json"] = serialize_filters(active_filters)
elif "_applied_filters_json" in st.session_state:
    active_filters = deserialize_filters(st.session_state["_applied_filters_json"])
elif st.session_state.get("filters"):
    active_filters = st.session_state["filters"]
    st.session_state["_applied_filters_json"] = serialize_filters(active_filters)
else:
    active_filters = []

st.session_state["filters"] = active_filters

# ── Apply filters and show status ───────────────────────────────────
if active_filters:
    st.session_state.decision_data.resetGlobalDataFilters()
    stats_before = get_first_level_stats(st.session_state.decision_data.sample)

    st.session_state.decision_data.applyGlobalDataFilters(active_filters)

    stats_after = get_first_level_stats(st.session_state.decision_data.sample)
    st.cache_data.clear()
    show_filtered_counts(stats_before, stats_after)

    col_download, col_reset = st.columns(2)
    with col_download:
        serialized_exprs = {}
        for i, expr in enumerate(active_filters):
            serialized_exprs[i] = json.loads(expr.meta.serialize(format="json"))
        st.download_button(
            label="Save current filters",
            data=json.dumps(serialized_exprs),
            file_name="DecisionAnalyzerFilters.json",
        )
    with col_reset:
        if st.button("Reset all filters", type="secondary"):
            st.session_state["_needs_filter_reset"] = True
            st.rerun()
else:
    st.session_state.decision_data.resetGlobalDataFilters()
    st.cache_data.clear()
    st.success("No data filters applied")
