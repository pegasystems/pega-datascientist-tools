# python/pdstools/app/decision_analyzer/pages/10_Arbitration_Component_Distribution.py
import polars as pl
import streamlit as st
from da_streamlit_utils import (
    channel_direction_selector,
    ensure_data,
    get_current_index,
    stage_level_selector,
    stage_selectbox,
    st_component_overview,
    st_priority_component_distribution,
)

from pdstools.decision_analyzer.utils import PRIO_COMPONENTS


"# Arbitration Distribution"

"""
Analyze the distribution of prioritization components. Since prioritization
uses a multiplicative formula (`Priority = Propensity × Value × Context
Weight × Levers`), components with extreme value ranges can dominate.

* **Distribution Shape**: Are components heavily skewed? Propensity often
  clusters near zero; Value and Levers can span orders of magnitude.
* **Component Ranges**: Which component has the widest spread and therefore
  the most influence on the final Priority?
* **Action Comparison**: Do certain actions have notably different component
  values?
"""

ensure_data()

st.session_state["sidebar"] = st.sidebar

scope_options = st.session_state.decision_data.getPossibleScopeValues()

# Determine which components are actually present in the data
available_cols = set(st.session_state.decision_data.sample.collect_schema().names())
component_options = [c for c in PRIO_COMPONENTS if c in available_cols]

if "scope" not in st.session_state:
    st.session_state.scope = scope_options[0]
if "prioritization_component" not in st.session_state:
    st.session_state.prioritization_component = component_options[0]

# Sidebar controls
with st.session_state["sidebar"]:
    stage_level_selector()

    stage_selectbox(default="Arbitration")

    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        index=scope_index,
        key="scope",
    )
    channel_direction_selector()

# Apply channel filter to sample data
filtered_data = st.session_state.decision_data.filtered_sample

# Check for empty results when a specific channel is selected
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.len()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()

channel_filter = st.session_state.get("page_channel_expr")

# ---------------------------------------------------------------------------
# Overview: all components at a glance
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Overview — All Components"
    st.caption(
        "A compact violin panel showing every prioritization component at once. "
        "The embedded box plot marks the median and interquartile range; the "
        "violin shape reveals the full density — skew, bimodality, long tails — "
        "that histograms and box plots alone can hide."
    )
    overview_data = st.session_state.decision_data.all_components_distribution(
        st.session_state.scope,
        stage=st.session_state.stage,
        additional_filters=channel_filter,
    )
    overview_fig = st_component_overview(overview_data, component_options, st.session_state.scope)
    st.plotly_chart(overview_fig)

# ---------------------------------------------------------------------------
# Detail: single-component deep dive
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Component Detail"

    value_data = st.session_state.decision_data.priority_component_distribution(
        component=st.session_state.prioritization_component,
        granularity=st.session_state.scope,
        stage=st.session_state.stage,
        additional_filters=channel_filter,
    )

    color_discrete_map = st.session_state.decision_data.color_mappings.get(st.session_state.scope)

    violin_fig, ecdf_fig, stats_df = st_priority_component_distribution(
        value_data,
        component=st.session_state.prioritization_component,
        granularity=st.session_state.scope,
        color_discrete_map=color_discrete_map,
    )

    violin_tab, ecdf_tab, stats_tab = st.tabs(["Violin Plot", "Cumulative Distribution", "Summary Statistics"])

    with violin_tab:
        st.caption(
            "The violin shape shows the full density of the component — no binning "
            "artifacts. The embedded box marks the median and IQR."
        )
        st.plotly_chart(violin_fig)

    with ecdf_tab:
        """
        The empirical cumulative distribution function (ECDF) is lossless: every
        data point is represented. Read it as *"X% of actions have a value ≤ Y"*.
        Particularly useful for skewed components like Propensity.
        """
        st.plotly_chart(ecdf_fig)

    with stats_tab:
        """Key percentiles and summary statistics per group."""
        st.dataframe(stats_df, hide_index=True)

    component_index = get_current_index(component_options, "prioritization_component")
    st.selectbox(
        "Prioritization Component:",
        options=component_options,
        index=component_index,
        key="prioritization_component",
    )
