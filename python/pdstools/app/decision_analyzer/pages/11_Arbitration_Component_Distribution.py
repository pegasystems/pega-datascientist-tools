# python/pdstools/app/decision_analyzer/pages/10_Arbitration_Component_Distribution.py
import streamlit as st
from da_streamlit_utils import (
    ensure_data,
    get_current_index,
    stage_level_selector,
    stage_selectbox,
    st_component_overview,
    st_priority_component_distribution,
)

from pdstools.decision_analyzer.utils import PRIO_COMPONENTS


"# Arbitration Component Distribution"

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

# ---------------------------------------------------------------------------
# Overview: all components at a glance
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Overview — All Components"
    """
    A compact violin panel showing every prioritization component at once.
    The embedded box plot marks the median and interquartile range; the
    violin shape reveals the full density — skew, bimodality, long tails —
    that histograms and box plots alone can hide.
    """
    overview_data = st.session_state.decision_data.all_components_distribution(
        st.session_state.scope,
        stage=st.session_state.stage,
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
    )

    violin_fig, ecdf_fig, stats_df = st_priority_component_distribution(
        value_data,
        component=st.session_state.prioritization_component,
        granularity=st.session_state.scope,
    )

    violin_tab, ecdf_tab, stats_tab = st.tabs(["Violin Plot", "Cumulative Distribution", "Summary Statistics"])

    with violin_tab:
        """
        The violin shape shows the full density of the component — no binning
        artifacts. The embedded box marks the median and IQR.
        """
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
