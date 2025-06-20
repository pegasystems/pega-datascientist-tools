import streamlit as st
from da_streamlit_utils import (
    ensure_data,
    get_current_index,
    st_priority_component_distribution,
)

from pdstools.decision_analyzer.utils import NBADScope_Mapping

# # TODO Finish up to show effect on proposition distribution (side to side)

"# Arbitration Component Distribution"

"""
A closer look at the values associated with actions.

* Is my value distribution very skewed? Are there actions with significantly different values than the others?
* What's the range of the values?

"""
ensure_data()

st.session_state["sidebar"] = st.sidebar

scope_options = st.session_state.decision_data.getPossibleScopeValues()
component_options = ["Propensity", "Value", "Context Weight", "Levers"]

if "scope" not in st.session_state:
    st.session_state.scope = scope_options[0]
if "prioritization_component" not in st.session_state:
    st.session_state.prioritization_component = component_options[0]


valueData = st.session_state.decision_data.priority_component_distribution(
    component=st.session_state.prioritization_component,
    granularity=st.session_state.scope,
)

with st.container(border=True):
    hist_tab, box_tab = st.tabs(["Histogram", "Box Plot"])
    histogram, box_plot = st_priority_component_distribution(
        valueData,
        component=st.session_state.prioritization_component,
        granularity=st.session_state.scope,
    )
    with hist_tab:
        st.plotly_chart(
            histogram,
            use_container_width=True,
        )
    with box_tab:
        st.plotly_chart(
            box_plot,
            use_container_width=True,
        )

    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        format_func=lambda option: NBADScope_Mapping[option],
        index=scope_index,
        key="scope",
    )
    component_index = get_current_index(component_options, "prioritization_component")
    st.selectbox(
        "Prioritization Component:",
        options=component_options,
        index=component_index,
        key="prioritization_component",
    )
