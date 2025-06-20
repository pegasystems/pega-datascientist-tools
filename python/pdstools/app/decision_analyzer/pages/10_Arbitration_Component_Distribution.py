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
Analyze the distribution of Prioritization Components, recognizing that since prioritization uses a multiplication formula, components with excessively high or low value ranges may dominate the decision-making process.
Use histograms to visualize the volume of actions within specific value ranges, and box plots to perform detailed distribution analysis.

Key questions this analysis helps answer:

* **Distribution Analysis**: Are any of your prioritization components (Propensity, Value, Context Weight, or Levers) heavily skewed? Does this align with your business expectations?

* **Component Ranges**: What is the range and spread of each component? For instance, if Value ranges from 0.1 to 100, the multiplication-based prioritization may be dominated by this component.

* **Action Comparison**: Do certain actions have significantly different component values compared to others? This can reveal potential configuration issues or business rule impacts.

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
