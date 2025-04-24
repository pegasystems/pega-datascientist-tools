import streamlit as st
from da_streamlit_utils import (
    ensure_data,
    get_current_index,
    st_value_distribution,
)

from pdstools.decision_analyzer.utils import NBADScope_Mapping

# # TODO Finish up to show effect on proposition distribution (side to side)

"# Business Value Analysis"

"""
A closer look at the values associated with actions.

* Is my value distribution very skewed? Are there actions with significantly different values than the others?
* What's the range of the values?

"""
ensure_data()

st.session_state["sidebar"] = st.sidebar

scope_options = st.session_state.decision_data.getPossibleScopeValues()
if "scope" not in st.session_state:
    st.session_state.scope = scope_options[0]

valueData = st.session_state.decision_data.getValueDistributionData()

with st.container(border=True):
    st.plotly_chart(
        st_value_distribution(valueData, st.session_state.scope),
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
