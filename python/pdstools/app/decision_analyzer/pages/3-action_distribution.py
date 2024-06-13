import streamlit as st

from pdstools.decision_analyzer.utils import NBADScope_Mapping
from da_streamlit_utils import (
    get_current_scope_index,
    get_current_stage_index,
    ensure_data,
)


# st.set_option("global.showWarningOnDirectExecution", False)

# TODO the stages need to be much more dynamic, driven from the data and potentially be many - see latest mocks Dennis as well in GOAL-25903
# TODO given there can be many stages, we should perhaps see about selecting multiple, coordinate w Dennis on this but only if we have such data
# TODO consider a top K argument again, to limit the bars, also consider the plotly option to always show all ticks, easily misleading otherwise (or show somewhere that there are more)
# TODO consider a plot of how often "winning", at rank x, or is this simply the final stage

"""
# Distribution of the Actions

This is simply showing the overall distribution of the actions at the selected stage. This can help 
answering questions like:

* What are the most common offers? In which issues/groups?

* What are the most common offers for a certain group of issues, within a certain channel etc - by applying filters first, or using
the graph controls directly.

* Does the distribution look okay?
"""

ensure_data()

st.session_state["sidebar"] = st.sidebar

with st.session_state["sidebar"]:
    scope_options = st.session_state.decision_data.getPossibleScopeValues()
    stage_options = st.session_state.decision_data.getPossibleStageValues()

    stage_index = get_current_stage_index(stage_options)
    st.selectbox(
        "Select Stage",
        options=stage_options,
        index=stage_index,
        format_func=lambda option: st.session_state.decision_data.NBADStages_Mapping[
            option
        ],
        key="stage",
    )
distribution_data = st.session_state.decision_data.getDistributionData(
    st.session_state.stage, scope_options
)
st.plotly_chart(
    st.session_state.decision_data.plot.plot_distribution_as_treemap(
        df=distribution_data,
        stage=st.session_state.stage,
        scope_options=scope_options,
    ),
    use_container_width=True,
)

"""
## Trend Chart

NB current sample data is only a few minutes of data from a batch run that
we artificially stretched to a few weeks.
"""

with st.container(border=True):
    if "scope" not in st.session_state:
        st.session_state.scope = scope_options[0]

    fig, warning_message = st.session_state.decision_data.plot.plot_trend_chart(
        st.session_state.stage, st.session_state.scope
    )
    if warning_message:
        st.warning(warning_message)
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    scope_index = get_current_scope_index(scope_options)
    st.selectbox(
        "Granularity:",
        options=scope_options,
        format_func=lambda option: NBADScope_Mapping[option],
        index=scope_index,
        key="scope",
    )
