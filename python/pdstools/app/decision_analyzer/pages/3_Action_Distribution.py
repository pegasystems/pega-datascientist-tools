import streamlit as st


from da_streamlit_utils import (
    get_current_index,
    ensure_data,
    stage_level_selector,
)


# st.set_option("global.showWarningOnDirectExecution", False)

# TODO the stages need to be much more dynamic, driven from the data and potentially be many - see latest mocks Dennis as well in GOAL-25903
# TODO given there can be many stages, we should perhaps see about selecting multiple, coordinate w Dennis on this but only if we have such data
# TODO consider a top K argument again, to limit the bars, also consider the plotly option to always show all ticks, easily misleading otherwise (or show somewhere that there are more)
# TODO consider a plot of how often "winning", at rank x, or is this simply the final stage

"""
# Distribution of the Actions

Shows which actions are present at a given pipeline stage. Use the **Stage
Granularity** toggle in the sidebar to switch between coarse stage groups
and individual stages, then pick a stage to inspect.

* What are the most common offers? In which issues/groups?
* How does the distribution change after filtering stages?
* Does the distribution look okay?
"""

ensure_data()

st.session_state["sidebar"] = st.sidebar

with st.session_state["sidebar"]:
    stage_level_selector()

    scope_options = st.session_state.decision_data.getPossibleScopeValues()
    stage_options = st.session_state.decision_data.getPossibleStageValues()

    stage_index = get_current_index(stage_options, "stage")
    st.selectbox(
        "Select Stage",
        options=stage_options,
        index=stage_index,
        key="stage",
    )
distribution_data = st.session_state.decision_data.getDistributionData(
    st.session_state.stage, scope_options
)
st.plotly_chart(
    st.session_state.decision_data.plot.distribution_as_treemap(
        df=distribution_data,
        stage=st.session_state.stage,
        scope_options=scope_options,
    ),
    use_container_width=True,
)

if "scope" not in st.session_state:
    st.session_state.scope = scope_options[0]

f"""
## Trend Chart

Number of decisions that included at least one action from each {st.session_state.scope} over time.

Note: Since a decision can contain actions across multiple [issues/groups], the same decision may be counted in several categories, so the stacked total may exceed the actual sampled decision count.
"""
with st.container(border=True):
    fig, warning_message = st.session_state.decision_data.plot.trend_chart(
        st.session_state.stage, st.session_state.scope
    )
    if warning_message:
        st.warning(warning_message)
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="scope",
    )
