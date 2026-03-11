import streamlit as st


from da_streamlit_utils import (
    get_current_index,
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)


# st.set_option("global.showWarningOnDirectExecution", False)

# TODO the stages need to be much more dynamic, driven from the data and potentially be many - see latest mocks Dennis as well in GOAL-25903
# TODO given there can be many stages, we should perhaps see about selecting multiple, coordinate w Dennis on this but only if we have such data
# TODO consider a top K argument again, to limit the bars, also consider the plotly option to always show all ticks, easily misleading otherwise (or show somewhere that there are more)
# TODO consider a plot of how often "winning", at rank x, or is this simply the final stage

"""
# Action Distribution

Understand which offers are being presented to customers at each stage of your decisioning pipeline.
Use the **Stage Granularity** toggle in the sidebar to switch between high-level stage groups
and individual stages, then select a stage to analyze.

**Key questions:**
* Which offers dominate your mix? Are they the right ones?
* How does filtering change which offers reach customers?
* Is your offer portfolio balanced or concentrated in a few actions?
"""

ensure_data()

st.session_state["sidebar"] = st.sidebar

with st.session_state["sidebar"]:
    stage_level_selector()

    scope_options = st.session_state.decision_data.getPossibleScopeValues()

    stage_selectbox()

with st.container(border=True):
    "## Offer Mix by Volume"

    st.caption(
        "Visualize the relative volume of each offer at the selected stage. Box sizes represent "
        "the number of times each action appears. Explore the hierarchy by clicking through "
        "Issue → Group → Action to see how your portfolio is composed."
    )

    distribution_data = st.session_state.decision_data.getDistributionData(st.session_state.stage, scope_options)
    st.plotly_chart(
        st.session_state.decision_data.plot.distribution_as_treemap(
            df=distribution_data,
            stage=st.session_state.stage,
            scope_options=scope_options,
        ),
    )

if "scope" not in st.session_state:
    st.session_state.scope = scope_options[0]

with st.container(border=True):
    "## Offer Trends Over Time"

    st.caption(
        f"Track how often each {st.session_state.scope} appears in customer decisions over time. "
        f"Spot seasonal patterns, campaign impacts, and shifts in your offer mix. "
        f"*Note: Since decisions can include multiple {st.session_state.scope.lower()}s, the stacked total "
        f"may exceed the actual decision count — this is expected when customers see diverse offers.*"
    )

    fig, warning_message = st.session_state.decision_data.plot.trend_chart(
        st.session_state.stage, st.session_state.scope
    )
    if warning_message:
        st.warning(warning_message)
    st.plotly_chart(
        fig,
    )

    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="scope",
    )
