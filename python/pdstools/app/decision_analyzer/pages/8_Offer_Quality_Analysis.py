import streamlit as st

from pdstools.decision_analyzer.plots import getTrendChart, offer_quality_piecharts
from da_streamlit_utils import (
    get_current_index,
    ensure_data,
)
# TODO generalize a bit: no actions, just one, with a low propensity, sufficient
# TODO support the propensity based categories for those stages that have it
# TODO code align the way we name the stages in the "remaining" view like done elsewhere
# TODO we store a little too much in the session, that seems unnecessary

"# Offer Quality Analysis"

"""
**Value Finder** has a unique concept of attaching a propensity to every action even if it is
filtered out before Arbitration. **Decision Analyzer** does not have these propensities - as
it is running from actual production data.

But there are a lot of Value Finder - like analyses that can still be done here. For example,
looking at the number of interactions with just very few actions. And for the actions at or
after arbitration, we can do the value finder analyses that relate to the propensity value
or the number of actions driven by immature, new, models.
"""
ensure_data()
st.session_state["sidebar"] = st.sidebar
defaultPercentile = 0.05

with st.session_state["sidebar"]:
    scope_options = st.session_state.decision_data.getPossibleScopeValues()
    stage_options = st.session_state.decision_data.getPossibleStageValues()

    default_propensity_th = [
        round(x, 4)
        for x in st.session_state.decision_data.getThresholdingData(
            "Propensity", [0, 5, 100]
        )["Threshold"].to_list()
    ]
    default_priority_th = [
        round(x, 4)
        for x in st.session_state.decision_data.getThresholdingData(
            "Priority", [0, 5, 100]
        )["Threshold"].to_list()
    ]

    # TODO too much kept in session state here, not necessary

    propensityTH = st.slider(
        "Propensity threshold",
        default_propensity_th[0],
        default_propensity_th[2],
        default_propensity_th[1],
        # step=(default_propensity_th[2]-default_propensity_th[0])/10,
        format="%.4f",
    )
    priorityTH = st.slider(
        "Priority threshold",
        default_priority_th[0],
        default_priority_th[2],
        default_priority_th[1],
        # step=(default_priority_th[2]-default_priority_th[0])/10,
        format="%.4f",
    )

    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Scope",
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="scope",
    )
    stage_index = get_current_index(stage_options, "stage")
    st.selectbox(
        "Select Stage",
        options=stage_options,
        index=stage_index,
        key="stage",
    )

action_counts = st.session_state.decision_data.filtered_action_counts(
    groupby_cols=["StageGroup", "Interaction ID", "day"] + [st.session_state.scope],
    priorityTH=priorityTH,
    propensityTH=propensityTH,
)

# Pie Chart

vf = st.session_state.decision_data.get_offer_quality(
    action_counts, group_by="Interaction ID"
)
# st.write(vf.head().collect())

st.plotly_chart(
    offer_quality_piecharts(
        vf,
        propensityTH=propensityTH,
        AvailableNBADStages=st.session_state.decision_data.AvailableNBADStages,
    ),
    width="stretch",
)

## Trend Chart

vf = st.session_state.decision_data.get_offer_quality(
    action_counts, group_by=["Interaction ID", "day"]
)

st.plotly_chart(
    getTrendChart(vf, stage=st.session_state.stage),
    width="stretch",
)
