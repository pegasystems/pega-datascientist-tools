import streamlit as st

from plots import getTrendChart, plot_offer_quality_piecharts
from da_streamlit_utils import (
    get_current_scope_index,
    get_current_stage_index,
)
from utils import (
    NBADScope_Mapping,
    ensure_data,
    filtered_action_counts,
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
            "FinalPropensity", [0, 5, 100]
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

    scope_index = get_current_scope_index(scope_options)
    st.selectbox(
        "Scope",
        options=scope_options,
        format_func=lambda option: NBADScope_Mapping[option],
        index=scope_index,
        key="scope",
    )
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

# TODO: see about moving this into a class
action_counts = filtered_action_counts(
    df=st.session_state.decision_data.sample,
    groupby_cols=["pxEngagementStage", "pxInteractionID", "day"]
    + [st.session_state.scope],
    priorityTH=priorityTH,
    propensityTH=propensityTH,
)

# Pie Chart

vf = st.session_state.decision_data.get_offer_quality(
    action_counts, group_by="pxInteractionID"
)
# st.write(vf.head().collect())

st.plotly_chart(
    plot_offer_quality_piecharts(
        vf,
        propensityTH=propensityTH,
        NBADStages_FilterView=st.session_state.decision_data.NBADStages_FilterView,
        NBADStages_Mapping=st.session_state.decision_data.NBADStages_Mapping,
    ),
    use_container_width=True,
)

## Trend Chart

vf = st.session_state.decision_data.get_offer_quality(
    action_counts, group_by=["pxInteractionID", "day"]
)

st.plotly_chart(
    getTrendChart(vf, stage=st.session_state.stage),
    use_container_width=True,
)
