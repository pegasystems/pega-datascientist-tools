# python/pdstools/app/decision_analyzer/pages/8_Offer_Quality_Analysis.py
import streamlit as st

from pdstools.decision_analyzer.plots import getTrendChart, offer_quality_piecharts
from da_streamlit_utils import (
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)
# TODO generalize a bit: no actions, just one, with a low propensity, sufficient
# TODO support the propensity based categories for those stages that have it
# TODO code align the way we name the stages in the "remaining" view like done elsewhere
# TODO we store a little too much in the session, that seems unnecessary

"# Offer Quality Analysis"

"""
Identify customer interactions where offer quality may be a concern. This analysis
helps you spot situations where customers see too few offers, or where offers have
low predicted response rates or priorities.

**Key insights:**
* Which customers are seeing limited choices?
* Are there interactions with only low-quality offers?
* Where might new or untested offers be impacting customer experience?

**Note:** Propensity-based quality assessment (distinguishing between "relevant" and
"irrelevant" offers) is only available from the arbitration stage onward. For earlier
stages, the analysis shows whether customers received at least one action, without
quality assessment.
"""
ensure_data()
st.session_state["sidebar"] = st.sidebar
defaultPercentile = 0.05


def _safe_thresholds(thresholding_data):
    """Extract threshold values, returning None if no data survives to arbitration."""
    values = thresholding_data["Threshold"].to_list()
    if all(v is None for v in values):
        return None
    return [round(v, 4) if v is not None else 0.0 for v in values]


propensity_th = _safe_thresholds(st.session_state.decision_data.getThresholdingData("Propensity", [0, 5, 100]))
priority_th = _safe_thresholds(st.session_state.decision_data.getThresholdingData("Priority", [0, 5, 100]))

if propensity_th is None or priority_th is None:
    st.warning(
        "⚠️ No actions survive to the arbitration stage in this data set. "
        "Offer Quality Analysis requires actions at or after arbitration. "
        "Please check your data or filters."
    )
    st.stop()

with st.session_state["sidebar"]:
    stage_level_selector()

    propensityTH = (
        st.slider(
            "Minimum propensity for relevance",
            propensity_th[0] * 100,
            propensity_th[2] * 100,
            propensity_th[1] * 100,
            format="%.2f%%",
            help="Offers with propensity below this value are considered irrelevant (low quality)",
        )
        / 100
    )
    priorityTH = st.slider(
        "Minimum priority for relevance",
        priority_th[0],
        priority_th[2],
        priority_th[1],
        # step=(priority_th[2]-priority_th[0])/10,
        format="%.4f",
        help="Offers with priority below this value are considered irrelevant (low quality)",
    )


action_counts = st.session_state.decision_data.filtered_action_counts(
    groupby_cols=[st.session_state.decision_data.level, "Interaction ID", "day"],
    priorityTH=priorityTH,
    propensityTH=propensityTH,
)

with st.container(border=True):
    "## Customer Segments by Offer Quality"

    vf = st.session_state.decision_data.get_offer_quality(action_counts, group_by="Interaction ID")

    st.plotly_chart(
        offer_quality_piecharts(
            vf,
            propensityTH=propensityTH,
            AvailableNBADStages=st.session_state.decision_data.AvailableNBADStages,
            level=st.session_state.decision_data.level,
        ),
        width="stretch",
    )

with st.container(border=True):
    "## Offer Quality Over Time"

    stage_selectbox()

    vf = st.session_state.decision_data.get_offer_quality(action_counts, group_by=["Interaction ID", "day"])

    st.plotly_chart(
        getTrendChart(vf, stage=st.session_state.stage, level=st.session_state.decision_data.level),
        width="stretch",
    )
