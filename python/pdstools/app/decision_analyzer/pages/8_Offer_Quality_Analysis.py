# python/pdstools/app/decision_analyzer/pages/8_Offer_Quality_Analysis.py
import polars as pl
import streamlit as st

from pdstools.decision_analyzer.plots import getTrendChart, offer_quality_piecharts
from da_streamlit_utils import (
    contextual_filters,
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


propensity_th = _safe_thresholds(st.session_state.decision_data.get_thresholding_data("Propensity", [0, 10, 100]))
priority_th = _safe_thresholds(st.session_state.decision_data.get_thresholding_data("Priority", [0, 10, 100]))

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
        priority_th[0],  # Default to minimum (0)
        # step=(priority_th[2]-priority_th[0])/10,
        format="%.4f",
        help="Offers with priority below this value are considered irrelevant (low quality)",
    )
    contextual_filters()


# Apply channel filter to sample data
filtered_data = st.session_state.decision_data.filtered_sample

# Check for empty results when a specific channel is selected
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.len()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()

channel_filter = st.session_state.get("page_channel_expr")

action_counts = st.session_state.decision_data.filtered_action_counts(
    groupby_cols=[st.session_state.decision_data.level, "Interaction ID", "day"],
    priorityTH=priorityTH,
    propensityTH=propensityTH,
    additional_filters=channel_filter,
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
    )

with st.container(border=True):
    "## Offer Quality Over Time"

    # Default to the first stage with propensity scores for more meaningful analysis
    stages_with_prop = st.session_state.decision_data.stages_with_propensity
    default_stage = stages_with_prop[0] if stages_with_prop else None
    stage_selectbox(default=default_stage)

    vf = st.session_state.decision_data.get_offer_quality(action_counts, group_by=["Interaction ID", "day"])

    st.plotly_chart(
        getTrendChart(vf, stage=st.session_state.stage, level=st.session_state.decision_data.level),
    )

with st.container(border=True):
    "## Offer Variation"

    st.caption(
        "Actions ranked from most to least frequently selected in final decisions. "
        "The curve shows how many actions are needed to cover a given fraction of decisions — "
        "a steep curve means a few actions dominate, while a flatter curve indicates broader variety. "
        "Broken out by Channel/Direction to reveal whether concentration differs across channels."
    )

    # Offer Variation uses Output stage and is intentionally shown across all channels
    # to give a global view of action concentration, colored by Channel/Direction.
    st.plotly_chart(
        st.session_state.decision_data.plot.action_variation(
            stage="Output",
            color_by="Channel/Direction",
        ),
    )
    action_variability_stats = st.session_state.decision_data.get_offer_variability_stats("Output")
    st.caption(
        f"**{action_variability_stats['n90']}** actions account for 90% of final decisions. "
        f"The personalization index is **{round(action_variability_stats['gini'], 3)}** "
        f"(0 = all actions equally frequent, 1 = one action wins everything)."
    )
