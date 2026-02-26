# python/pdstools/app/decision_analyzer/pages/7_Personalization_Analysis.py
import polars as pl
import streamlit as st
from da_streamlit_utils import (
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)

# TODO cosmetics nicer color scheme for the stages - do consistently in all plots then
# TODO for optionality plot allow to overlay propensity but also maybe add priority?
# TODO think more about the "offer variation" or personalization index - I now calculated PI from the AUC of the variation curve

"# Personalization Analysis"

"""
Analysis of the number of actions per customer. Do we have enough options for people? Global filters can
be applied like everywhere to look at e.g. a certain group of issues.

As and when we have more different stages in the data, these analyses
will automatically support those.

The propensity pre-arbitration is not available in the actual Decision Analyser data as this is only calculated
in the Arbitration stage for the actions prioritized by AI.
"""
ensure_data()
st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()

"### Optionality"

with st.container(border=True):
    """
    Showing the number of actions available and the average of the highest
    propensity when there are this number of actions. Generally, with more actions
    you would expect higher propensities as there is more to choose from.
    """

    st.plotly_chart(
        st.session_state.decision_data.plot.propensity_vs_optionality(
            stage=st.session_state.get("optionality_stage", "Arbitration"),
            df=st.session_state.decision_data.sample,
        ),
        use_container_width=True,
    )

    stage_selectbox(key="optionality_stage", default="Arbitration")

if st.session_state.decision_data.extract_type != "explainability_extract":
    with st.container(border=True):
        "## Optionality Funnel"
        "Distribution of Available action by Stage"
        if st.session_state.decision_data.extract_type == "decision_analyzer":
            st.plotly_chart(
                st.session_state.decision_data.plot.optionality_funnel(df=st.session_state.decision_data.sample),
                use_container_width=True,
            )


"## Optionality Trend chart"

"""
Showing the number of unique actions over time - so you can spot significant
changes in the number of available actions
"""


optionality_data_with_trend_per_stage = (
    st.session_state.decision_data.get_optionality_data_with_trend(df=st.session_state.decision_data.sample)
    .group_by(["day", st.session_state.decision_data.level])
    .agg(nOffers=pl.col("nOffers").max())
    .sort("day")
)

fig, warning = st.session_state.decision_data.plot.optionality_trend(
    optionality_data_with_trend_per_stage,
)
if warning is not None:
    st.warning(warning)
st.plotly_chart(
    fig,
    use_container_width=True,
)

"## Offer variation"

"""
How much variation is there in the offers? Does everyone get the same few actions or
is there a lot of variation in what we are offering?
"""


st.plotly_chart(
    st.session_state.decision_data.plot.action_variation(stage="Output"),
    use_container_width=True,
)
action_variability_stats = st.session_state.decision_data.get_offer_variability_stats("Output")
f"""
{action_variability_stats["n90"]} actions win in 90% of the final decisions made. The personalization index is **{round(action_variability_stats["gini"], 3)}**.
"""
