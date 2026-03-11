# python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py
import polars as pl
import streamlit as st
from da_streamlit_utils import (
    channel_direction_selector,
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)

# TODO cosmetics nicer color scheme for the stages - do consistently in all plots then
# TODO for optionality plot allow to overlay propensity but also maybe add priority?
# TODO think more about the "offer variation" or personalization index - I now calculated PI from the AUC of the variation curve

"# Optionality Analysis"

"""
Analysis of the number of actions per customer. Do we have enough options for people? Global filters can
be applied like everywhere to look at e.g. a certain group of issues.
"""
ensure_data()
st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()

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

with st.container(border=True):
    "## Optionality"

    st.caption(
        "Showing the number of actions available and the average of the highest "
        "propensity when there are this number of actions. Generally, with more actions "
        "you would expect higher propensities as there is more to choose from."
    )

    stage_selectbox(
        key="optionality_stage",
        default="Arbitration",
    )

    st.plotly_chart(
        st.session_state.decision_data.plot.propensity_vs_optionality(
            stage=st.session_state.get("optionality_stage", "Arbitration"),
            df=filtered_data,
        ),
    )

    st.caption(
        "Propensity is only available from the Arbitration stage onward, "
        "as it is calculated only for the actions that AI prioritizes at that point."
    )

if st.session_state.decision_data.extract_type != "explainability_extract":
    with st.container(border=True):
        "## Optionality Funnel"
        "Distribution of Available action by Stage"
        if st.session_state.decision_data.extract_type == "decision_analyzer":
            st.plotly_chart(
                st.session_state.decision_data.plot.optionality_funnel(df=filtered_data),
            )


with st.container(border=True):
    "## Optionality Trend"

    st.caption(
        "Showing the number of unique actions over time - so you can spot significant "
        "changes in the number of available actions."
    )

    optionality_data_with_trend_per_stage = (
        st.session_state.decision_data.get_optionality_data_with_trend(df=filtered_data)
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
    )

with st.container(border=True):
    "## Offer Variation"

    st.caption(
        "How much variation is there in the offers? Does everyone get the same few actions or "
        "is there a lot of variation in what we are offering?"
    )

    # Offer Variation uses Output stage and is intentionally excluded from
    # channel filtering to show global variation.
    st.plotly_chart(
        st.session_state.decision_data.plot.action_variation(stage="Output"),
    )
    action_variability_stats = st.session_state.decision_data.get_offer_variability_stats("Output")
    st.caption(
        f"{action_variability_stats['n90']} actions win in 90% of the final decisions made. "
        f"The personalization index is **{round(action_variability_stats['gini'], 3)}**."
    )
