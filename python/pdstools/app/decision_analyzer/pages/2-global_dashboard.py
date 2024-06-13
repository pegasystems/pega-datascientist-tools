import streamlit as st
from da_streamlit_utils import ensure_data

# TODO see if we can speed up on first use - it's doing a lot now
# TODO add the last Pie of Value Finder, generalize the code from the Offer Quality Analysis and re-use

ensure_data()
st.session_state["sidebar"] = st.sidebar

"# Decision Insights"

"""
Quick insights into your decisioning implementation at a glance.
"""

col1, col2 = st.columns(2)

with col1:
    "## :green[Overview]"

    overview = st.session_state.decision_data.get_overview_stats

    f"""
    In total, there are **{overview["Actions"]} actions** available in **{overview["Channels"]} channels**. The data 
    was recorded over **{overview["Duration"].days} days** from **{overview["StartDate"]}** where **{overview["Decisions"]} decisions** 
    (**{round(overview["Decisions"]/overview["Duration"].days)}** decisions per day) were made for
    in total **{overview["Customers"]} different customers**. For each decision there was an 
    average of **{overview["avgOffersAtEligibility"]}** actions available in total and 
    an average of  **{overview["avgOffersAtArbitration"]}** before arbitration.

    """

    "## :blue[Optionality Analysis]"

    """
    The number of actions available at arbitration vs the propensity to accept those. As
    there are more actions available, generally the success rates increase (and thus propensities).
    """
    # st.write(st.session_state.decision_data.getOptionalityData().explain(optimized=False))
    st.plotly_chart(
        st.session_state.decision_data.plot.plot_propensity_vs_optionality(
            "Arbitration"
        ).update_layout(showlegend=False, height=300),
        use_container_width=True,
    )

    "## :violet[Personalization]"

    f"""
    The personalization index is **{round(st.session_state.decision_data.get_offer_variability_stats("Final")["gini"],3)}**.
    """
    st.plotly_chart(
        st.session_state.decision_data.plot.plot_action_variation(
            "Final"
        ).update_layout(width=300, height=300),
        use_container_width=True,
    )

with col2:
    "## :orange[Influence of Prioritization Factors]"

    """
    Showing the percentage of decisions influenced by the various prioritization factors. In a
    more emphathetic, user centric, approach, the model propensities would be the major
    factor.
    """

    st.plotly_chart(
        st.session_state.decision_data.plot.plot_sensitivity(
            win_rank=1,
            hide_priority=True,
        ).update_layout(
            height=300,
        ),
        use_container_width=True,
    )

    "## Quality of the Offers"

    """
    TODO - a summary of the value finder-like analysis at arbitration
    """

    # # TODO: this a bit too much code, see offer quality page and TODOs there to move into data class etc
    # # TODO: just one to show one (final) distribution here pretty much similar to Value Finder

    # default_propensity_th = [
    #     round(x, 4)
    #     for x in st.session_state.decision_data.getThresholdingData(
    #         "FinalPropensity", [0, 5, 100]
    #     )["Threshold"].to_list()
    # ]
    # default_priority_th = [
    #     round(x, 4)
    #     for x in st.session_state.decision_data.getThresholdingData(
    #         "Priority", [0, 5, 100]
    #     )["Threshold"].to_list()
    # ]
    # st.write(f"{default_propensity_th=}")
    # arbitration_action_counts = filtered_action_counts(
    #     df=st.session_state.decision_data.decision_data.filter(
    #         pl.col("pxEngagementStage").is_in(["Arbitration", "Final"])
    #     ),
    #     groupby_cols=["pxEngagementStage", "pxInteractionID"],
    #     priorityTH=default_propensity_th[1],
    #     propensityTH=default_priority_th[1],
    # )
    # st.dataframe(arbitration_action_counts.collect())
    # # Pie Chart
    # vf = st.session_state.decision_data.get_offer_quality(arbitration_action_counts, group_by="pxInteractionID")
    # st.dataframe(vf.head().collect())

    # st.plotly_chart(
    #     plot_offer_quality_piecharts(vf, propensityTH=default_propensity_th, st.session_state.decision_data.NBADStages_FilterView, st.session_data.decision_data.NBADStages_Mapping),
    #     use_container_width=True,
    # )
