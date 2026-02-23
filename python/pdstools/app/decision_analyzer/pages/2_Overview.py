# python/pdstools/app/decision_analyzer/pages/2_Overview.py
import streamlit as st
from da_streamlit_utils import ensure_data

# TODO see if we can speed up on first use - it's doing a lot now

ensure_data()
st.session_state["sidebar"] = st.sidebar

"# Overview"

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
    in total **{overview["Customers"]} different customers**.

    """

    "## :blue[Optionality Analysis]"

    """
    The number of actions available at arbitration vs the propensity to accept those. As
    there are more actions available, generally the success rates increase (and thus propensities).
    """
    st.plotly_chart(
        st.session_state.decision_data.plot.propensity_vs_optionality(
            "Arbitration"
        ).update_layout(showlegend=False, height=300),
        width="stretch",
    )

with col2:
    "## :orange[Influence of Prioritization Factors]"

    """
    Showing the percentage of decisions influenced by the various prioritization factors. In a
    more emphathetic, user centric, approach, the model propensities would be the major
    factor.
    """

    st.plotly_chart(
        st.session_state.decision_data.plot.sensitivity(
            win_rank=1,
            hide_priority=True,
        ).update_layout(
            height=300,
        ),
        width="stretch",
    )
