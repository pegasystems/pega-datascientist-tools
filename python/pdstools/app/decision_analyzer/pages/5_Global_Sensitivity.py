import streamlit as st
from da_streamlit_utils import ensure_data, get_current_index

# TODO The coloring at Action level is way to busy - maybe limit to a top-N or so, probably something we need more often in general
# TODO Infer the top-X by channel from the data (max rank per channel for Final records)

"# Global Sensitivity Analysis"

"""
Understand which factors have the biggest impact on which offers win. This shows
how much each factor (customer response likelihood, business value, strategic levers)
influences your final offer selections.

**How to read this:** Each factor shows what percentage of winning offers would change
if that factor were removed. Higher percentages mean that factor is more influential in
driving offer selection.
"""

ensure_data()
st.session_state["sidebar"] = st.sidebar

with st.session_state["sidebar"]:
    st.number_input(
        "Top-N actions that define Winning",
        min_value=1,
        max_value=st.session_state.decision_data.max_win_rank,  # TODO why restrict to 10, lets use the upper bound from the data. Calculating these columns is expensive. Maybe add an option at the filter or homepage to increase that range
        value=st.session_state.win_rank if "win_rank" in st.session_state else 1,
        key="win_rank",
    )

with st.container(border=True):
    "## What Drives Your Offer Selection?"

    st.plotly_chart(
        st.session_state.decision_data.plot.sensitivity(
            st.session_state.win_rank,
        ),
        width="stretch",
    )

with st.container(border=True):
    "## Win/Loss Distribution"

    """
    See which offers most often win or lose in the final selection. This reveals
    which offers dominate your customer interactions and which rarely make it through.
    """

    scope_options = st.session_state.decision_data.getPossibleScopeValues()

    if "glob_sensitivity_scope" not in st.session_state:
        st.session_state.glob_sensitivity_scope = scope_options[0]

    st.plotly_chart(
        st.session_state.decision_data.plot.global_winloss_distribution(
            level=st.session_state.glob_sensitivity_scope,
            win_rank=st.session_state.win_rank,
        ),
        width="stretch",
    )

    scope_index = get_current_index(scope_options, "glob_sensitivity_scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="glob_sensitivity_scope",
    )
