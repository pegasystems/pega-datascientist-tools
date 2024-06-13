import streamlit as st
from da_streamlit_utils import get_current_index, ensure_data
from pdstools.decision_analyzer.utils import NBADScope_Mapping

# TODO The coloring at Action level is way to busy - maybe limit to a top-N or so, probably something we need more often in general
# TODO Infer the top-X by channel from the data (max rank per channel for Final records)

"# Global Sensitivity Analysis"

"""
Showing the *overall* effect of propensity, value, levers and context weights
on the decisions made.

Sensitivity is defined as the number of winning actions that change when
omitting just this one arbitration factor. The percentages indicates how 
many of the total (final) decisions would change. The percentages will 
usually not add up to 100%.
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


# How often would it still be rank 1 under different prioritization schemes
with st.container(border=True):
    st.plotly_chart(
        st.session_state.decision_data.plot.plot_sensitivity(
            st.session_state.win_rank,
        ),
        use_container_width=True,
    )

"""
Distribution of the actions that "win" in arbitration and the actions that "lose" in arbitration.
"""

with st.container(border=True):
    scope_options = st.session_state.decision_data.getPossibleScopeValues()

    if "glob_sensitivity_scope" not in st.session_state:
        st.session_state.glob_sensitivity_scope = scope_options[0]

    st.plotly_chart(
        st.session_state.decision_data.plot.plot_global_winloss_distribution(
            level=st.session_state.glob_sensitivity_scope,
            win_rank=st.session_state.win_rank,
        ),
        use_container_width=True,
    )

    scope_index = get_current_index(scope_options, "glob_sensitivity_scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        format_func=lambda option: NBADScope_Mapping[option],
        index=scope_index,
        key="glob_sensitivity_scope",
    )
