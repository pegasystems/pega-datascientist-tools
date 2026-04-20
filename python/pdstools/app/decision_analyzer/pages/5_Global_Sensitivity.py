import polars as pl
import streamlit as st
from da_streamlit_utils import contextual_filters, ensure_data, get_current_index

from pdstools.decision_analyzer.utils import apply_filter

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
        "Define winning: rank in top N",
        min_value=1,
        max_value=st.session_state.decision_data.max_win_rank,
        value=st.session_state.win_rank if "win_rank" in st.session_state else 1,
        key="win_rank",
        help="A win means at least one offer ranks N or better.",
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

total_decisions = apply_filter(filtered_data, channel_filter).select(pl.n_unique("Interaction ID")).collect().item()

with st.container(border=True):
    "## What Drives Your Offer Selection?"

    st.plotly_chart(
        st.session_state.decision_data.plot.sensitivity(
            st.session_state.win_rank,
            additional_filters=channel_filter,
            total_decisions=total_decisions,
        ),
    )

with st.container(border=True):
    "## Win/Loss Distribution"

    st.caption(
        "See which offers most often win or lose in the final selection. This reveals "
        "which offers dominate your customer interactions and which rarely make it through."
    )

    scope_options = st.session_state.decision_data.get_possible_scope_values()

    if "glob_sensitivity_scope" not in st.session_state:
        st.session_state.glob_sensitivity_scope = scope_options[0]

    st.plotly_chart(
        st.session_state.decision_data.plot.global_winloss_distribution(
            level=st.session_state.glob_sensitivity_scope,
            win_rank=st.session_state.win_rank,
            additional_filters=channel_filter,
        ),
    )

    scope_index = get_current_index(scope_options, "glob_sensitivity_scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="glob_sensitivity_scope",
    )
