from __future__ import annotations

import polars as pl
import streamlit as st

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    ANALYSIS_GRANULARITY_LABEL,
    collect_page_filters,
    contextual_filters,
    ensure_data,
    show_action_color_notice,
    show_prioritization_explanation,
    win_rank_selector,
)
from pdstools.decision_analyzer.utils import apply_filter
from pdstools.utils.streamlit_utils import get_current_index, standard_page_config

standard_page_config(page_title="Global Sensitivity · Decision Analysis")

"# Global Sensitivity Analysis"

"""
Understand which factors have the biggest impact on which offers win. This shows
how much each factor (customer response likelihood, business value, strategic levers)
influences your final offer selections.
"""

st.info(
    "- What percentage of winning offers would change if each factor were removed?\n"
    "- Which factors have the greatest influence on offer selection?",
    title="Key questions",
)

ensure_data()
st.session_state["sidebar"] = st.sidebar

with st.session_state["sidebar"]:
    win_rank_selector()
    contextual_filters()

# Apply channel filter to sample data
filtered_data = st.session_state.decision_data.filtered(collect_page_filters())

# Check for empty results when a specific channel is selected
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.len()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        show_prioritization_explanation()
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

    mandatory_actions = st.session_state.decision_data.mandatory_actions
    if mandatory_actions:
        st.info(
            f"{len(mandatory_actions)} action(s) are auto-detected as mandatory "
            f"(priority ≥ 5M); they bypass arbitration and always rank top. "
            "These are marked with ★ in the chart below."
        )

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
        ANALYSIS_GRANULARITY_LABEL,
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="glob_sensitivity_scope",
    )
    show_action_color_notice(st.session_state.glob_sensitivity_scope)

show_prioritization_explanation()
