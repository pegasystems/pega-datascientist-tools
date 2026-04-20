# python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py
import polars as pl
import streamlit as st
from da_streamlit_utils import (
    contextual_filters,
    ensure_data,
    ensure_funnel,
    get_current_index,
    polars_lazyframe_hashing,
    stage_level_selector,
    stage_selectbox,
)

"# Action Funnel"

"""
Understand how your offers flow through the decisioning pipeline and where they drop off.
See which filtering rules are impacting your offer mix and identify opportunities to improve
customer reach.

**Key questions:**
* Which offers are being filtered out too early?
* Are high-value offers getting blocked by business rules?
* Where in the pipeline do specific offer categories drop off?
"""

ensure_data()
ensure_funnel()


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def decision_funnel(scope, level=None, channel_filter=None):
    return st.session_state.decision_data.plot.decision_funnel(
        scope=scope,
        additional_filters=channel_filter,
    )


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def decisions_without_actions_plot(level=None, channel_filter=None):
    return st.session_state.decision_data.plot.decisions_without_actions_plot(
        additional_filters=channel_filter,
    )


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def funnel_summary(scope, level=None, channel_filter=None):
    available_df, passing_df, filtered_df = st.session_state.decision_data.get_funnel_data(
        scope=scope,
        additional_filters=channel_filter,
    )
    return st.session_state.decision_data.get_funnel_summary(
        available_df, passing_df, additional_filters=channel_filter
    )


st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()

    scope_options = st.session_state.decision_data.get_possible_scope_values()

    if "scope" not in st.session_state:
        st.session_state.scope = scope_options[0]
    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        index=scope_index,
        key="scope",
    )
    contextual_filters()

filtered_data = st.session_state.decision_data.filtered_sample

if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.len()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()

channel_filter = st.session_state.get("page_channel_expr")

with st.container(border=True):
    passing_tab, filtered_tab, without_tab = st.tabs(
        ["Passing Actions", "Filtered Actions", "Decisions without Actions"]
    )

    passing_fig, filtered_fig = decision_funnel(
        scope=st.session_state.scope,
        level=st.session_state.decision_data.level,
        channel_filter=channel_filter,
    )

    with passing_tab:
        st.caption(
            "Actions passing **out of** each stage. The first column (**Available Actions**) is a synthetic "
            "entry baseline showing what goes **into** stage 1. Funnel height shows **Average Actions per Decision**; "
            "**Reach** shows the percentage of decisions with at least one offer."
        )
        st.plotly_chart(passing_fig)

    with filtered_tab:
        st.caption(
            "Actions **removed** at each stage. Large bars indicate stages with high filtering impact. "
            "This view includes all product stages in order."
        )
        st.plotly_chart(filtered_fig)

    with without_tab:
        st.caption(
            "Percentage of decisions that **newly lose all remaining actions** at each stage — i.e. "
            "the stage where a decision first becomes zero-action."
        )
        without_fig = decisions_without_actions_plot(
            level=st.session_state.decision_data.level,
            channel_filter=channel_filter,
        )
        st.plotly_chart(without_fig)

"""
## Filter Impact Details

For each stage, shows average actions per decision entering the stage, how many pass through,
how many are filtered out, and the percentage of decisions with at least one action.
"""

summary_df = funnel_summary(
    scope=st.session_state.scope,
    level=st.session_state.decision_data.level,
    channel_filter=channel_filter,
)
st.dataframe(summary_df)


# ---------------------------------------------------------------------------
# Component Analysis (replaces former "Filter Impact by Offer" + "Deep Dive")
# ---------------------------------------------------------------------------
has_components = "Component Name" in st.session_state.decision_data.decision_data.collect_schema().names()
if has_components:
    da = st.session_state.decision_data

    with st.container(border=True):
        "## Component Analysis"
        st.caption(
            "Drill into which filter components remove the most offers at a specific stage. "
            "Select a stage below, then explore individual components and the offers they filter."
        )

        col1, col2 = st.columns(2)
        with col1:
            stage_selectbox(
                label=f"{da.level}:",
                key="component_stage",
            )
        with col2:
            impact_top_n = st.number_input(
                "Top items per component:",
                min_value=1,
                max_value=30,
                value=5,
                key="impact_top_n",
            )

        # Build filters: combine channel filter with stage filter
        stage_filter = pl.col(da.level) == st.session_state.component_stage
        combined_filters = [f for f in [channel_filter, stage_filter] if f is not None]

        component_names = (
            da.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
            .filter(stage_filter)
            .select("Component Name")
            .unique()
            .collect()
            .get_column("Component Name")
            .sort()
            .to_list()
        )

        if not component_names:
            st.info("No filter components found at this stage.")
        else:
            sort_options_display = [
                "Filtered Decisions",
                "Average Value",
                "Average Priority",
                "Average Propensity",
            ]
            sort_options_mapping = {
                "Filtered Decisions": "Filtered Decisions",
                "Average Value": "avg_Value",
                "Average Priority": "avg_Priority",
                "Average Propensity": "avg_Propensity",
            }

            col3, col4 = st.columns(2)
            with col3:
                selected_component = st.selectbox(
                    "Drill into component:",
                    options=component_names,
                    key="drilldown_component",
                )
            with col4:
                sort_by_display = st.selectbox(
                    "Sort by:",
                    options=sort_options_display,
                    key="drilldown_sort",
                )
            sort_by = sort_options_mapping[sort_by_display]

            impact_fig = da.plot.component_action_impact(
                top_n=impact_top_n,
                scope=st.session_state.scope,
                additional_filters=combined_filters,
            )
            st.plotly_chart(impact_fig)

            drilldown_df = da.get_component_drilldown(
                component_name=selected_component,
                scope=st.session_state.scope,
                additional_filters=combined_filters,
                sort_by=sort_by,
            )
            display_df = drilldown_df.rename(
                {c: c.replace("avg_", "Average ") for c in drilldown_df.columns if c.startswith("avg_")}
            )
            st.dataframe(display_df)
