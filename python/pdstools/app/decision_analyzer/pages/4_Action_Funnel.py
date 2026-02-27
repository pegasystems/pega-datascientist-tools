# python/pdstools/app/decision_analyzer/pages/4_Action_Funnel.py
import io

import polars as pl
import streamlit as st
from da_streamlit_utils import (
    ensure_data,
    ensure_funnel,
    get_current_index,
    polars_lazyframe_hashing,
    stage_level_selector,
)

# TODO Have a toggle to look at it both from a Passing and Filtered perspective like in Dennis' designs
# TODO Later, the concept of stages should be generalized - there are many more and we should not be restrictive - can we pick up from the data?
# TODO coloring of the component filtering can be used better, now a little boring - but perhaps there will be a type, now only prop filters but there may be when rules as well
# TODO the coloring at Action level is way to busy - maybe limit to a top-N or so, probably something we need more often in general
# TODO be ready for the many stages - see Dennis' designs

"# Funnel Analysis"

"""
This gives a view of which actions are filtered out where in the
decision funnel, but also by what component.

This helps answering questions like: Where do my "cards offers" get dropped? What gets filtered in which stage?

"""

ensure_data()
ensure_funnel()


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def decision_funnel(
    scope,
    level=None,
    return_df=False,
):
    return st.session_state.decision_data.plot.decision_funnel(scope=scope, return_df=return_df)


st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()

    scope_options = st.session_state.decision_data.getPossibleScopeValues()

    if "scope" not in st.session_state:
        st.session_state.scope = scope_options[0]
    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Granularity:",
        options=scope_options,
        index=scope_index,
        key="scope",
    )


with st.container(border=True):
    remaining_tab, filtered_tab = st.tabs(["Remaining", "Filtered"])
    with remaining_tab:
        st.write("""
        The Funnel illustrates how many actions arrived to each Stage.

        The **Granularity** defines the breakdown of the chart. You can for example look at
        Groups, Issues, individual Actions or, if available, Treatments.
        """)
        remanining_funnel, filtered_funnel = decision_funnel(
            scope=st.session_state.scope,
            level=st.session_state.decision_data.level,
        )
        st.plotly_chart(
            remanining_funnel,
            use_container_width=True,
        )

    with filtered_tab:
        st.plotly_chart(
            filtered_funnel,
            use_container_width=True,
        )

"""
Decision Analyzer offers a unique perspective with all the details of what
action got dropped in which stage and by what component.

"""

data = (
    st.session_state.decision_data.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
    .group_by(["Stage Order", "Stage Group", "Stage", "Component Name"])
    .agg(pl.len().alias("filter count"))
    .with_columns(
        (
            pl.format(
                "{}%",
                ((pl.col("filter count") / pl.sum("filter count")) * 100).round(1),
            )
        ).alias("percent of all filters")
    )
    .collect()
    .sort("filter count", descending=True)
)
st.dataframe(data)


@st.cache_data
def convert_polars_df(df):
    buffer = io.StringIO()
    df.write_csv(buffer)
    buffer.seek(0)
    return buffer.getvalue().encode("utf-8")


csv = convert_polars_df(data)
st.download_button(
    label="Click to Download",
    file_name="file.csv",
    data=csv,
)

# ---------------------------------------------------------------------------
# Component → Action Impact
# ---------------------------------------------------------------------------
has_components = "Component Name" in st.session_state.decision_data.decision_data.collect_schema().names()
if has_components:
    with st.container(border=True):
        "## Component → Action Impact"
        """
        Which specific actions are most affected by each filter component?
        This shows the top actions that each component filters out.
        """
        impact_top_n = st.number_input(
            "Actions per component:",
            min_value=1,
            max_value=30,
            value=5,
            key="impact_top_n",
        )
        impact_fig = st.session_state.decision_data.plot.component_action_impact(
            top_n=impact_top_n,
            scope=st.session_state.scope,
        )
        st.plotly_chart(impact_fig, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Component Drilldown
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        "## Component Drilldown"
        """
        Select a filter component to see all actions it drops, enriched with
        scoring context (average Priority / Value / Propensity from surviving
        rows of the same action). Sort by value to find high-value actions
        being removed.
        """
        component_names = (
            st.session_state.decision_data.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
            .select("Component Name")
            .unique()
            .collect()
            .get_column("Component Name")
            .sort()
            .to_list()
        )
        if component_names:
            selected_component = st.selectbox(
                "Select component:",
                options=component_names,
                key="drilldown_component",
            )
            sort_options = ["Filtered Decisions", "avg_Value", "avg_Priority"]
            sort_by = st.selectbox(
                "Sort by:",
                options=sort_options,
                key="drilldown_sort",
            )
            drilldown_fig = st.session_state.decision_data.plot.component_drilldown(
                component_name=selected_component,
                sort_by=sort_by,
            )
            st.plotly_chart(drilldown_fig, use_container_width=True)

            # Also show the raw data table
            drilldown_df = st.session_state.decision_data.getComponentDrilldown(
                component_name=selected_component,
            )
            st.dataframe(drilldown_df)
        else:
            st.warning("No filter components found in the data.")
