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
        Track how many offers **enter** each stage of your decisioning pipeline. The numbers show
        offers arriving at each stage (not exiting) — the narrowing funnel reveals where offers drop
        off, helping you spot bottlenecks or overly aggressive filtering.

        Use **Granularity** (sidebar) to analyze at different levels — from high-level offer categories
        (Issue/Group) down to individual actions or treatments.
        """)
        remanining_funnel, filtered_funnel = decision_funnel(
            scope=st.session_state.scope,
            level=st.session_state.decision_data.level,
        )
        st.plotly_chart(
            remanining_funnel,
            width="stretch",
        )

    with filtered_tab:
        st.plotly_chart(
            filtered_funnel,
            width="stretch",
        )

"""
## Filter Impact Details

See exactly which business rules are removing offers, at which stage, and how frequently.
This table shows all filter components ranked by impact, helping you identify rules that
may need adjustment.
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
    label="Download as CSV",
    file_name="filter_impact_analysis.csv",
    data=csv,
    help="Download the complete filter impact analysis as a CSV file",
)

# ---------------------------------------------------------------------------
# Component → Action Impact
# ---------------------------------------------------------------------------
has_components = "Component Name" in st.session_state.decision_data.decision_data.collect_schema().names()
if has_components:
    with st.container(border=True):
        "## Filter Impact by Offer"
        """
        Discover which offers are most affected by each business rule. Identify if critical
        offers are being blocked unintentionally and understand which filters have the
        strongest impact on your offer portfolio.
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
        st.plotly_chart(impact_fig, width="stretch")

    # ---------------------------------------------------------------------------
    # Component Drilldown
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        "## Deep Dive: Filter Details"
        """
        Investigate a specific filter to see all offers it removes. Each offer shows its average
        business value, priority, and propensity score — making it easy to spot if high-value
        offers are being filtered out. Sort by value to identify the most impactful removals.
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
            st.plotly_chart(drilldown_fig, width="stretch")

            # Also show the raw data table
            drilldown_df = st.session_state.decision_data.getComponentDrilldown(
                component_name=selected_component,
            )
            st.dataframe(drilldown_df)
        else:
            st.warning("No filter components found in the data.")
