import io

import polars as pl
import streamlit as st
from da_streamlit_utils import (
    ensure_data,
    ensure_funnel,
    get_current_index,
    get_data_filters,
    show_filtered_counts,
    polars_lazyframe_hashing,
)

from pdstools.decision_analyzer.utils import (
    NBADScope_Mapping,
    get_first_level_stats,
)

# TODO Have a toggle to look at it both from a Passing and Filtered perspective like in Dennis' designs
# TODO Later, the concept of stages should be generalized - there are many more and we should not be restrictive - can we pick up from the data?
# TODO coloring of the component filtering can be used better, now a little boring - but perhaps there will be a type, now only prop filters but there may be when rules as well
# TODO the control for the number of filter components should move to that section, not be in the side bar
# TODO not sure about the optional additional filtering - what is the exact use case again?
# TODO the coloring at Action level is way to busy - maybe limit to a top-N or so, probably something we need more often in general
# TODO be ready for the many stages - see Dennis' designs

"# Funnel Analysis"

"""
This gives a view of which actions are filtered out where in the
decision funnel, but also by what component.

This helps answering questions like: Where do my “cards offers” get dropped? What gets filtered in which stage?

"""

ensure_data()
ensure_funnel()


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def decision_funnel(
    scope,
    additional_filters,
    return_df=False,
):
    return st.session_state.decision_data.plot.decision_funnel(
        scope=scope, additional_filters=additional_filters, return_df=return_df
    )


st.session_state["sidebar"] = st.sidebar
if "local_filters" in st.session_state:
    del st.session_state["local_filters"]
with st.session_state["sidebar"]:
    scope_options = st.session_state.decision_data.getPossibleScopeValues()
    stage_options = st.session_state.decision_data.getPossibleStageValues()

    # TODO bring this to the chart? limiting long lists is useful across many pages
    st.session_state.top_k = st.number_input(
        "How many filter components to show:",
        min_value=1,
        max_value=30,
        value=10,
        help="Maximum number of rows in box plots",
    )

    ## Filtering UI

    "### Optional additional filtering"

    # TODO always save state and when returning fill in the shown options in the dropdown
    # TODO also lets sort the columns from the df in some reasonable way - context keys first etc
    # TODO probably need to take over this "filter_dataframe" which also shows the
    # TODO dropdown with "Filter dataframe on"
    st.session_state["local_filters"] = get_data_filters(
        st.session_state.decision_data.decision_data,
        columns=st.session_state.decision_data.getAvailableFieldsForFiltering(
            categoricalOnly=True
        ),
        queries=[],
        filter_type="local",
    )

    if st.session_state["local_filters"] != []:
        statsBeforeExtraFilter = get_first_level_stats(
            st.session_state.decision_data.decision_data
        )
        statsAfterExtraFilter = get_first_level_stats(
            st.session_state.decision_data.decision_data,
            st.session_state["local_filters"],
        )
        show_filtered_counts(statsBeforeExtraFilter, statsAfterExtraFilter)
    else:
        st.success("No extra data filters applied")

    # st.session_state.decision_data.level = st.multiselect(
    #     "Stage Granularity",
    #     ["StageGroup", "Stage"],
    #     default="StageGroup"
    # )


with st.container(border=True):
    remaining_tab, filtered_tab = st.tabs(["Remaining", "Filtered"])
    with remaining_tab:
        st.write("""
        The Funnel illustrates how many actions arrived to each Stage.

        The **Granularity** defines the breakdown of the chart. You can for example look at
        Groups, Issues, individual Actions or, if available, Treatments.
        """)
        if "scope" not in st.session_state:
            st.session_state.scope = scope_options[0]
        remanining_funnel, filtered_funnel = decision_funnel(
            scope=st.session_state.scope,
            additional_filters=st.session_state["local_filters"],
        )
        st.plotly_chart(
            remanining_funnel,
            use_container_width=True,
        )
        scope_index = get_current_index(scope_options, "scope")

    with filtered_tab:
        st.plotly_chart(
            filtered_funnel,
            use_container_width=True,
        )
    st.selectbox(
        "Granularity:",
        options=scope_options,
        format_func=lambda option: NBADScope_Mapping[option],
        index=scope_index,
        key="scope",
    )

"""
Decision Analyzer offers a unique perspective with all the details of what
action got dropped in which stage and by what component.

"""

data = st.session_state.decision_data.decision_data.filter(
    pl.col("pxRecordType") == "FILTERED_OUT"
)
if st.session_state["local_filters"] != []:
    data.filter(st.session_state["local_filters"])
data = (
    data.group_by(["StageOrder", "StageGroup", "Stage", "pxComponentName"])
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
