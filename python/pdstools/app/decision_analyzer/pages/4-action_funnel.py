import streamlit as st

from da_streamlit_utils import (
    get_current_scope_index,
    get_data_filters,
    show_filtered_counts,
    ensure_data,
    ensure_getFilterComponentData,
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

This helps answering questions like: Where do my “cards offers” get dropped? Wat gets filtered in which stage?

The **Granularity** defines the breakdown of the chart. You can for example look at 
Groups, Issues, individual Actions or, if available, Treatments.
"""

ensure_data()
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

    # st.write(st.session_state["local_filters"])

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
    # st.write(st.session_state["local_filters"])

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


with st.container(border=True):
    if "scope" not in st.session_state:
        st.session_state.scope = scope_options[0]

    st.plotly_chart(
        st.session_state.decision_data.plot.plot_decision_funnel(
            scope=st.session_state.scope,
            NBADStages_Mapping=st.session_state.decision_data.NBADStages_Mapping,
            additional_filters=st.session_state["local_filters"],
        ),
        use_container_width=True,
    )
    scope_index = get_current_scope_index(scope_options)
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

if not ensure_getFilterComponentData():
    st.warning("'pxComponentName' column is needed for the component analysis")
    st.stop()

st.plotly_chart(
    st.session_state.decision_data.plot.plot_filtering_components(
        stages=stage_options,
        top_n=st.session_state.top_k,
        AvailableNBADStages=st.session_state.decision_data.AvailableNBADStages,
        additional_filters=st.session_state["local_filters"],
    ),
    use_container_width=True,
)
