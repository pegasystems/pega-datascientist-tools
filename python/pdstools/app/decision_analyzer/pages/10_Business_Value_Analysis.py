import streamlit as st

st.warning(
    "In maintenance!!, please see: https://streamlit-dev.dsmcloud.io/Business%20Value%20Analysis for the older version. If the link doesn't work, contact Yusuf Uyanik."
)
# import polars as pl
# import streamlit as st

# from da_streamlit_utils import (
#     get_current_index,
#     st_value_distribution,
#     ensure_data,
# )
# from pdstools.decision_analyzer.utils import NBADScope_Mapping

# # TODO Finish up to show effect on proposition distribution (side to side)

# "# Business Value Analysis"

# """
# A closer look at the values associated with actions.

# * Is my value distribution very skewed? Are there actions with significantly different values than the others?
# * What's the range of the values?

# """
# ensure_data()
# st.warning(
#     "Current sample data action values are artificial so the analysis is just an example."
# )

# st.session_state["sidebar"] = st.sidebar

# scope_options = st.session_state.decision_data.getPossibleScopeValues()
# if "scope" not in st.session_state:
#     st.session_state.scope = scope_options[0]

# valueData = st.session_state.decision_data.getValueDistributionData()

# with st.container(border=True):
#     st.plotly_chart(
#         st_value_distribution(valueData, st.session_state.scope),
#         use_container_width=True,
#     )

#     scope_index = get_current_index(scope_options, "scope")
#     st.selectbox(
#         "Granularity:",
#         options=scope_options,
#         format_func=lambda option: NBADScope_Mapping[option],
#         index=scope_index,
#         key="scope",
#     )

# "Actions having different values:"

# st.dataframe(
#     valueData.filter(pl.col("Value_min") != pl.col("Value_max")).collect(),
#     hide_index=True,
#     column_config=NBADScope_Mapping,
# )
