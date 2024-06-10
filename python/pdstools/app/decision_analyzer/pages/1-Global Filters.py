import json

import polars as pl
import streamlit as st

from da_streamlit_utils import get_data_filters, show_filtered_counts, ensure_data
from pdstools.decision_analyzer.utils import get_first_level_stats

# TODO Customize the filter dropdown, instead of using the one from PDS tools, so to focus on the more logical filter fields
# TODO Audience selection would be a meaningful filter, but there is no such thing in the data NOTE: What is audience? A subset of customers?
# TODO Lets see if we can make re-applying the filters more easy, do that by default, including configurations like the reference set NOTE: I didn't understand this comment, what reference set?
# TODO Anonymization should perhaps be a check box?  Yusuf: You can now upload your own data. Sample data is the anonymized version
# TODO Summarize the current filtering even if not re-applying it. Add possibility to reset the filter in the Global Filter page.
# TODO Make rest of code robust against heavy filtering, e.g. dropping stages etc

"""# Global Filters"""

"""Here you can filter the decision data for analysis. You don't need to, if
you don't, then **all** data will be used.

In the drop down below, select the columns you wish to filter (e.g. Group, Issue, Channel etc).
For each column, a new configuration screen will be added in which you can specify
what values you want to use in the Decision Analyzer.

In the future, we may have sampling settings here as well.
"""

ensure_data()

## Filtering UI
expr_list = []
with st.container(border=True):
    """
    You can (optionally) save and re-apply filters you defined earlier:
    """
    uploaded_file = st.file_uploader(
        "Use the same data filters you used in previous sessions:", type=["json"]
    )
    if uploaded_file:
        imported_filters = json.load(uploaded_file)
        for key, val in imported_filters.items():
            expr_list.append(pl.Expr.from_json(json.dumps(val)))

st.session_state["filters"] = get_data_filters(
    st.session_state.decision_data.unfiltered_raw_decision_data,
    columns=st.session_state.decision_data.getAvailableFieldsForFiltering(),
    queries=expr_list,
    filter_type="global",
)

# Allow for optional save
# TODO: consider doing this by default
if st.session_state["filters"] != []:
    deserialize_exprs = {}
    for i, expr in enumerate(st.session_state["filters"]):
        deserialize_exprs[i] = json.loads(expr.meta.write_json())
    data = json.dumps(deserialize_exprs)
    st.download_button(
        label="Save current filters",
        data=data,
        file_name="DecisionAnalyzerFilters.json",
    )

if st.session_state["filters"] != []:
    st.session_state.decision_data.resetGlobalDataFilters()
    statsBeforeFilter = get_first_level_stats(st.session_state.decision_data.sample)

    st.session_state.decision_data.applyGlobalDataFilters(st.session_state["filters"])

    statsAfterFilter = get_first_level_stats(st.session_state.decision_data.sample)
    st.cache_data.clear()
    show_filtered_counts(statsBeforeFilter, statsAfterFilter)
else:
    st.session_state.decision_data.resetGlobalDataFilters()
    st.cache_data.clear()
    st.success("No data filters applied")
