# python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer - Overall Summary")

ensure_impact_analyzer()

ia = st.session_state["impact_analyzer"]

"""
# Overall Summary

This page shows lift metrics aggregated across all channels.
"""

with st.expander("Display options", expanded=True):
    metric = st.selectbox(
        "Metric",
        options=["CTR_Lift", "Value_Lift"],
        index=0,
    )

facet = "Channel" if "Channel" in ia.ia_data.collect_schema().names() else None

fig = ia.plot.overview(metric=metric, facet=facet)
st.plotly_chart(fig, width="stretch")

table = ia.plot.overview(metric=metric, facet=facet, return_df=True).collect()
st.dataframe(table)
