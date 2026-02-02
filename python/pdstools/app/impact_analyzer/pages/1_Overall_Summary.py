from pathlib import Path
import sys

import streamlit as st


repo_python_path = Path(__file__).resolve().parents[4]
if str(repo_python_path) not in sys.path:
    sys.path.insert(0, str(repo_python_path))

from pdstools.app.impact_analyzer.ia_streamlit_utils import (  # noqa: E402
    ensure_impact_analyzer,
)


st.set_page_config(layout="wide", page_title="Impact Analyzer - Overall Summary")

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
st.plotly_chart(fig, use_container_width=True)

table = ia.plot.overview(metric=metric, facet=facet, return_df=True).collect()
st.dataframe(table)
