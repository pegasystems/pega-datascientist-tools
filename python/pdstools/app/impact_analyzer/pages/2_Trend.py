from pathlib import Path
import sys

import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import (  # noqa: E402
    ensure_impact_analyzer,
)


repo_python_path = Path(__file__).resolve().parents[4]
if str(repo_python_path) not in sys.path:
    sys.path.insert(0, str(repo_python_path))

st.set_page_config(layout="wide", page_title="Impact Analyzer - Trend")

ensure_impact_analyzer()

ia = st.session_state["impact_analyzer"]

"""
# Trend

This page shows lift metrics over time.
"""

with st.expander("Display options", expanded=True):
    metric = st.selectbox(
        "Metric",
        options=["CTR_Lift", "Value_Lift"],
        index=0,
    )
    granularity = st.text_input(
        "Granularity",
        value="1d",
        help="Examples: 1d, 1w, 1mo, 2mo",
    ).strip()

facet = "Channel" if "Channel" in ia.ia_data.collect_schema().names() else None

fig = ia.plot.trend(metric=metric, facet=facet, every=granularity)
st.plotly_chart(fig, use_container_width=True)

table = ia.plot.trend(
    metric=metric, facet=facet, every=granularity, return_df=True
).collect()
st.dataframe(table)
