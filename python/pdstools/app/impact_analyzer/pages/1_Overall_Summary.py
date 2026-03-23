# python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer · Overall Summary")

ia = ensure_impact_analyzer()

"# Overall Summary"

"""
View lift metrics aggregated across all channels. This page shows the overall
impact of your experiments, comparing treatment vs control groups.
"""

facet = "Channel" if "Channel" in ia.ia_data.collect_schema().names() else None

with st.container(border=True):
    "## Lift Overview"

    metric = st.selectbox(
        "Metric",
        options=["CTR_Lift", "Value_Lift"],
        index=0,
    )

    fig = ia.plot.overview(metric=metric, facet=facet)
    st.plotly_chart(fig)

with st.container(border=True):
    "## Detailed Metrics"

    st.caption("Tabular view of lift metrics with exact values. Use this for detailed analysis and reporting.")

    table = ia.plot.overview(metric=metric, facet=facet, return_df=True).collect()
    st.dataframe(table, width="stretch")
