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

with st.container(border=True):
    "## Display Options"

    st.caption(
        "Select which metric to visualize. CTR Lift shows the relative improvement "
        "in click-through rates, while Value Lift shows the impact on business value."
    )

    metric = st.selectbox(
        "Metric",
        options=["CTR_Lift", "Value_Lift"],
        index=0,
    )

facet = "Channel" if "Channel" in ia.ia_data.collect_schema().names() else None

with st.container(border=True):
    "## Lift Overview"

    st.caption("Interactive chart showing lift metrics. Hover for details, click and drag to zoom.")

    fig = ia.plot.overview(metric=metric, facet=facet)
    st.plotly_chart(fig, width="stretch")

with st.container(border=True):
    "## Detailed Metrics"

    st.caption("Tabular view of lift metrics with exact values. Use this for detailed analysis and reporting.")

    table = ia.plot.overview(metric=metric, facet=facet, return_df=True).collect()
    st.dataframe(table, width="stretch")
