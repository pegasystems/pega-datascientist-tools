# python/pdstools/app/impact_analyzer/pages/2_Trend.py
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer · Trend")

ia = ensure_impact_analyzer()

"# Trend Analysis"

"""
Explore how lift metrics change over time. Use this page to identify trends,
seasonal patterns, and the stability of your experiment results.
"""

with st.container(border=True):
    "## Display Options"

    st.caption(
        "Choose which lift metric to analyze and the time granularity. "
        "The chart will update to show trends for your selection."
    )

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

with st.container(border=True):
    "## Time Series View"

    st.caption(
        "Trend chart showing how your selected metric evolves over time. "
        "Look for consistent patterns and significant changes."
    )

    fig = ia.plot.trend(metric=metric, facet=facet, every=granularity)
    st.plotly_chart(fig, width="stretch")

with st.container(border=True):
    "## Detailed Metrics"

    st.caption("Tabular view of trend data with exact values per time period.")

    table = ia.plot.trend(
        metric=metric,
        facet=facet,
        every=granularity,
        return_df=True,
    ).collect()
    st.dataframe(table, use_container_width=True)
