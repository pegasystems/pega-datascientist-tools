# python/pdstools/app/impact_analyzer/pages/2_Trend.py
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.utils.cdh_utils import is_valid_polars_duration
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer · Trend")

ia = ensure_impact_analyzer()

"# Trend Analysis"

"""
Explore how lift metrics change over time. Use this page to identify trends,
seasonal patterns, and the stability of your experiment results.
"""


facet = "Channel" if "Channel" in ia.ia_data.collect_schema().names() else None

with st.container(border=True):
    "## Time Series View"

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox(
            "Metric",
            options=["CTR_Lift", "Value_Lift"],
            index=0,
        )
    with col2:
        granularity = (
            st.text_input(
                "Granularity",
                value="1d",
                help="Examples: 1d, 1w, 1mo, 2mo, 1y, 1h30m",
            )
            .strip()
            .lower()
        )

    if not is_valid_polars_duration(granularity):
        st.warning(
            "Invalid granularity. Use one or more positive integer + unit segments: "
            "`y`, `mo`, `q`, `w`, `d`, `h`, `m`, `s`, `ms`, `us`, or `ns` "
            "(for example `1d`, `1w`, `2mo`, `1h30m`)."
        )
        st.stop()

    fig = ia.plot.trend(metric=metric, facet=facet, every=granularity)
    st.plotly_chart(fig)

with st.container(border=True):
    "## Detailed Metrics"

    st.caption("Tabular view of trend data with exact values per time period.")

    table = ia.plot.trend(
        metric=metric,
        facet=facet,
        every=granularity,
        return_df=True,
    ).collect()
    st.dataframe(table, width="stretch")
