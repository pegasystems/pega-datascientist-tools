# python/pdstools/app/impact_analyzer/pages/2_Click_Through_Rates.py
"""Click Through Rates page — flat per-experiment-per-channel metrics table.

Lists every experiment for every channel in one table (no aggregate
row, no sidebar filter). Pairs with a Control Fraction heatmap so
configuration mismatches across channels stay visible.

The previous per-channel charts (lift bar chart, response-rate heatmap,
impression redistribution) were removed pending a redesign. New visuals
will be re-added as separate plan items.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer · Click Through Rates")

ia = ensure_impact_analyzer()

"# Click Through Rates"

_has_channel = "Channel" in ia.ia_data.collect_schema().names()

_df = ia.summarize_experiments(by="Channel").collect() if _has_channel else ia.summarize_experiments().collect()

if _df.is_empty():
    st.info("No experiment data available.")
    st.stop()

_columns: list[pl.Expr | str] = []
if _has_channel:
    _columns.append("Channel")
_columns.extend(
    [
        "Experiment",
        pl.col("Impressions_Test").cast(pl.Int64).alias("Impressions (Test)"),
        pl.col("Impressions_Control").cast(pl.Int64).alias("Impressions (Control)"),
        pl.col("Accepts_Test").cast(pl.Int64).alias("Accepts (Test)"),
        pl.col("Accepts_Control").cast(pl.Int64).alias("Accepts (Control)"),
        (pl.col("CTR_Test") * 100).round(4).alias("CTR Test (%)"),
        (pl.col("CTR_Control") * 100).round(4).alias("CTR Control (%)"),
        (pl.col("CTR_Lift") * 100).round(2).alias("Engagement Lift (%)"),
        (pl.col("Value_Lift") * 100).round(2).alias("Value Lift (%)"),
        (pl.col("Control_Fraction") * 100).round(2).alias("Control Fraction (%)"),
    ]
)

display_df = _df.select(_columns)
if _has_channel:
    # Sort by Channel first; the Enum cast applied by
    # summarize_experiments preserves the canonical product order of
    # Experiment within each channel.
    display_df = display_df.sort(["Channel", "Experiment"])

st.markdown("### Counts by Channel and Experiment")
st.dataframe(display_df, hide_index=True)

# Control Fraction heatmap — Channel x Experiment. Built in the library
# (ia.plot.control_fraction_heatmap) so a notebook can reproduce it
# with one call.
if _has_channel:
    st.markdown("### Control Fraction by Channel")
    st.caption(
        "Each cell is the share of impressions assigned to the control "
        "group for that (Channel, Experiment) pair. Rows that vary across "
        "channels usually indicate a misconfigured control percentage in "
        "one of the channels."
    )
    fig = ia.plot.control_fraction_heatmap()
    st.plotly_chart(fig)

# CTR trend by control group, faceted by channel. One trace per control
# group (six in total for the default experiment set), so all experiments
# are visible in a single plot. Bogus control groups with no impressions
# in the data are dropped so the legend stays clean.
if "SnapshotTime" in ia.ia_data.collect_schema().names():
    _active_groups = (
        ia.summarize_control_groups()
        .filter(pl.col("Impressions").is_not_null() & (pl.col("Impressions") > 0))
        .select("ControlGroup")
        .collect()["ControlGroup"]
        .to_list()
    )
    if _active_groups:
        st.markdown("### CTR Trend by Control Group")
        st.caption(
            "Click-through rate over time for each control group, faceted "
            "by channel. Inactive control groups (no impressions in the "
            "data) are hidden."
        )
        fig = ia.plot.control_groups_trend(
            metric="CTR",
            facet="Channel" if _has_channel else None,
            every="1d",
            query=pl.col("ControlGroup").is_in(_active_groups),
        )
        st.plotly_chart(fig)
