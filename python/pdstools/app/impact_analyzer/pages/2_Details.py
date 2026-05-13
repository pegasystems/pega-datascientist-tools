# python/pdstools/app/impact_analyzer/pages/2_Details.py
"""Details page — flat per-experiment metrics table.

Minimal presentation of the per-experiment lift table, scoped by the
shared sidebar Channel/Direction filter (rendered by
``render_channel_filter`` so the selection persists across pages).

This page intentionally has no other visuals right now — the previous
per-channel charts (lift bar chart, response-rate heatmap, impression
redistribution) were removed pending a redesign. New visuals will be
re-added as separate plan items.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import (
    ensure_impact_analyzer,
    render_channel_filter,
)
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer · Details")

ia = ensure_impact_analyzer()

"# Details"

_channel_filter = render_channel_filter(ia)

# Source the table from the channel-aware aggregate when a specific
# channel is selected, otherwise from the cross-channel aggregate.
# Both paths yield one row per experiment.
if _channel_filter == "Any":
    _df = ia.summarize_experiments().collect()
    st.caption("All channels (aggregate)")
else:
    _df = ia.summarize_experiments(by="Channel").filter(pl.col("Channel") == _channel_filter).drop("Channel").collect()
    st.caption(f"Channel: **{_channel_filter}**")

if _df.is_empty():
    st.info("No experiment data available for this scope.")
    st.stop()

display_df = _df.select(
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
).sort("Experiment")

st.dataframe(display_df, hide_index=True)
