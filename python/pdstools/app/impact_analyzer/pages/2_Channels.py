# python/pdstools/app/impact_analyzer/pages/2_Channels.py
"""Channels page — per-channel pivot of lift metrics.

For a chosen experiment, breaks down engagement and value lift by channel.
Intended as the dimension-pivoted complement to the experiment-centric
Overall Summary page. Future work will generalise the pivot axis to
Issue / Group / Action (see
docs/plans/impact-analyzer/drill-zoom-into-issue-group-action.md).
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.impactanalyzer.statistics import (
    Z_95,
    accept_rate,
    binomial_se,
    lift_se,
    value_se,
)
from pdstools.utils.streamlit_utils import standard_page_config

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
standard_page_config(page_title="Impact Analyzer · Channels")

ia = ensure_impact_analyzer()

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------
_BLUE = "#1F6BFF"
_GREEN = "#2ca02c"
_RED = "#d62728"
_GREY = "#8795A1"
_TEXT = "#1B2A4A"
_BAND = "rgba(31,119,180,0.12)"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.metric-label { font-size: 11px; color: #8795A1; text-transform: uppercase;
                letter-spacing: 0.5px; }
.metric-value { font-size: 22px; font-weight: 700; color: #1B2A4A; }
.metric-value.positive { color: #2ca02c; }
.metric-value.negative { color: #d62728; }
.metric-value.neutral  { color: #8795A1; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.{digits}f}%"


def _lift_color(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return _GREY
    return _GREEN if v >= 0 else _RED


def _lift_class(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "neutral"
    return "positive" if v >= 0 else "negative"


# ═══════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════
"# Channels"

# ═══════════════════════════════════════════════════════════════════════════
# Experiment selector
# ═══════════════════════════════════════════════════════════════════════════
overall_df = ia.summarize_experiments().collect()
active_experiments = overall_df.filter(pl.col("Impressions_Test").is_not_null() & (pl.col("Impressions_Test") > 0))[
    "Experiment"
].to_list()

if not active_experiments:
    st.warning("No active experiments found in the data.")
    st.stop()

st.markdown("**Select experiment**")
selected_experiment = st.selectbox(
    "Select experiment",
    options=active_experiments,
    help="Choose an experiment to break down by channel.",
    label_visibility="collapsed",
)

# Show overall lift for context
overall_row = overall_df.filter(pl.col("Experiment") == selected_experiment)
if len(overall_row) > 0:
    row = overall_row.row(0, named=True)
    o_ctr = row.get("CTR_Lift")
    o_val = row.get("Value_Lift")
    st.markdown("##### Overall experiment lift (for reference)")
    c1, c2, c3 = st.columns(3)
    with c1:
        cls = _lift_class(o_ctr)
        st.markdown(
            f'<div class="metric-label">Engagement Lift</div><div class="metric-value {cls}">{_pct(o_ctr)}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        cls = _lift_class(o_val)
        st.markdown(
            f'<div class="metric-label">Value Lift</div><div class="metric-value {cls}">{_pct(o_val)}</div>',
            unsafe_allow_html=True,
        )
    with c3:
        cf = row.get("Control_Fraction")
        st.markdown(
            f'<div class="metric-label">Control Fraction</div><div class="metric-value">{_pct(cf)}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Per-channel data
# ═══════════════════════════════════════════════════════════════════════════
by_channel_df = (
    ia.summarize_experiments(by="Channel")
    .filter(pl.col("Experiment") == selected_experiment)
    .filter(pl.col("Impressions_Test").is_not_null() & (pl.col("Impressions_Test") > 0))
    .sort("Channel")
    .collect()
)

if len(by_channel_df) == 0:
    st.info("No per-channel data available for this experiment.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# Per-channel lift with CI
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Per-Channel Engagement Lift")

metric_toggle = st.radio(
    "Metric",
    ["Engagement Lift (CTR)", "Value Lift"],
    horizontal=True,
    key="drilldown_metric",
)
use_value = metric_toggle == "Value Lift"
lift_col = "Value_Lift" if use_value else "CTR_Lift"
metric_label = "Value Lift" if use_value else "Engagement Lift"

channels = by_channel_df["Channel"].to_list()
lifts = by_channel_df[lift_col].to_list()

# Compute SE for CI whiskers
ci_errors = []
for row in by_channel_df.iter_rows(named=True):
    n_t = int(row.get("Impressions_Test") or 0)
    a_t = int(row.get("Accepts_Test") or 0)
    n_c = int(row.get("Impressions_Control") or 0)
    a_c = int(row.get("Accepts_Control") or 0)
    if n_t > 0 and n_c > 0 and a_t > 0 and a_c > 0:
        if use_value:
            vpi_t = row.get("ValuePerImpression_Test") or 0
            vpi_c = row.get("ValuePerImpression_Control") or 0
            if vpi_t > 0 and vpi_c > 0:
                av_t = vpi_t * n_t / a_t
                av_c = vpi_c * n_c / a_c
                vse_t = value_se(a_t, n_t, av_t)
                vse_c = value_se(a_c, n_c, av_c)
                se = math.sqrt(vse_t**2 + vse_c**2) / vpi_c
            else:
                # PDC: VPI not available, approximate from engagement SE
                p_t = accept_rate(a_t, n_t)
                p_c = accept_rate(a_c, n_c)
                se = lift_se(p_t, p_c, binomial_se(a_t, n_t), binomial_se(a_c, n_c))
            ci_errors.append(Z_95 * se * 100)
        else:
            p_t = accept_rate(a_t, n_t)
            p_c = accept_rate(a_c, n_c)
            se_t = binomial_se(a_t, n_t)
            se_c = binomial_se(a_c, n_c)
            se = lift_se(p_t, p_c, se_t, se_c)
            ci_errors.append(Z_95 * se * 100)
    else:
        ci_errors.append(0)

lift_pcts = [(v * 100 if v is not None and not math.isnan(v) else 0) for v in lifts]
colors = [_lift_color(v) for v in lifts]

fig = go.Figure()

# Horizontal bar chart with error bars
fig.add_trace(
    go.Bar(
        y=channels,
        x=lift_pcts,
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            array=ci_errors,
            visible=True,
            color=_GREY,
            thickness=1.5,
        ),
        text=[f"{v:+.2f}%" for v in lift_pcts],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    )
)

# Reference line at zero
fig.add_vline(x=0, line_dash="dash", line_color=_GREY, line_width=1)

fig.update_layout(
    title=f"{metric_label} by Channel — {selected_experiment}",
    xaxis_title=f"{metric_label} (%)",
    yaxis_title="",
    height=max(300, len(channels) * 50 + 100),
    margin=dict(l=20, r=40, t=50, b=40),
    showlegend=False,
    plot_bgcolor="white",
    yaxis=dict(autorange="reversed"),
)
fig.update_xaxes(zeroline=True, zerolinecolor=_GREY, gridcolor="#f0f0f0")
fig.update_yaxes(gridcolor="#f0f0f0")

st.plotly_chart(fig)

# ═══════════════════════════════════════════════════════════════════════════
# Heatmap — CTR Test vs Control by channel
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Response Rate Comparison")

ctr_test = by_channel_df["CTR_Test"].to_list()
ctr_ctrl = by_channel_df["CTR_Control"].to_list()

heatmap_z = [
    [(v * 100 if v is not None and not math.isnan(v) else 0) for v in ctr_ctrl],
    [(v * 100 if v is not None and not math.isnan(v) else 0) for v in ctr_test],
]

fig_hm = go.Figure(
    data=go.Heatmap(
        z=heatmap_z,
        x=channels,
        y=["Control", "Test"],
        colorscale="Blues",
        text=[
            [f"{v:.4f}%" for v in heatmap_z[0]],
            [f"{v:.4f}%" for v in heatmap_z[1]],
        ],
        texttemplate="%{text}",
        hovertemplate="Channel: %{x}<br>Group: %{y}<br>CTR: %{z:.4f}%<extra></extra>",
        colorbar=dict(title="CTR %"),
    )
)
fig_hm.update_layout(
    title="Click-Through Rate (%) — Test vs Control by Channel",
    height=250,
    margin=dict(l=20, r=20, t=50, b=40),
    xaxis_title="Channel",
)

st.plotly_chart(fig_hm)

# ═══════════════════════════════════════════════════════════════════════════
# Impression redistribution analysis
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Impression Redistribution")
st.caption(
    "How NBA reallocates impressions across channels compared to the control group. "
    "Large shifts indicate the strategy is actively steering traffic."
)

# Compute impression shares
total_test = by_channel_df["Impressions_Test"].sum()
total_ctrl = by_channel_df["Impressions_Control"].sum()

redist_rows = []
for row in by_channel_df.iter_rows(named=True):
    ch = row["Channel"]
    n_t = int(row.get("Impressions_Test") or 0)
    n_c = int(row.get("Impressions_Control") or 0)
    share_t = n_t / total_test if total_test > 0 else 0
    share_c = n_c / total_ctrl if total_ctrl > 0 else 0
    shift = share_t - share_c
    redist_rows.append(
        {
            "channel": ch,
            "share_t": share_t,
            "share_c": share_c,
            "shift": shift,
            "n_t": n_t,
            "n_c": n_c,
        }
    )

# Grouped bar chart — impression share Test vs Control
fig_redist = go.Figure()
fig_redist.add_trace(
    go.Bar(
        y=[r["channel"] for r in redist_rows],
        x=[r["share_t"] * 100 for r in redist_rows],
        orientation="h",
        name="Test (NBA)",
        marker_color=_BLUE,
        text=[f"{r['share_t'] * 100:.1f}%" for r in redist_rows],
        textposition="auto",
    )
)
fig_redist.add_trace(
    go.Bar(
        y=[r["channel"] for r in redist_rows],
        x=[r["share_c"] * 100 for r in redist_rows],
        orientation="h",
        name="Control",
        marker_color=_GREY,
        text=[f"{r['share_c'] * 100:.1f}%" for r in redist_rows],
        textposition="auto",
    )
)
fig_redist.update_layout(
    barmode="group",
    title="Impression Share (%) — Test vs Control",
    xaxis_title="Share of total impressions (%)",
    height=max(300, len(channels) * 55 + 100),
    margin=dict(l=20, r=20, t=50, b=40),
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.1),
    yaxis=dict(autorange="reversed"),
)
fig_redist.update_xaxes(gridcolor="#f0f0f0")
st.plotly_chart(fig_redist)

# Shift summary
gaining = [r for r in redist_rows if r["shift"] > 0.005]
losing = [r for r in redist_rows if r["shift"] < -0.005]
if gaining or losing:
    col_g, col_l = st.columns(2)
    with col_g:
        if gaining:
            st.markdown("**NBA sends MORE traffic to:**")
            for r in sorted(gaining, key=lambda x: -x["shift"]):
                st.markdown(
                    f"- **{r['channel']}**: +{r['shift'] * 100:.1f}pp "
                    f"({r['share_c'] * 100:.1f}% → {r['share_t'] * 100:.1f}%)"
                )
        else:
            st.info("No channels gained significant share.")
    with col_l:
        if losing:
            st.markdown("**NBA sends LESS traffic to:**")
            for r in sorted(losing, key=lambda x: x["shift"]):
                st.markdown(
                    f"- **{r['channel']}**: {r['shift'] * 100:.1f}pp "
                    f"({r['share_c'] * 100:.1f}% → {r['share_t'] * 100:.1f}%)"
                )
        else:
            st.info("No channels lost significant share.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Detailed table
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Detailed Per-Channel Metrics")

display_df = by_channel_df.select(
    "Channel",
    pl.col("Impressions_Test").cast(pl.Int64).alias("Impressions (Test)"),
    pl.col("Impressions_Control").cast(pl.Int64).alias("Impressions (Control)"),
    pl.col("Accepts_Test").cast(pl.Int64).alias("Accepts (Test)"),
    pl.col("Accepts_Control").cast(pl.Int64).alias("Accepts (Control)"),
    (pl.col("CTR_Test") * 100).round(4).alias("CTR Test (%)"),
    (pl.col("CTR_Control") * 100).round(4).alias("CTR Control (%)"),
    (pl.col("CTR_Lift") * 100).round(2).alias("Engagement Lift (%)"),
    (pl.col("Value_Lift") * 100).round(2).alias("Value Lift (%)"),
    (pl.col("Control_Fraction") * 100).round(2).alias("Control Fraction (%)"),
)

st.dataframe(display_df, hide_index=True)
