# python/pdstools/app/impact_analyzer/pages/4_Drill_Down.py
"""Drill Down page — per-channel lift analysis with cannibalization disclaimers.

Allows users to drill into individual channels within an experiment to see
per-channel engagement and value lift. Includes prominent warnings about
**lift cannibalization**: per-channel lift can be negative even when overall
lift is positive, because NBA redistributes impressions across channels.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.impactanalyzer.statistics import (
    FORMULAS,
    Z_95,
    accept_rate,
    binomial_se,
    calculate_engagement_lift,
    calculate_lift,
    is_significant,
    lift_se,
    value_se,
)
from pdstools.utils.streamlit_utils import standard_page_config

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
standard_page_config(page_title="Impact Analyzer · Drill Down")

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
.cannib-warning {
    background: #FFF3CD; border-left: 5px solid #FFC107; padding: 16px 20px;
    border-radius: 6px; margin-bottom: 20px; font-size: 14px; line-height: 1.6;
    color: #664D03;
}
.cannib-warning strong { color: #664D03; }
.metric-label { font-size: 11px; color: #8795A1; text-transform: uppercase;
                letter-spacing: 0.5px; }
.metric-value { font-size: 22px; font-weight: 700; color: #1B2A4A; }
.metric-value.positive { color: #2ca02c; }
.metric-value.negative { color: #d62728; }
.metric-value.neutral  { color: #8795A1; }
.formula-box {
    background: #F0F4FA; border-left: 4px solid #1F6BFF; padding: 12px 16px;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
    line-height: 1.7; border-radius: 4px; margin: 8px 0;
    white-space: pre-wrap; color: #1B2A4A;
}
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


def _num(v, digits: int = 10) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _formula_box(lines: list[str] | str) -> None:
    if isinstance(lines, str):
        lines = [lines]
    st.markdown(
        f'<div class="formula-box">{chr(10).join(lines)}</div>',
        unsafe_allow_html=True,
    )


def _formula(latex: str, description: str = "", substitution: str = "") -> None:
    if description:
        st.markdown(
            f"<p style='color:#5A6B80;font-size:13px;margin-bottom:2px'>{description}</p>",
            unsafe_allow_html=True,
        )
    st.latex(latex)
    if substitution:
        st.markdown(
            f'<div class="formula-box" style="margin-top:-8px">{substitution}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Title & disclaimer
# ═══════════════════════════════════════════════════════════════════════════
"# Drill Down"

st.markdown(
    """
<div class="cannib-warning">
<strong>Important: Lift Cannibalization</strong><br>
When you drill down into individual channels within an experiment, the per-channel
lift numbers can be <strong>misleading</strong>. It is common for one or more channels
to show <strong>negative lift</strong> even when the <strong>overall experiment lift
is positive</strong>.<br><br>
This happens because NBA <em>redistributes</em> impressions: it shifts traffic toward
the best-performing channels and away from weaker ones. The control group, by contrast,
distributes randomly. So a channel that receives fewer NBA impressions may show a lower
click-through rate — not because the model is broken, but because the best customers
for that channel were <em>deliberately</em> redirected to a higher-value channel.<br><br>
<strong>Always evaluate lift at the overall experiment level first.</strong>
The per-channel view here is for diagnostic exploration only.
</div>
""",
    unsafe_allow_html=True,
)

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
    help="Choose an experiment to drill down into by channel.",
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
# Forest plot — per-channel lift with CI
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
# Channel performance details cards
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Channel Performance Details")
st.caption("Expand any channel below to see full formula walkthroughs, just like the Overall Summary page.")

for row in by_channel_df.iter_rows(named=True):
    ch = row["Channel"]
    n_t = int(row.get("Impressions_Test") or 0)
    a_t = int(row.get("Accepts_Test") or 0)
    n_c = int(row.get("Impressions_Control") or 0)
    a_c = int(row.get("Accepts_Control") or 0)
    ctr_lift = row.get("CTR_Lift")
    val_lift = row.get("Value_Lift")

    lift_str = _pct(ctr_lift)
    cls = _lift_class(ctr_lift)
    sig_icon = "+" if ctr_lift is not None and abs(ctr_lift) > 0 else "-"

    with st.expander(
        f"**{ch}** — Eng. Lift: {lift_str}  |  Test: {n_t:,}  |  Ctrl: {n_c:,}",
        expanded=False,
    ):
        if n_t == 0 or n_c == 0:
            st.warning("Insufficient data for this channel.")
            continue

        p_t = accept_rate(a_t, n_t)
        p_c = accept_rate(a_c, n_c)
        se_t = binomial_se(a_t, n_t)
        se_c = binomial_se(a_c, n_c)

        eng = calculate_engagement_lift(a_t, n_t, a_c, n_c)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(
            f'<div class="metric-label">ENG. LIFT</div>'
            f'<div class="metric-value {_lift_class(eng.lift)}">{_pct(eng.lift, 4)}</div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="metric-label">VALUE LIFT</div>'
            f'<div class="metric-value {_lift_class(val_lift)}">{_pct(val_lift, 4)}</div>',
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="metric-label">TEST n</div><div class="metric-value">{n_t:,}</div>',
            unsafe_allow_html=True,
        )
        m4.markdown(
            f'<div class="metric-label">CTRL n</div><div class="metric-value">{n_c:,}</div>',
            unsafe_allow_html=True,
        )

        # Significance
        if eng.significant:
            st.success(f"Statistically significant at 95% confidence for {ch}")
        else:
            st.warning(f"Not yet significant for {ch} — more data needed")

        # Arm comparison
        st.table(
            {
                "": ["Impressions", "Accepts", "Accept Rate"],
                "Test": [f"{n_t:,}", f"{a_t:,}", _pct(p_t, 6)],
                "Control": [f"{n_c:,}", f"{a_c:,}", _pct(p_c, 6)],
            }
        )

        # Formula walkthrough
        st.markdown("##### Step 1 — Accept Rates")
        _formula(
            FORMULAS["accept_rate"].latex,
            "Accept rate for each arm",
            f"p_test = {a_t} / {n_t} = {_num(p_t)}\np_ctrl = {a_c} / {n_c} = {_num(p_c)}",
        )

        st.markdown("##### Step 2 — Standard Errors")
        _formula(
            FORMULAS["binomial_se"].latex,
            "Binomial standard error",
            f"SE_test = {_num(se_t)}\nSE_ctrl = {_num(se_c)}",
        )

        st.markdown("##### Step 3 — Engagement Lift")
        _formula(
            FORMULAS["lift"].latex,
            "",
            f"Lift = ({_num(p_t)} - {_num(p_c)}) / {_num(p_c)} = {_num(eng.lift)}"
            if eng.lift is not None
            else "Lift = N/A (control rate is 0)",
        )

        st.markdown("##### Step 4 — Confidence Interval (Delta Method)")
        _formula(
            FORMULAS["lift_se"].latex,
            "",
            f"SE(Lift) = {_num(eng.se)}" if eng.se is not None else "SE(Lift) = N/A",
        )
        if eng.lift is not None and eng.se is not None:
            lo = eng.lift - Z_95 * eng.se
            hi = eng.lift + Z_95 * eng.se
            sig_txt = "YES — CI does not contain 0" if eng.significant else "NO — CI contains 0"
            _formula_box(
                [
                    f"95% CI = [{lo:.10f},  {hi:.10f}]",
                    f"Statistically significant?  {sig_txt}",
                ]
            )

        # Value lift details
        vpi_t = row.get("ValuePerImpression_Test")
        vpi_c = row.get("ValuePerImpression_Control")
        if vpi_t and vpi_c and vpi_t > 0 and vpi_c > 0 and p_t > 0 and p_c > 0:
            st.markdown("##### Step 5 — Value Lift")
            av_t = vpi_t / p_t
            av_c = vpi_c / p_c
            _formula(
                FORMULAS["vpi"].latex,
                "Value Per Impression",
                f"AV_test = {_num(av_t, 4)}  |  AV_ctrl = {_num(av_c, 4)}\n"
                f"VPI_test = {_num(p_t)} × {_num(av_t, 4)} = {_num(vpi_t)}\n"
                f"VPI_ctrl = {_num(p_c)} × {_num(av_c, 4)} = {_num(vpi_c)}",
            )
            v_lift = calculate_lift(vpi_t, vpi_c)
            vse_t = value_se(a_t, n_t, av_t)
            vse_c = value_se(a_c, n_c, av_c)
            v_se = lift_se(vpi_t, vpi_c, vse_t, vse_c)
            vl_lo = v_lift - Z_95 * v_se
            vl_hi = v_lift + Z_95 * v_se
            v_sig = is_significant(v_lift, v_se)
            _formula(
                r"\text{ValueLift} = \frac{\text{VPI}_{\text{test}} - "
                r"\text{VPI}_{\text{ctrl}}}{\text{VPI}_{\text{ctrl}}}",
                "",
                f"ValueLift = {_num(v_lift)}",
            )
            _formula_box(
                [
                    f"SE(ValueLift) = {_num(v_se)}",
                    f"95% CI = [{vl_lo:.10f},  {vl_hi:.10f}]",
                    f"Statistically significant?  {'YES' if v_sig else 'NO'}",
                ]
            )

        # Cannibalization context for this channel
        if ctr_lift is not None and o_ctr is not None:
            if ctr_lift < 0 and o_ctr > 0:
                st.warning(
                    f"**{ch}** shows negative lift ({_pct(ctr_lift)}) while the "
                    f"overall experiment lift is positive ({_pct(o_ctr)}). "
                    f"This is a classic **lift cannibalization** pattern — NBA is "
                    f"redirecting the best customers away from this channel to "
                    f"higher-value alternatives."
                )
            elif ctr_lift > 0 and o_ctr > 0 and ctr_lift > o_ctr * 1.5:
                st.info(
                    f"**{ch}** shows lift ({_pct(ctr_lift)}) well above the "
                    f"overall ({_pct(o_ctr)}). This channel is likely a "
                    f"**beneficiary** of NBA's redistribution — it receives "
                    f"better-targeted customers."
                )

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

# ═══════════════════════════════════════════════════════════════════════════
# Educational expander — Why lift cannibalization happens
# ═══════════════════════════════════════════════════════════════════════════
with st.expander("📖 Why does lift cannibalization happen?", expanded=False):
    st.markdown(
        r"""
### Lift Cannibalization Explained

When Pega's NBA (Next-Best-Action) strategy decides which action to show each
customer, it uses **adaptive models** to pick the action with the highest
expected value. The **control group**, by contrast, selects actions
**randomly** (or by a simpler strategy like propensity-only).

This creates an asymmetry:

| | Test (NBA) | Control (Random) |
|---|---|---|
| **Best customers** for Action A | → Shown Action A | → Might get A, B, or C randomly |
| **Best customers** for Action B | → Shown Action B | → Might get A, B, or C randomly |
| **Marginal customers** | → Shown highest-value action | → Random assignment |

#### The cannibalization effect

Consider a simplified example with 6 customers and 3 actions (A, B, C):

- Customers C1–C3 are best suited for **Action A** (high propensity)
- Customers C4–C6 are better for **Action B** or **Action C**

**NBA (Test group):** Correctly routes C1–C3 → Action A, C4–C5 → Action B,
C6 → Action C. Each customer gets their best action.

**Random (Control group):** All customers have equal chance of seeing any action.
Some C4–C6 customers randomly get Action A even though they're not ideal for it.

**Result by action:**
- **Action A overall lift** might be *negative* — because in the control group,
  Action A also gets shown to C4–C6 (poor fit), dragging down its average CTR.
  But in the test group, Action A is only shown to C1–C3 (great fit) — yet with
  fewer total impressions. The *rate* may be higher, but the *mix* of customers
  is completely different.
- **Action B and C** might show *positive* lift because NBA sends them their
  best-fit customers.
- **Overall lift is positive** because the total accept rate across all
  customers is higher when each gets their best action.

#### Key takeaway

> **Per-channel or per-action lift is not a reliable indicator of model quality.**
> It reflects the *redistribution* of impressions, not the performance of the
> model for that specific channel. Always evaluate lift at the **overall
> experiment level** as the primary metric.

This phenomenon is well-documented in A/B testing literature as
**Simpson's Paradox** — where a trend that appears in sub-groups reverses
when the groups are combined.
"""
    )
