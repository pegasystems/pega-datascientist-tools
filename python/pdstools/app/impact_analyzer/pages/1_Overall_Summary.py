# python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py
"""Overall Summary page — experiment cards with transparency + overview table.

Combines the per-experiment transparency cards (trend charts, CI bars,
formula walkthroughs) with the original tabular overview at the bottom.
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
    LiftResult,
    accept_rate,
    binomial_se,
    calculate_engagement_lift,
    calculate_lift,
    is_significant,
    lift_se,
    value_se,
    value_variance,
)
from pdstools.utils.streamlit_utils import standard_page_config

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
standard_page_config(page_title="Impact Analyzer · Overall Summary")

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
# Lightweight CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.formula-box {
    background: #F0F4FA; border-left: 4px solid #1F6BFF; padding: 12px 16px;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
    line-height: 1.7; border-radius: 4px; margin: 8px 0;
    white-space: pre-wrap; color: #1B2A4A;
}
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
# Helpers — formatting
# ---------------------------------------------------------------------------
def _pct(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.{digits}f}%"


def _num(v, digits: int = 10) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _lift_class(v) -> str:
    if v is None:
        return "neutral"
    return "positive" if v >= 0 else "negative"


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _chart_lift_trend(trend_df, exp_name, metric="CTR_Lift", label="Engagement Lift %"):
    """Build a trend line chart with CI bands + EWMA for one experiment."""
    df = trend_df.filter(
        (pl.col("Experiment") == exp_name) & pl.col(metric).is_not_null() & pl.col(metric).is_finite()
    ).sort("SnapshotTime")
    if len(df) < 2:
        return None

    xs = df["SnapshotTime"].to_list()
    lifts_raw = df[metric].to_list()

    ci_lo, ci_hi = [], []
    is_value = "Value" in metric
    for row in df.iter_rows(named=True):
        n_t = int(row.get("Impressions_Test") or 0)
        a_t = int(row.get("Accepts_Test") or 0)
        n_c = int(row.get("Impressions_Control") or 0)
        a_c = int(row.get("Accepts_Control") or 0)
        lift_val = row[metric]
        if n_t > 0 and n_c > 0 and a_t > 0 and a_c > 0:
            if is_value:
                # Value lift CI: use value_se when VPI available (VBD),
                # else fall back to engagement-based SE (PDC where VPI=null)
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
            else:
                se_t = binomial_se(a_t, n_t)
                se_c = binomial_se(a_c, n_c)
                p_t = accept_rate(a_t, n_t)
                p_c = accept_rate(a_c, n_c)
                se = lift_se(p_t, p_c, se_t, se_c)
            hw = Z_95 * se
            ci_lo.append((lift_val - hw) * 100)
            ci_hi.append((lift_val + hw) * 100)
        else:
            ci_lo.append(None)
            ci_hi.append(None)

    lifts_pct = [v * 100 for v in lifts_raw]

    span = min(7, len(lifts_pct))
    alpha = 2.0 / (span + 1)
    ewma = []
    for v in lifts_pct:
        ewma.append(v if not ewma else alpha * v + (1 - alpha) * ewma[-1])

    fig = go.Figure()
    valid = [(x, lo, hi) for x, lo, hi in zip(xs, ci_lo, ci_hi, strict=False) if lo is not None]
    if valid:
        bx = [v[0] for v in valid]
        blo = [v[1] for v in valid]
        bhi = [v[2] for v in valid]
        fig.add_trace(
            go.Scatter(
                x=bx + bx[::-1],
                y=bhi + blo[::-1],
                fill="toself",
                fillcolor=_BAND,
                line=dict(width=0),
                hoverinfo="skip",
                name="95 % CI",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=lifts_pct,
            mode="lines",
            line=dict(color=_BLUE, width=1.5),
            name=label,
            hovertemplate="%{x|%b %d}<br>Lift: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ewma,
            mode="lines",
            line=dict(color=_GREEN, width=2.5),
            name="Smoothed (EWMA)",
            hovertemplate="%{x|%b %d}<br>Smoothed: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=2.5, line_dash="solid", line_color=_GREY, opacity=0.8)
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFBFD",
        font=dict(color=_TEXT, size=11),
        margin=dict(l=50, r=10, t=10, b=40),
        xaxis=dict(title="", gridcolor="#EEF0F4"),
        yaxis=dict(title=label, gridcolor="#EEF0F4", zeroline=False, ticksuffix="%"),
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers — rendering
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Render one experiment card
# ---------------------------------------------------------------------------
def _render_experiment_card(row: dict, idx: int, trend_df: pl.DataFrame | None = None) -> None:
    """Render a full experiment card with metrics, charts and formula steps."""

    exp_name: str = row["Experiment"]
    test_impr = int(row.get("Impressions_Test") or 0)
    test_acc = int(row.get("Accepts_Test") or 0)
    ctrl_impr = int(row.get("Impressions_Control") or 0)
    ctrl_acc = int(row.get("Accepts_Control") or 0)
    is_active = test_impr > 0 or ctrl_impr > 0

    # ── Header ────────────────────────────────────────────────────────
    badge_html = (
        '<span style="display:inline-block;background:#2ca02c;color:white;'
        "font-size:11px;font-weight:700;padding:2px 10px;border-radius:3px;"
        'vertical-align:middle;margin-left:8px">ACTIVE</span>'
        if is_active
        else '<span style="display:inline-block;background:#8795A1;color:white;'
        "font-size:11px;font-weight:700;padding:2px 10px;border-radius:3px;"
        'vertical-align:middle;margin-left:8px">INACTIVE</span>'
    )
    st.markdown(f"#### {exp_name} {badge_html}", unsafe_allow_html=True)

    if not is_active:
        st.info("No data received yet for this experiment.")
        return

    # ── Compute via library functions ─────────────────────────────────
    p_t = accept_rate(test_acc, test_impr)
    p_c = accept_rate(ctrl_acc, ctrl_impr)
    se_t = binomial_se(test_acc, test_impr)
    se_c = binomial_se(ctrl_acc, ctrl_impr)

    eng = calculate_engagement_lift(test_acc, test_impr, ctrl_acc, ctrl_impr)

    # Value lift — use VPI columns if available
    vpi_t = row.get("ValuePerImpression_Test")
    vpi_c = row.get("ValuePerImpression_Control")
    val_lift_pdc = row.get("Value_Lift")

    # VPI can be nan (PDC data) — normalise to None
    if vpi_t is not None and (isinstance(vpi_t, float) and math.isnan(vpi_t)):
        vpi_t = None
    if vpi_c is not None and (isinstance(vpi_c, float) and math.isnan(vpi_c)):
        vpi_c = None
    if val_lift_pdc is not None and (isinstance(val_lift_pdc, float) and math.isnan(val_lift_pdc)):
        val_lift_pdc = None

    # Derive action values for transparency display
    av_t = (vpi_t / p_t) if (vpi_t is not None and p_t) else None
    av_c = (vpi_c / p_c) if (vpi_c is not None and p_c) else None

    val = None
    if av_t is not None and av_c is not None and vpi_c > 0:
        _vse_t = value_se(test_acc, test_impr, av_t)
        _vse_c = value_se(ctrl_acc, ctrl_impr, av_c)
        _val_lift = calculate_lift(vpi_t, vpi_c)
        _val_se = lift_se(vpi_t, vpi_c, _vse_t, _vse_c)
        val = LiftResult(
            lift=_val_lift,
            se=_val_se,
            significant=is_significant(_val_lift, _val_se),
            test_rate=vpi_t,
            control_rate=vpi_c,
            test_se=_vse_t,
            control_se=_vse_c,
        )
    elif val_lift_pdc is not None and p_c > 0:
        _val_se = lift_se(p_t, p_c, se_t, se_c)
        val = LiftResult(
            lift=val_lift_pdc,
            se=_val_se,
            significant=is_significant(val_lift_pdc, _val_se),
            test_rate=p_t,
            control_rate=p_c,
            test_se=se_t,
            control_se=se_c,
        )

    # ── Significance callout ──────────────────────────────────────────
    if eng.significant:
        st.success("Statistically significant at 95 % confidence")
    else:
        st.warning("Not yet significant — more data needed")

    # ── Metrics row ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f'<div class="metric-label">ENG. LIFT</div>'
        f'<div class="metric-value {_lift_class(eng.lift)}">{_pct(eng.lift)}</div>',
        unsafe_allow_html=True,
    )
    vl_val = val.lift if val else val_lift_pdc
    m2.markdown(
        f'<div class="metric-label">VALUE LIFT</div>'
        f'<div class="metric-value {_lift_class(vl_val)}">{_pct(vl_val)}</div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-label">TEST n</div><div class="metric-value">{test_impr:,}</div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-label">CTRL n</div><div class="metric-value">{ctrl_impr:,}</div>',
        unsafe_allow_html=True,
    )

    # ── Test vs control comparison table ──────────────────────────────
    st.table(
        {
            "": ["Impressions", "Accepts", "Accept Rate"],
            "Test": [f"{test_impr:,}", f"{test_acc:,}", _pct(p_t, 6)],
            "Control": [f"{ctrl_impr:,}", f"{ctrl_acc:,}", _pct(p_c, 6)],
        }
    )

    # ── Numeric lift summaries — Engagement / Value ─────────────────
    tab_eng, tab_val = st.tabs(["Engagement Lift", "Value Lift"])

    with tab_eng:
        if trend_df is not None:
            _fig_t = _chart_lift_trend(
                trend_df,
                exp_name,
                metric="CTR_Lift",
                label="Engagement Lift %",
            )
            if _fig_t is not None:
                st.plotly_chart(_fig_t, key=f"os_trend_eng_{idx}")

        ci_hw = eng.ci_95()
        st.markdown(
            f"Engagement Lift = **{_pct(eng.lift)}** · "
            f"SE = `{_num(eng.se)}` · "
            f"95 % CI = [`{_num(eng.lift - ci_hw)}`, `{_num(eng.lift + ci_hw)}`]"
        )

        with st.expander("**Formula details**"):
            st.markdown("##### Step 1 — Accept Rates")
            _formula(
                FORMULAS["accept_rate"].latex,
                "Accept rate for the test and control groups",
                f"p_test = {test_acc} / {test_impr} = {_num(p_t)}\np_ctrl = {ctrl_acc} / {ctrl_impr} = {_num(p_c)}",
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
            _formula(FORMULAS["significance"].latex)
            if eng.lift is not None and eng.se is not None:
                lo = eng.lift - Z_95 * eng.se
                hi = eng.lift + Z_95 * eng.se
                sig = "YES — CI does not contain 0" if eng.significant else "NO — CI contains 0"
                _formula_box(
                    [
                        f"= [{lo:.10f},  {hi:.10f}]",
                        f"Statistically significant?  {sig}",
                    ]
                )

            if trend_df is not None:
                st.markdown("##### Trend Chart — CI Band & Smoothing")
                st.latex(FORMULAS["ci_band"].latex)
                st.caption(FORMULAS["ci_band"].description)
                st.latex(FORMULAS["ewma"].latex)
                st.caption(FORMULAS["ewma"].description)

    with tab_val:
        if val is not None:
            if trend_df is not None:
                _fig_tv = _chart_lift_trend(
                    trend_df,
                    exp_name,
                    metric="Value_Lift",
                    label="Value Lift %",
                )
                if _fig_tv is not None:
                    st.plotly_chart(_fig_tv, key=f"os_trend_val_{idx}")

            val_hw = val.ci_95()
            st.markdown(
                f"Value Lift = **{_pct(val.lift)}** · "
                f"SE = `{_num(val.se)}` · "
                f"95 % CI = [`{_num(val.lift - val_hw)}`, `{_num(val.lift + val_hw)}`]"
            )
            if av_t is None:
                st.caption("Note: SE approximated from engagement SEs (per-group VPI not available in this dataset).")
        else:
            st.caption("Value lift requires action-value data.")

        with st.expander("**Formula details**"):
            st.markdown("##### Step 1 — Accept Rates")
            _formula(
                FORMULAS["accept_rate"].latex,
                "Accept rate for the test and control groups",
                f"p_test = {test_acc} / {test_impr} = {_num(p_t)}\np_ctrl = {ctrl_acc} / {ctrl_impr} = {_num(p_c)}",
            )

            st.markdown("##### Step 2 — Standard Errors")
            _formula(
                FORMULAS["binomial_se"].latex,
                "Binomial standard error",
                f"SE_test = {_num(se_t)}\nSE_ctrl = {_num(se_c)}",
            )

            st.markdown("##### Step 3 — Value Lift (VPI-based)")
            if val is not None and av_t is not None and av_c is not None:
                _formula(
                    FORMULAS["vpi"].latex,
                    "Value Per Impression",
                    f"AV_test = {_num(av_t, 4)}  |  AV_ctrl = {_num(av_c, 4)}\n"
                    f"VPI_test = {_num(p_t)} × {_num(av_t, 4)} = {_num(vpi_t)}\n"
                    f"VPI_ctrl = {_num(p_c)} × {_num(av_c, 4)} = {_num(vpi_c)}",
                )

                vvar_t = value_variance(test_acc, test_impr, av_t)
                vvar_c = value_variance(ctrl_acc, ctrl_impr, av_c)
                vse_t = value_se(test_acc, test_impr, av_t)
                vse_c = value_se(ctrl_acc, ctrl_impr, av_c)

                _formula(
                    r"\text{Var}(\text{VPI}) = p\,(1-p) \times \text{AV}^2",
                    "Bernoulli per-observation variance",
                    f"Var_test = {_num(vvar_t)}\nVar_ctrl = {_num(vvar_c)}",
                )
                _formula(
                    r"\text{SE}(\text{VPI}) = \sqrt{{\text{{Var}}}/ n}",
                    "",
                    f"SE_vpi_test = {_num(vse_t)}\nSE_vpi_ctrl = {_num(vse_c)}",
                )
                _formula(
                    r"\text{ValueLift} = \frac{\text{VPI}_{\text{test}} - "
                    r"\text{VPI}_{\text{ctrl}}}{\text{VPI}_{\text{ctrl}}}",
                    "",
                    f"ValueLift = {_num(val.lift)}",
                )
                if val.se is not None:
                    vl_lo = val.lift - Z_95 * val.se
                    vl_hi = val.lift + Z_95 * val.se
                    vl_sig = "YES" if val.significant else "NO"
                    _formula_box(
                        [
                            f"SE(ValueLift) = {_num(val.se)}",
                            f"95% CI = [{vl_lo:.10f},  {vl_hi:.10f}]",
                            f"Statistically significant?  {vl_sig}",
                        ]
                    )
            elif val is not None:
                _formula(
                    r"\text{ValueLift} = \frac{\text{VPI}_{\text{test}} - "
                    r"\text{VPI}_{\text{ctrl}}}{\text{VPI}_{\text{ctrl}}}",
                    "Pre-computed from the data source",
                    f"ValueLift = {_num(val.lift)}",
                )
                if val.se is not None:
                    vl_lo = val.lift - Z_95 * val.se
                    vl_hi = val.lift + Z_95 * val.se
                    vl_sig = "YES" if val.significant else "NO"
                    _formula_box(
                        [
                            f"SE(ValueLift) ≈ {_num(val.se)}  (approximated from engagement SEs)",
                            f"95% CI ≈ [{vl_lo:.10f},  {vl_hi:.10f}]",
                            f"Statistically significant?  {vl_sig}",
                        ]
                    )
                st.caption(
                    "Per-group action values not available in this dataset. "
                    "Value lift is pre-computed; SE is approximated from engagement SEs."
                )
            elif val_lift_pdc is not None:
                st.markdown(f"Value Lift = **{_pct(val_lift_pdc)}** (pre-computed)")
                st.caption(
                    "Per-group action values not available. Only the raw value lift from the data source is shown."
                )
            else:
                st.caption("Value lift data not available for this experiment.")

            if trend_df is not None:
                st.markdown("##### Trend Chart — CI Band & Smoothing")
                st.latex(FORMULAS["ci_band"].latex)
                st.caption(FORMULAS["ci_band"].description)
                st.latex(FORMULAS["ewma"].latex)
                st.caption(FORMULAS["ewma"].description)


# ===========================================================================
# MAIN PAGE
# ===========================================================================
"# Overall Summary"

"""
View lift metrics aggregated across all channels by default. Each experiment
card shows trend charts, confidence intervals, and the exact formulas behind
every number. Use the **Channel / Direction** filter in the sidebar to narrow
the lift chart, cards, and detailed table to a single channel.
"""

# --- Gather data -----------------------------------------------------------
experiments = ia.summarize_experiments().collect()

if experiments.is_empty():
    st.warning("No experiment data available.")
    st.stop()

# Check for multi-snapshot trend data. We compute the channel-aggregated
# trend by default and a per-channel trend frame so the cards' trend
# sparklines stay accurate when the sidebar channel filter is active.
_trend_df: pl.DataFrame | None = None
_trend_df_by_channel: pl.DataFrame | None = None
if "SnapshotTime" in ia.ia_data.collect_schema().names():
    _n_snaps = ia.ia_data.select("SnapshotTime").unique().collect().height
    if _n_snaps >= 2:
        try:
            _trend_df = ia.summarize_experiments(by="SnapshotTime").collect()
        except Exception:
            pass
        if "Channel" in ia.ia_data.collect_schema().names():
            try:
                _trend_df_by_channel = ia.summarize_experiments(by=["Channel", "SnapshotTime"]).collect()
            except Exception:
                pass

# --- Per-experiment lift overview -----------------------------------------
rows = experiments.to_dicts()


def _build_lift_chart_data(rows_data: list[dict]) -> list[dict]:
    """Compute lift ± SE for each active experiment."""
    plot_data = []
    for r in rows_data:
        t_i = int(r.get("Impressions_Test") or 0)
        t_a = int(r.get("Accepts_Test") or 0)
        c_i = int(r.get("Impressions_Control") or 0)
        c_a = int(r.get("Accepts_Control") or 0)
        if t_i > 0 and c_i > 0:
            eng = calculate_engagement_lift(t_a, t_i, c_a, c_i)

            vpi_t = r.get("ValuePerImpression_Test")
            vpi_c = r.get("ValuePerImpression_Control")
            val_lift_pdc = r.get("Value_Lift")
            if vpi_t is not None and isinstance(vpi_t, float) and math.isnan(vpi_t):
                vpi_t = None
            if vpi_c is not None and isinstance(vpi_c, float) and math.isnan(vpi_c):
                vpi_c = None
            if val_lift_pdc is not None and isinstance(val_lift_pdc, float) and math.isnan(val_lift_pdc):
                val_lift_pdc = None

            p_t = accept_rate(t_a, t_i)
            p_c = accept_rate(c_a, c_i)
            av_t = (vpi_t / p_t) if (vpi_t is not None and p_t) else None
            av_c = (vpi_c / p_c) if (vpi_c is not None and p_c) else None

            val_lift = None
            val_se_val = None
            if av_t is not None and av_c is not None and vpi_c and vpi_c > 0:
                val_lift = calculate_lift(vpi_t, vpi_c)
                _vse_t = value_se(t_a, t_i, av_t)
                _vse_c = value_se(c_a, c_i, av_c)
                val_se_val = lift_se(vpi_t, vpi_c, _vse_t, _vse_c)
            elif val_lift_pdc is not None and p_c > 0:
                val_lift = val_lift_pdc
                se_t = binomial_se(t_a, t_i)
                se_c = binomial_se(c_a, c_i)
                val_se_val = lift_se(p_t, p_c, se_t, se_c)

            plot_data.append(
                {
                    "name": r["Experiment"],
                    "eng_lift": eng.lift,
                    "eng_se": eng.se,
                    "val_lift": val_lift,
                    "val_se": val_se_val,
                }
            )
    return plot_data


_lift_chart_data = _build_lift_chart_data(rows)

# Per-channel summary (short-term facet — see
# docs/plans/impact-analyzer/overall-show-channel-in-tiles.md for the proper fix
# that surfaces channel in the experiment cards too). We pre-compute once so
# the same channel selector can drive the chart, the cards, and the summary
# table; channels that yield no plottable rows are hidden from the dropdown.
_schema_names = ia.ia_data.collect_schema().names()
_per_channel_chart_data: dict[str, list[dict]] = {}
_per_channel_rows: dict[str, list[dict]] = {}
if "Channel" in _schema_names:
    try:
        _per_channel_summary = ia.summarize_experiments(by="Channel").collect()
        for _ch in sorted(_per_channel_summary["Channel"].drop_nulls().unique().to_list()):
            _ch_rows = _per_channel_summary.filter(pl.col("Channel") == _ch).to_dicts()
            _ch_chart = _build_lift_chart_data(_ch_rows)
            if _ch_chart:
                _per_channel_chart_data[_ch] = _ch_chart
                _per_channel_rows[_ch] = _ch_rows
    except Exception:
        _per_channel_chart_data = {}
        _per_channel_rows = {}
_channel_options: list[str] = list(_per_channel_chart_data.keys())
_channel_filter = "Any"

# Page-wide channel filter lives in the sidebar (DA convention) so it
# governs the chart, the experiment cards, and the summary table from one
# place. "Any" is the aggregate sentinel — matches DA's terminology.
if _channel_options:
    with st.sidebar:
        st.divider()
        st.caption("**Filters**")
        _channel_filter = st.selectbox(
            "Channel / Direction",
            ["Any", *_channel_options],
            help="Filter the lift chart, experiment cards, and summary table to a specific channel. 'Any' shows the aggregate across all channels.",
            key="lift_chart_channel_filter",
        )

if _lift_chart_data:
    with st.container(border=True):
        kpi_metric = st.radio(
            "KPI",
            ["Engagement Lift", "Value Lift"],
            horizontal=True,
            label_visibility="collapsed",
        )
        _lift_key = "eng" if kpi_metric == "Engagement Lift" else "val"

        if _channel_filter != "Any":
            _lift_chart_data_active = _per_channel_chart_data.get(_channel_filter, _lift_chart_data)
        else:
            _lift_chart_data_active = _lift_chart_data

        _title_suffix = (
            "All Experiments" if _channel_filter == "Any" else f"All Experiments — channel: {_channel_filter}"
        )
        st.markdown(f"### {kpi_metric} — {_title_suffix}")
        st.caption(
            "Each row is one experiment. Diamond = point estimate, "
            "whiskers = 95 % CI.  Green = significant positive, red = significant "
            "negative, grey = not significant."
        )

        if not _lift_chart_data_active:
            st.info(f"No experiment data for channel **{_channel_filter}**.")

        fig = go.Figure()
        names = []
        _narrow_ci = []
        for _i, d in enumerate(_lift_chart_data_active):
            lift = d[f"{_lift_key}_lift"]
            se = d[f"{_lift_key}_se"]
            if lift is None or se is None:
                continue

            lp = lift * 100
            hw = Z_95 * se * 100
            lo, hi = lp - hw, lp + hw
            sig = abs(lift) > Z_95 * se
            colour = _GREEN if (sig and lp > 0) else _RED if (sig and lp < 0) else _GREY

            names.append(d["name"])
            y_pos = len(names) - 1

            # Track experiments where CI is too narrow to see
            if hw < 0.5:
                _narrow_ci.append(f"**{d['name']}**: {lp:+.2f}% ± {hw:.3f}%")

            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[y_pos, y_pos],
                    mode="lines",
                    line=dict(color=colour, width=2.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[y_pos, y_pos],
                    mode="markers",
                    marker=dict(symbol="line-ns-open", size=12, color=colour, line_width=2.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[lp],
                    y=[y_pos],
                    mode="markers",
                    marker=dict(symbol="diamond", size=14, color=colour, line=dict(color="white", width=1.5)),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{d['name']}</b><br>"
                        f"Lift: {lp:+.3f}%<br>"
                        f"95% CI: [{lo:.3f}%, {hi:.3f}%]<br>"
                        f"{'Significant' if sig else 'Not significant'}"
                        f"<extra></extra>"
                    ),
                )
            )

        fig.add_vline(x=0, line_width=2.5, line_dash="solid", line_color=_GREY, opacity=0.8)

        fig.update_layout(
            height=max(180, 60 * len(names)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FAFBFD",
            font=dict(color=_TEXT, size=12),
            margin=dict(l=220, r=30, t=10, b=40),
            yaxis=dict(tickvals=list(range(len(names))), ticktext=names, showgrid=False, zeroline=False),
            xaxis=dict(title=f"{kpi_metric} %", gridcolor="#EEF0F4", zeroline=False, ticksuffix="%"),
        )
        st.plotly_chart(fig, key="lift_chart")

        if _narrow_ci:
            st.info(
                "🔍 **CI whiskers too narrow to display visually** "
                "(high sample sizes produce very tight intervals):\n\n" + "\n".join(f"- {c}" for c in _narrow_ci),
                icon="ℹ️",
            )

# --- Experiment cards (2-column grid) --------------------------------------
# When a channel is selected above, cards and the summary table show
# per-channel metrics for that channel; otherwise they aggregate across
# channels.
if _channel_filter != "Any":
    _cards_rows = _per_channel_rows.get(_channel_filter, rows)
    _cards_scope_caption = f"Showing experiments for channel **{_channel_filter}**."
else:
    _cards_rows = rows
    _cards_scope_caption = None

if _cards_scope_caption:
    st.caption(_cards_scope_caption)

active = [r for r in _cards_rows if (r.get("Impressions_Test") or 0) > 0 or (r.get("Impressions_Control") or 0) > 0]
inactive = [r for r in _cards_rows if r not in active]

tab_active, tab_inactive = st.tabs(
    [
        f"Active ({len(active)})",
        f"Inactive ({len(inactive)})",
    ]
)

# Pick the right trend frame: per-channel slice when filtered, otherwise
# the channel-aggregated frame. Both are pre-computed once above.
if _channel_filter != "Any" and _trend_df_by_channel is not None:
    _card_trend_df = _trend_df_by_channel.filter(pl.col("Channel") == _channel_filter)
    if _card_trend_df.is_empty():
        _card_trend_df = None
else:
    _card_trend_df = _trend_df

with tab_active:
    if not active:
        st.info("No active experiments. Load data to see results.")
    else:
        cols = st.columns(2)
        for i, row in enumerate(active):
            with cols[i % 2]:
                with st.container(border=True):
                    _render_experiment_card(row, i, trend_df=_card_trend_df)

with tab_inactive:
    if not inactive:
        st.success("All experiments are active!")
    else:
        for row in inactive:
            st.markdown(
                f'**{row["Experiment"]}** — <span style="color:#8795A1">INACTIVE</span> · No data received.',
                unsafe_allow_html=True,
            )

# --- Detailed metrics table ------------------------------------------------
# Match the channel filter:
#   - "Any"  → aggregate across channels (no facet), one row per experiment,
#              matching the cards and lift chart above.
#   - <channel selected> → narrow to that channel (facet="Channel" so the
#              Channel column is present for the query to bind against).
_has_channel = "Channel" in ia.ia_data.collect_schema().names()
if _channel_filter != "Any" and _has_channel:
    _facet = "Channel"
    _detailed_query = pl.col("Channel") == _channel_filter
else:
    _facet = None
    _detailed_query = None

with st.container(border=True):
    "## Detailed Metrics"

    st.caption("Tabular view of lift metrics with exact values. Use this for detailed analysis and reporting.")

    table = ia.plot.overview(facet=_facet, query=_detailed_query, return_df=True).collect()
    st.dataframe(table)
