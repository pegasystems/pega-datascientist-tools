# python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py
import plotly.express as px
import polars as pl
import streamlit as st

from da_streamlit_utils import ensure_data

"# Propensity Thresholding Analysis"

"""
Explore how applying a propensity threshold affects the volume and distribution
of actions at arbitration.

* What is the effect of new offers (new models with propensity 0.5, showing a peak there)?
* What should my propensity threshold be and how does it affect volumes and distributions?
  Filter on a specific channel globally to answer this per channel.
* Is the random control group working? (propensity spreading 0–1)
"""

ensure_data()

da = st.session_state.decision_data

# ---------------------------------------------------------------------------
# Sidebar: threshold slider
# ---------------------------------------------------------------------------
st.session_state["sidebar"] = st.sidebar

with st.session_state["sidebar"]:
    value_range = da.getThresholdingData("Propensity", quantile_range=[0, 100])["Threshold"].to_list()

    if all(v is None for v in value_range):
        st.warning(
            "⚠️ No actions survive to the arbitration stage in this data set. "
            "Thresholding Analysis requires actions at or after arbitration. "
            "Please check your data or filters."
        )
        st.stop()

    value_range = [v if v is not None else 0.0 for v in value_range]

    current_threshold = (
        st.slider(
            "Propensity threshold",
            value_range[0] * 100,
            value_range[1] * 100,
            format="%.2f%%",
        )
        / 100
    )

# ---------------------------------------------------------------------------
# Collect sampled propensity values from arbitration once
# ---------------------------------------------------------------------------
arb_propensities = (
    da.getPreaggregatedFilterView.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
    .select(pl.col("Propensity").explode(), pl.col("Decisions"))
    .collect()
)

total_decisions = arb_propensities["Decisions"].sum()
above_decisions = arb_propensities.filter(pl.col("Propensity") >= current_threshold)["Decisions"].sum()
below_decisions = total_decisions - above_decisions
pct_filtered = (below_decisions / total_decisions * 100) if total_decisions > 0 else 0.0

# ---------------------------------------------------------------------------
# Section 1: Volume impact metrics
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Threshold Impact"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total at Arbitration", f"{total_decisions:,.0f}")
    c2.metric("Above Threshold", f"{above_decisions:,.0f}")
    c3.metric("Below Threshold", f"{below_decisions:,.0f}")
    c4.metric("% Filtered", f"{pct_filtered:.1f}%")

# ---------------------------------------------------------------------------
# Section 2: Propensity distribution with threshold line
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Propensity and Priority Distributions"

    col1, col2 = st.columns(2)
    with col1:
        propensity_hist = px.histogram(
            arb_propensities,
            x="Propensity",
            y="Decisions",
        )
        propensity_hist.update_xaxes(tickformat=",.0%")
        propensity_hist.add_vline(
            x=current_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Threshold: {current_threshold:.2%}",
            annotation_position="top right",
            annotation_font_color="red",
        )
        st.plotly_chart(propensity_hist, use_container_width=True)

    with col2:
        st.plotly_chart(
            px.histogram(
                da.getPreaggregatedFilterView.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
                .select(pl.col("Priority").explode(), pl.col("Decisions"))
                .collect(),
                x="Priority",
                y="Decisions",
                log_y=True,
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Section 3: Distribution of surviving actions at the threshold
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Action Distribution above Threshold"

    surviving = (
        da.getPreaggregatedFilterView.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
        .select(
            pl.col("Issue"),
            pl.col("Group"),
            pl.col("Propensity").explode().alias("Propensity"),
            pl.col("Decisions"),
        )
        .filter(pl.col("Propensity") >= current_threshold)
        .group_by(["Issue", "Group"])
        .agg(pl.sum("Decisions"))
        .sort("Decisions", descending=True)
        .filter(pl.col("Decisions") > 0)
        .collect()
    )

    if surviving.height == 0:
        st.info("No actions survive at this threshold.")
    else:
        fig = px.bar(
            surviving,
            x="Group",
            y="Decisions",
            color="Issue",
            template="pega",
        )
        fig.update_xaxes(tickangle=45, automargin=True, title="")
        fig.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Decile chart (static context)
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Propensity Decile Overview"

    st.plotly_chart(
        da.plot.threshold_deciles("Propensity", "Propensity"),
        use_container_width=True,
    )
