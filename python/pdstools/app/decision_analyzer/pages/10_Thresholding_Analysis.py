# python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py
import plotly.express as px
import polars as pl
import streamlit as st

from da_streamlit_utils import ensure_data

"# Thresholding Analysis"

"""
Explore how setting minimum thresholds for response likelihood (propensity) and priority
affects which offers reach customers. Use this to fine-tune your offer quality standards
and understand the trade-offs between offer volume and predicted performance.

**Key questions:**
* How many offers would be filtered if we raised quality thresholds?
* What's the right balance between offer volume and customer relevance?
* How do threshold changes affect specific offer categories?
"""

ensure_data()

da = st.session_state.decision_data

# ---------------------------------------------------------------------------
# Collect sampled values from arbitration once
# ---------------------------------------------------------------------------
arb_data = (
    da.getPreaggregatedFilterView.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
    .select(
        pl.col("Propensity").explode(),
        pl.col("Priority").explode(),
        pl.col("Decisions"),
    )
    .collect()
)

total_action_appearances = arb_data["Decisions"].sum()

# ---------------------------------------------------------------------------
# Sidebar: threshold sliders for propensity and priority
# ---------------------------------------------------------------------------
with st.sidebar:
    prop_range = da.getThresholdingData("Propensity", quantile_range=[0, 100])["Threshold"].to_list()

    if all(v is None for v in prop_range):
        st.warning(
            "⚠️ No actions survive to the arbitration stage in this data set. "
            "Thresholding Analysis requires actions at or after arbitration. "
            "Please check your data or filters."
        )
        st.stop()

    prop_range = [v if v is not None else 0.0 for v in prop_range]

    propensity_threshold = (
        st.slider(
            "Propensity threshold",
            prop_range[0] * 100,
            prop_range[1] * 100,
            value=0.0,
            format="%.2f%%",
        )
        / 100
    )

    prio_range = da.getThresholdingData("Priority", quantile_range=[0, 100])["Threshold"].to_list()
    prio_range = [v if v is not None else 0.0 for v in prio_range]

    priority_threshold = st.slider(
        "Priority threshold",
        prio_range[0],
        prio_range[1],
        value=0.0,
        format="%.4f",
    )

# ---------------------------------------------------------------------------
# Apply both thresholds
# ---------------------------------------------------------------------------
above_mask = (arb_data["Propensity"] >= propensity_threshold) & (arb_data["Priority"] >= priority_threshold)
above_actions = arb_data.filter(above_mask)["Decisions"].sum()
below_actions = total_action_appearances - above_actions
pct_filtered = (below_actions / total_action_appearances * 100) if total_action_appearances > 0 else 0.0

# Count interactions that would have zero actions after thresholding.
# Uses the sample (interaction-level data) for an accurate per-interaction check.
arb_sample = da.sample.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down)).collect()
total_interactions = arb_sample["Interaction ID"].n_unique()
interactions_with_survivors = arb_sample.filter(
    (pl.col("Propensity") >= propensity_threshold) & (pl.col("Priority") >= priority_threshold)
)["Interaction ID"].n_unique()
empty_interactions = total_interactions - interactions_with_survivors
empty_pct = (empty_interactions / total_interactions * 100) if total_interactions > 0 else 0.0

# ---------------------------------------------------------------------------
# Section 1: Volume impact metrics
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Impact Summary"

    st.caption(
        "Shows how many offers meet your threshold criteria. **Decisions without actions** "
        "indicates customer interactions where no offers qualify — these customers would "
        "receive nothing."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Actions", f"{total_action_appearances:,.0f}")
    c2.metric("Above Both Thresholds", f"{above_actions:,.0f}")
    c3.metric("% Actions Filtered", f"{pct_filtered:.1f}%")
    c4.metric(
        "Decisions without Actions",
        f"{empty_interactions:,}",
        delta=f"{empty_pct:.1f}% of {total_interactions:,}",
        delta_color="inverse",
        help="Interactions where no action survives both thresholds — these decisions would have nothing to present.",
    )

# ---------------------------------------------------------------------------
# Section 2: Propensity and Priority distributions with threshold lines
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Response Likelihood and Priority Distributions"

    st.caption(
        "See how your offers are distributed across response likelihood (propensity) and "
        "priority values. The red dashed line shows your current threshold — offers below "
        "this line would be filtered out."
    )

    col1, col2 = st.columns(2)
    with col1:
        propensity_hist = px.histogram(
            arb_data,
            x="Propensity",
            y="Decisions",
            labels={"Decisions": "Actions"},
        )
        propensity_hist.update_xaxes(tickformat=",.0%")
        propensity_hist.add_vline(
            x=propensity_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Threshold: {propensity_threshold:.2%}",
            annotation_position="top right",
            annotation_font_color="red",
        )
        st.plotly_chart(propensity_hist)

    with col2:
        prio_data = (
            da.getPreaggregatedFilterView.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
            .select(pl.col("Priority").explode(), pl.col("Decisions"))
            .collect()
        )
        priority_hist = px.histogram(
            prio_data,
            x="Priority",
            y="Decisions",
            log_y=True,
            labels={"Decisions": "Actions"},
        )
        priority_hist.add_vline(
            x=priority_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Threshold: {priority_threshold:.4f}",
            annotation_position="top right",
            annotation_font_color="red",
        )
        st.plotly_chart(priority_hist)

# ---------------------------------------------------------------------------
# Section 3: Distribution of surviving actions above both thresholds
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Offers Meeting Quality Thresholds"

    surviving = (
        da.getPreaggregatedFilterView.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
        .select(
            pl.col("Issue"),
            pl.col("Group"),
            pl.col("Propensity").explode().alias("Propensity"),
            pl.col("Priority").explode().alias("Priority"),
            pl.col("Decisions"),
        )
        .filter((pl.col("Propensity") >= propensity_threshold) & (pl.col("Priority") >= priority_threshold))
        .group_by(["Issue", "Group"])
        .agg(pl.sum("Decisions").alias("Actions"))
        .sort("Actions", descending=True)
        .filter(pl.col("Actions") > 0)
        .collect()
    )

    if surviving.height == 0:
        st.info("No actions survive at these thresholds.")
    else:
        fig = px.bar(
            surviving,
            x="Group",
            y="Actions",
            color="Issue",
            template="pega",
        )
        fig.update_xaxes(tickangle=45, automargin=True, title="")
        fig.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig)
