# python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py
import math

import plotly.express as px
import polars as pl
import streamlit as st

from da_streamlit_utils import contextual_filters, ensure_data
from pdstools.decision_analyzer.utils import apply_filter

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

# Apply channel filter to sample data
filtered_data = da.filtered_sample

# Check for empty results when a specific channel is selected
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.len()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()

channel_filter = st.session_state.get("page_channel_expr")

# ---------------------------------------------------------------------------
# Collect sampled values from arbitration once
# ---------------------------------------------------------------------------
arb_data = (
    apply_filter(da.preaggregated_filter_view, channel_filter)
    .filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
    .select(
        pl.col("Propensity").explode(),
        pl.col("Priority").explode(),
        pl.col("Decisions"),
    )
    .collect()
)

total_action_appearances = arb_data["Decisions"].sum()

if arb_data.height == 0:
    st.warning(
        "⚠️ No actions survive to the arbitration stage for the current filters. Please check your data or filters."
    )
    st.stop()

# Slider upper bounds: use P75 of the filtered data so the slider has
# useful resolution even when the distribution is heavily skewed.
prop_max = max(arb_data["Propensity"].quantile(0.75), 1e-9)
prio_max = max(arb_data["Priority"].quantile(0.75), 1e-9)

# ---------------------------------------------------------------------------
# Session state for thresholds (shared between sliders and chart clicks)
# ---------------------------------------------------------------------------
if "propensity_threshold_pct" not in st.session_state:
    st.session_state.propensity_threshold_pct = 0.0
if "priority_threshold_val" not in st.session_state:
    st.session_state.priority_threshold_val = 0.0

# Pick up chart clicks from the previous run (before sliders render)
_prev_prop = st.session_state.get("prop_chart")
if _prev_prop and getattr(_prev_prop, "selection", None) and _prev_prop.selection.points:
    _x = _prev_prop.selection.points[0]["x"]
    if isinstance(_x, (int, float)):
        st.session_state.propensity_threshold_pct = min(_x * 100, prop_max * 100)

_prev_prio = st.session_state.get("prio_chart")
if _prev_prio and getattr(_prev_prio, "selection", None) and _prev_prio.selection.points:
    _x = _prev_prio.selection.points[0]["x"]
    if isinstance(_x, (int, float)):
        st.session_state.priority_threshold_val = min(float(_x), prio_max)

prio_step = prio_max / 100
prio_decimals = max(4, -math.floor(math.log10(prio_step)) + 1)

# ---------------------------------------------------------------------------
# Sidebar: threshold sliders for propensity and priority
# ---------------------------------------------------------------------------
with st.sidebar:
    propensity_threshold = (
        st.slider(
            "Propensity threshold",
            0.0,
            prop_max * 100,
            format="%.2f%%",
            key="propensity_threshold_pct",
        )
        / 100
    )

    priority_threshold = st.slider(
        "Priority threshold",
        0.0,
        prio_max,
        key="priority_threshold_val",
        step=prio_step,
        format=f"%.{prio_decimals}f",
    )
    contextual_filters()

# ---------------------------------------------------------------------------
# Apply both thresholds
# ---------------------------------------------------------------------------
above_mask = (arb_data["Propensity"] >= propensity_threshold) & (arb_data["Priority"] >= priority_threshold)
above_actions = arb_data.filter(above_mask)["Decisions"].sum()
below_actions = total_action_appearances - above_actions
pct_filtered = (below_actions / total_action_appearances * 100) if total_action_appearances > 0 else 0.0

# Count interactions that would have zero actions after thresholding.
# Uses the sample (interaction-level data) for an accurate per-interaction check.
arb_sample = filtered_data.filter(pl.col(da.level).is_in(da.stages_from_arbitration_down)).collect()
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
        "Shows how many offers meet your threshold criteria. **% Decisions without actions** "
        "indicates decisions where no offers qualify — these customers would "
        "receive nothing."
    )

    c1, c2, c3, c4 = st.columns(4)
    avg_actions = total_action_appearances / total_interactions if total_interactions > 0 else 0.0
    avg_qualifying = above_actions / total_interactions if total_interactions > 0 else 0.0
    c1.metric("Avg Actions per Decision", f"{avg_actions:.1f}")
    c2.metric("Avg Qualifying per Decision", f"{avg_qualifying:.1f}")
    c3.metric("% Actions Filtered", f"{pct_filtered:.1f}%")
    c4.metric(
        "% Decisions without Actions",
        f"{empty_pct:.1f}%",
        delta_color="inverse",
        help="Decisions where no action survives both thresholds — these decisions would have nothing to present.",
    )

# ---------------------------------------------------------------------------
# Section 2: Propensity and Priority distributions with threshold lines
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Response Likelihood and Priority Distributions"

    st.caption(
        "See how your offers are distributed across response likelihood (propensity) and "
        "priority values. The red dashed line shows your current threshold — "
        "**click a bar** to set the threshold, or use the sidebar sliders."
    )

    # Filter to P75 range so histogram bins are meaningful
    plot_data = arb_data.filter((pl.col("Propensity") <= prop_max) & (pl.col("Priority") <= prio_max))

    col1, col2 = st.columns(2)
    with col1:
        propensity_hist = px.histogram(
            plot_data,
            x="Propensity",
            y="Decisions",
            nbins=50,
        )
        propensity_hist.update_yaxes(title_text="Actions")
        propensity_hist.add_vline(
            x=propensity_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Threshold: {propensity_threshold:.2%}",
            annotation_position="top right",
            annotation_font_color="red",
        )
        prop_event = st.plotly_chart(propensity_hist, on_select="rerun", key="prop_chart")

    with col2:
        priority_hist = px.histogram(
            plot_data,
            x="Priority",
            y="Decisions",
            nbins=50,
            log_y=True,
        )
        priority_hist.update_yaxes(title_text="Actions")
        priority_hist.add_vline(
            x=priority_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Threshold: {priority_threshold:.4f}",
            annotation_position="top right",
            annotation_font_color="red",
        )
        prio_event = st.plotly_chart(priority_hist, on_select="rerun", key="prio_chart")


# ---------------------------------------------------------------------------
# Section 3: Distribution of surviving actions above both thresholds
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Offers Meeting Quality Thresholds"

    surviving = (
        apply_filter(da.preaggregated_filter_view, channel_filter)
        .filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
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
