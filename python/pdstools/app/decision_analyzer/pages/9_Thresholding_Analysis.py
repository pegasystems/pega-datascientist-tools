# python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py
import plotly.express as px
import polars as pl
import streamlit as st

from da_streamlit_utils import ensure_data

"# Thresholding Analysis"

"""
Explore how applying propensity and priority thresholds affects the volume and
distribution of actions reaching arbitration.

* What is the effect of new offers (new models with propensity 0.5, showing a peak there)?
* What should my propensity threshold be and how does it affect volumes and distributions?
  Filter on a specific channel globally to answer this per channel.
* Is the random control group working? (propensity spreading 0–1)
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
        value=prio_range[0],
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
    "## Threshold Impact"

    st.caption(
        "**Total actions** counts each action reaching arbitration separately "
        "(multiple per interaction). **Decisions without actions** counts interactions "
        "where *no* action survives both thresholds."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Actions", f"{total_action_appearances:,.0f}")
    c2.metric("Above Both Thresholds", f"{above_actions:,.0f}")
    c3.metric("Below Either Threshold", f"{below_actions:,.0f}")
    c4.metric("% Actions Filtered", f"{pct_filtered:.1f}%")
    c5.metric(
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
    "## Propensity and Priority Distributions"

    st.caption(
        "Each bar shows the number of actions in that value bin "
        "(weighted by pre-aggregated counts). The dashed red line marks the threshold."
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
        st.plotly_chart(propensity_hist, use_container_width=True)

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
        st.plotly_chart(priority_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Distribution of surviving actions above both thresholds
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Action Distribution above Thresholds"

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
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Decile charts (static context)
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Decile Overviews"

    st.caption(
        "Each bar shows the count of actions with values below that decile threshold. "
        "The line shows the threshold value at each decile."
    )

    dec_col1, dec_col2 = st.columns(2)
    with dec_col1:
        st.plotly_chart(
            da.plot.threshold_deciles("Propensity", "Propensity"),
            use_container_width=True,
        )
    with dec_col2:
        st.plotly_chart(
            da.plot.threshold_deciles("Priority", "Priority"),
            use_container_width=True,
        )
