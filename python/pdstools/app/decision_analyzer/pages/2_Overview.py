import polars as pl
import streamlit as st
from pdstools.app.decision_analyzer.da_streamlit_utils import (
    collect_page_filters,
    ensure_data,
    polars_lazyframe_hashing,
)

from pdstools.decision_analyzer.plots import offer_quality_single_pie
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Overview · Decision Analysis")

ensure_data()
st.session_state["sidebar"] = st.sidebar

# Ensure we're using Stage Group level for Overview analyses
# (other pages may have changed the level to "Stage")
da = st.session_state.decision_data
if da.level != "Stage Group" and "Stage Group" in da.available_levels:
    da.set_level("Stage Group")

"# Overview"

"""
Key metrics and insights about your offer strategy at a glance. See how many offers
reach customers, which factors drive decisions, and where opportunities exist to
improve customer reach.
"""

# Display sample information if available
sample_metadata = st.session_state.get("sample_metadata")
if sample_metadata:
    sample_pct = sample_metadata["sample_percentage"]
    source_file = sample_metadata.get("source_file", "unknown")

    if sample_pct < 100.0:
        st.info(
            f"📊 This data represents **{sample_pct:.2f}%** of the original dataset. Original source: `{source_file}`"
        )

# Find the best stage for overview analyses
da = st.session_state.decision_data
best_stage_for_overview = None
stages_with_prop = da.stages_with_propensity if da.stages_with_propensity else []

# Prefer ActionPropensity stage, otherwise use first stage with propensities
if "ActionPropensity" in stages_with_prop:
    best_stage_for_overview = "ActionPropensity"
elif stages_with_prop:
    best_stage_for_overview = stages_with_prop[0]
elif "Arbitration" in da.AvailableNBADStages:
    # Fallback to Arbitration if no propensity stages available
    best_stage_for_overview = "Arbitration"

# ---------------------------------------------------------------------------
# Cached helpers — expensive per-stage computations are memoised so rerenders
# (filter interactions, page navigations) don't re-run heavy .collect() calls.
# ---------------------------------------------------------------------------


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def _check_stage_has_data(best_stage: str, level: str) -> bool:
    """Return True when *best_stage* has at least one row in the sample."""
    return st.session_state.decision_data.sample.filter(pl.col(level) == best_stage).limit(1).collect().height > 0


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def _propensity_vs_optionality_plot(best_stage: str, level: str):
    return st.session_state.decision_data.plot.propensity_vs_optionality(best_stage).update_layout(
        showlegend=False, height=300
    )


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def _sensitivity_plot(level: str, *page_filter_exprs: pl.Expr):
    _da = st.session_state.decision_data
    filters = list(page_filter_exprs)
    total_decisions = _da.filtered(filters).select(pl.n_unique("Interaction ID")).collect().item()
    return _da.plot.sensitivity(
        win_rank=1,
        hide_priority=True,
        total_decisions=total_decisions,
    ).update_layout(height=300)


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def _offer_quality_pie(best_stage: str, level: str):
    """Compute thresholds, action counts, quality breakdown and pie chart in one cached call."""
    _da = st.session_state.decision_data
    # Use 10th percentile thresholds (same defaults as Offer Quality page)
    propensity_th = _da.scoring.get_thresholding_data("Propensity", [0, 10, 100])
    priority_th = _da.scoring.get_thresholding_data("Priority", [0, 10, 100])

    prop_values = propensity_th["Threshold"].to_list()
    prio_values = priority_th["Threshold"].to_list()

    if all(v is None for v in prop_values) or all(v is None for v in prio_values):
        return None

    propensity_th = prop_values[1] if prop_values[1] is not None else 0.10
    priority_th = prio_values[0] if prio_values[0] is not None else 0.0

    action_counts = _da.aggregates.filtered_action_counts(
        groupby_cols=["Stage Group", "Interaction ID"],
        priority_th=priority_th,
        propensity_th=propensity_th,
    )
    quality_data = _da.aggregates.get_offer_quality(action_counts, group_by="Interaction ID")
    return offer_quality_single_pie(
        quality_data,
        stage=best_stage,
        propensity_th=propensity_th,
        level="Stage Group",
    )


# ---------------------------------------------------------------------------

# Check if we have data at the selected stage (cached: avoids .collect() on every rerender)
has_arbitration_data = False
if best_stage_for_overview:
    has_arbitration_data = _check_stage_has_data(best_stage_for_overview, da.level)

col1, col2 = st.columns(2)

with col1:
    "## :green[Source Data]"

    extract_type = st.session_state.decision_data.extract_type
    format_label = (
        "Explainability Extract (v1) — arbitration stage only"
        if extract_type == "explainability_extract"
        else "Action Analysis / EEV2 (v2) — full pipeline"
    )
    st.caption(f"Data format: **{format_label}**")

    overview = st.session_state.decision_data.overview_stats

    f"""
    In total, there are **{overview["Actions"]} actions** available in **{overview["Channels"]} channels**. The data
    was recorded over **{overview["Duration"].days} days** from **{overview["StartDate"]}** where **{overview["Decisions"]} decisions**
    (**{round(overview["Decisions"] / overview["Duration"].days)}** decisions per day) were made for
    in total **{overview["Customers"]} different customers**.

    """

    "## :blue[Customer Choice]"

    if has_arbitration_data:
        """
        Shows how many offers reach customers and how likely they are to respond. More
        offers typically means higher engagement.
        """
        st.plotly_chart(
            _propensity_vs_optionality_plot(best_stage_for_overview, da.level),
        )
    else:
        st.warning(
            "No actions survive to the arbitration stage in this data set. Optionality analysis is not available."
        )

with col2:
    "## :orange[What Drives Your Offers]"

    if has_arbitration_data:
        """
        See which factors influence which offers reach customers. Customer likelihood to
        respond (propensity) should typically drive decisions for a customer-centric
        approach, balanced with business value.
        """
        st.plotly_chart(
            _sensitivity_plot(da.level, *collect_page_filters()),
        )
    else:
        st.warning(
            "No actions survive to the arbitration stage in this data set. Sensitivity analysis is not available."
        )

    "## :red[Offer Quality]"

    if has_arbitration_data:
        """
        Shows the distribution of customer interactions by offer quality at the
        arbitration stage. Green indicates customers received relevant offers (propensity
        above 10th percentile), while red shows customers without offers. Orange shows
        customers with only irrelevant offers (propensity below 10th percentile).
        """
        pie_fig = _offer_quality_pie(best_stage_for_overview, da.level)
        if pie_fig is not None:
            st.plotly_chart(pie_fig)
        else:
            st.warning("Offer quality analysis requires propensity and priority thresholds.")
    else:
        st.warning(
            "No actions survive to the arbitration stage in this data set. Offer quality analysis is not available."
        )
