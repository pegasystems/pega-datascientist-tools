# python/pdstools/app/decision_analyzer/pages/3_Overview.py
import polars as pl
import streamlit as st
from da_streamlit_utils import ensure_data
from pdstools.decision_analyzer.plots import offer_quality_single_pie

# TODO see if we can speed up on first use - it's doing a lot now

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

    st.info(f"📊 This data represents **{sample_pct:.2f}%** of the original dataset. Original source: `{source_file}`")

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

# Check if we have data at the selected stage
has_arbitration_data = False
if best_stage_for_overview:
    stage_data = da.sample.filter(pl.col(da.level) == best_stage_for_overview).collect()
    has_arbitration_data = stage_data.height > 0

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
            st.session_state.decision_data.plot.propensity_vs_optionality(best_stage_for_overview).update_layout(
                showlegend=False, height=300
            ),
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
            st.session_state.decision_data.plot.sensitivity(
                win_rank=1,
                hide_priority=True,
            ).update_layout(
                height=300,
            ),
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

        # Use 10th percentile thresholds (same defaults as Offer Quality page)
        propensity_th = st.session_state.decision_data.get_thresholding_data("Propensity", [0, 10, 100])
        priority_th = st.session_state.decision_data.get_thresholding_data("Priority", [0, 10, 100])

        prop_values = propensity_th["Threshold"].to_list()
        prio_values = priority_th["Threshold"].to_list()

        if not all(v is None for v in prop_values) and not all(v is None for v in prio_values):
            propensityTH = prop_values[1] if prop_values[1] is not None else 0.10
            priorityTH = prio_values[0] if prio_values[0] is not None else 0.0

            action_counts = st.session_state.decision_data.filtered_action_counts(
                groupby_cols=["Stage Group", "Interaction ID"],
                priorityTH=priorityTH,
                propensityTH=propensityTH,
            )

            quality_data = st.session_state.decision_data.get_offer_quality(
                action_counts,
                group_by="Interaction ID",
            )

            st.plotly_chart(
                offer_quality_single_pie(
                    quality_data,
                    stage=best_stage_for_overview,
                    propensityTH=propensityTH,
                    level="Stage Group",
                ),
            )
        else:
            st.warning("Offer quality analysis requires propensity and priority thresholds.")
    else:
        st.warning(
            "No actions survive to the arbitration stage in this data set. Offer quality analysis is not available."
        )
