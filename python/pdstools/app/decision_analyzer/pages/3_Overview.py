# python/pdstools/app/decision_analyzer/pages/3_Overview.py
import streamlit as st
from da_streamlit_utils import ensure_data
from pdstools.decision_analyzer.plots import offer_quality_single_pie

# TODO see if we can speed up on first use - it's doing a lot now

ensure_data()
st.session_state["sidebar"] = st.sidebar

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
    method = sample_metadata["method"]
    source_file = sample_metadata.get("source_file", "unknown")

    method_label = "exact" if method == "exact" else "approximate"
    st.info(
        f"📊 This data represents **{sample_pct:.1f}%** of the original dataset ({method_label} calculation). "
        f"Original source: `{source_file}`"
    )

has_arbitration_data = (
    "Arbitration" in st.session_state.decision_data.AvailableNBADStages
    and st.session_state.decision_data.arbitration_stage.collect().height > 0
)

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

    overview = st.session_state.decision_data.get_overview_stats

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
            st.session_state.decision_data.plot.propensity_vs_optionality("Arbitration").update_layout(
                showlegend=False, height=300
            ),
            width="stretch",
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
            width="stretch",
        )
    else:
        st.warning(
            "No actions survive to the arbitration stage in this data set. Sensitivity analysis is not available."
        )

    "## :red[Offer Quality]"

    if has_arbitration_data:
        """
        Shows the distribution of customer interactions by offer quality at the
        arbitration stage. Green indicates customers received relevant offers,
        while red shows customers without offers.
        """

        # Use 5th percentile thresholds (same defaults as Offer Quality page)
        propensity_th = st.session_state.decision_data.getThresholdingData("Propensity", [0, 5, 100])
        priority_th = st.session_state.decision_data.getThresholdingData("Priority", [0, 5, 100])

        prop_values = propensity_th["Threshold"].to_list()
        prio_values = priority_th["Threshold"].to_list()

        if not all(v is None for v in prop_values) and not all(v is None for v in prio_values):
            propensityTH = prop_values[1] if prop_values[1] is not None else 0.05
            priorityTH = prio_values[1] if prio_values[1] is not None else 0.0

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
                    stage="Arbitration",
                    propensityTH=propensityTH,
                    level="Stage Group",
                ),
                use_container_width=True,
            )
        else:
            st.warning("Offer quality analysis requires propensity and priority thresholds.")
    else:
        st.warning(
            "No actions survive to the arbitration stage in this data set. Offer quality analysis is not available."
        )
