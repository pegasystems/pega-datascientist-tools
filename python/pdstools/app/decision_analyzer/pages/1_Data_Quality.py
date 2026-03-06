# python/pdstools/app/decision_analyzer/pages/1_Data_Quality.py
import streamlit as st
from da_streamlit_utils import ensure_data

ensure_data()
st.session_state["sidebar"] = st.sidebar

"# Data Quality"

"""
Review data quality checks and validation warnings for your dataset. This page highlights
potential issues with your data that may affect analysis accuracy or indicate configuration
problems.
"""

# Get the DecisionAnalyzer instance
da = st.session_state.decision_data

# Display validation warnings
warnings_found = False

# Column validation errors
if da.validation_error:
    st.warning(f"**Column Validation Issue:**\n\n{da.validation_error}")
    warnings_found = True

# Propensity validation warnings
if hasattr(da, "propensity_validation_warning") and da.propensity_validation_warning:
    st.warning(da.propensity_validation_warning)
    warnings_found = True

# If no warnings, show success message
if not warnings_found:
    st.success(
        "✅ **No data quality issues detected**\n\n"
        "Your data has passed all validation checks. You can proceed with analysis."
    )

# Show data quality summary
st.markdown("---")
"## Data Quality Summary"

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Extract Type", "v2 (Full Pipeline)" if da.extract_type == "decision_analyzer" else "v1 (Arbitration Only)"
    )

    # Show propensity availability
    if "Propensity" in da.sample.collect_schema().names():
        propensity_stages = len(da.stages_with_propensity) if da.stages_with_propensity else 0
        st.metric("Stages with Propensity", f"{propensity_stages} of {len(da.AvailableNBADStages)}")
    else:
        st.metric("Propensity Data", "Not Available")

with col2:
    # Show completeness metrics
    total_rows = da.decision_data.select("Interaction ID").collect().height
    st.metric("Total Interactions", f"{total_rows:,}")

    # Show stages available
    st.metric("Available Stages", f"{len(da.AvailableNBADStages)}")

# Additional info section
with st.expander("ℹ️ About Data Quality Checks"):
    """
    This page performs the following data quality validations:

    **Column Validation:**
    - Checks that all required columns are present in the dataset
    - Validates column data types match expected schema

    **Propensity Validation:**
    - Detects invalid propensity values (> 1.0) that violate probability constraints
    - Flags unusually high propensities (> 10%) that may indicate:
      - Model calibration issues
      - Different modeling approach than typical marketing use cases
      - Data extraction or transformation problems

    **Future Checks:**
    More validation checks will be added here as the tool evolves, such as:
    - Duplicate interaction detection
    - Temporal consistency checks
    - Stage progression validation
    - Missing value analysis
    """
