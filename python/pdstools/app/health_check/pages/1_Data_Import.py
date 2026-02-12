import streamlit as st
from pdstools.utils import streamlit_utils

"# Importing the data"
st.write(
    """We currently support three ways to upload your own data,
and one convenient way to test out the Health Check using a CDH Sample dataset.
Select an option below to get started. Advanced options can be configured at the
bottom of this page."""
)

# Initialize session state for import options if not already set
if "extract_pyname_keys" not in st.session_state:
    st.session_state["extract_pyname_keys"] = True
if "infer_schema_length" not in st.session_state:
    st.session_state["infer_schema_length"] = 10000

# Data import section using current session state values
streamlit_utils.import_datamart(
    extract_pyname_keys=st.session_state["extract_pyname_keys"],
    infer_schema_length=st.session_state["infer_schema_length"],
)

# Advanced options at the bottom
st.write("### Advanced options")
with st.expander("Configure import settings"):
    st.session_state["extract_pyname_keys"] = st.checkbox(
        "Extract Treatments",
        value=st.session_state["extract_pyname_keys"],
        help="""By default, ADM has a few "Context Keys" it uses to
        distinguish between models, such as Issue, Group, Channel, or Name.
        However, if you've setup custom context keys that are not part of a regular
        CDH setup, they are embedded in the "pyName" column. Setting this checkbox
        tells us to try to extract these keys into separate columns.""",
    )

    st.session_state["infer_schema_length"] = st.number_input(
        "Schema inference length",
        min_value=1000,
        value=st.session_state["infer_schema_length"],
        step=10000,
        help="""Number of rows to scan when inferring the schema for CSV/JSON files.
        For large production datasets, you may need to increase this value (e.g., 200000)
        if columns are not being detected correctly. Higher values use more memory but
        provide more accurate schema detection.""",
    )

    st.info("💡 Changes to these options will take effect when you re-import the data.")

# Keep only essential session state keys
essential_keys = [
    "dm",
    "params",
    "logger",
    "log_buffer",
    "data_source",
    "prediction",
    "extract_pyname_keys",
    "infer_schema_length",
]
for key in list(st.session_state.keys()):
    if key not in essential_keys and not key.startswith("_"):
        del st.session_state[key]
