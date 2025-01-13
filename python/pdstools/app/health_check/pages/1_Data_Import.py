import streamlit as st
from pdstools.utils import streamlit_utils

"# Importing the data"
st.write(
    """We currently support three ways to upload your own data,
and one convenient way to test out the Health Check using a CDH Sample dataset.
Select an option below to get started, or first configure some advanded options
in the expanding section below."""
)

with st.expander("Configure advanced options", expanded=True):
    extract_pyname_keys = st.checkbox(
        "Extract Treatments",
        True,
        help="""By default, ADM has a few "Context Keys" it uses to
        distinguish between models, such as Issue, Group, Channel, or Name.
        However, if you've setup custom context keys that are not part of a regular
        CDH setup, they are embedded in the "pyName" column. Setting this checkbox
        tells us to try to extract these keys into separate columns.""",
    )

streamlit_utils.import_datamart(extract_pyname_keys=extract_pyname_keys)

# Keep only essential session state keys
essential_keys = ["dm", "params", "logger", "log_buffer", "data_source"]
for key in list(st.session_state.keys()):
    if key not in essential_keys and not key.startswith("_"):
        del st.session_state[key]
