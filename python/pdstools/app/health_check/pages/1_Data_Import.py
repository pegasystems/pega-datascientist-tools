import streamlit as st
from pdstools.utils import streamlit_utils

"# Importing the data"
st.write(
    """We currently support three ways to upload your own data,
and one convenient way to test out the Health Check using a CDH Sample dataset. 
Select an option below to get started, or first configure some advanded options
in the expanding section below."""
)

opts = {}
with st.expander("Configure advanced options", expanded=True):
    opts["extract_keys"] = st.checkbox(
        "Extract Treatments",
        True,
        help="""By default, ADM has a few "Context Keys" it uses to 
        distinguish between models, such as Issue, Group, Channel, or Name. 
        However, if you've setup custom context keys that are not part of a regular 
        CDH setup, they are embedded in the "pyName" column. Setting this checkbox
        tells us to try to extract these keys into separate columns.""",
    )


streamlit_utils.import_datamart(**opts)
for key in st.session_state.keys():
    if key not in ["dm", "params", "logger", "log_buffer"]:
        del st.session_state[key]
