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
    col1, col2 = st.columns([4, 7])
    with col1:
        opts["extract_keys"] = st.checkbox(
            "Extract Treatments",
            True,
            help="""By default, ADM has a few "Context Keys" it uses to 
            distinguish between models, such as Issue, Group, Channel, or Name. 
            However, if you've setup custom context keys that are not part of a regular 
            CDH setup, they are embedded in the "pyName" column. Setting this checkbox
            tells us to try to extract these keys into separate columns.""",
        )
    with col2:
        default_keys = ["Channel", "Direction", "Issue", "Group"]
        if opts["extract_keys"]:
            default_keys = default_keys + ["Treatment"]
        opts["context_keys"] = st.multiselect(
            "Select the keys used as context keys", default_keys, default_keys
        )

streamlit_utils.import_datamart(**opts)
for key in st.session_state.keys():
    if key not in ["dm", "params"]:
        del st.session_state[key]
