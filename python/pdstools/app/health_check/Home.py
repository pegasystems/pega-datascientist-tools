import shutil

import streamlit as st

st.set_page_config(
    page_title="Home",
    menu_items={
        "Report a bug": "https://github.com/pegasystems/pega-datascientist-tools/issues",
        "Get help": "https://pegasystems.github.io/pega-datascientist-tools/Python/examples.html",
    },
)


import pdstools
from pdstools.utils.cdh_utils import setup_logger
from pdstools.utils.streamlit_utils import st_get_latest_pdstools_version

if "log_buffer" not in st.session_state:
    logger, st.session_state.log_buffer = setup_logger()

"""
# Welcome to the Health Check app!

This app helps you analyze ADMDatamart data by allowing you to generate a Health Check document with visuals as well as excel
export for self exploration of the data. 

To be up to date with changes in the app, make sure to regularly run a `pip install --upgrade pdstools`!
"""

pandoc_available = shutil.which("pandoc") is not None

if not pandoc_available:
    st.warning(
        """
    ⚠️ Pandoc is not available on your system. Some features of this app may not work correctly.
    
    Please install Pandoc from https://pandoc.org/installing.html to ensure full functionality.
    """
    )


current_version = pdstools.__version__
latest_version = st_get_latest_pdstools_version()

if latest_version and current_version != latest_version:
    st.warning(
        f"""
    ⚠️ You are using pdstools version {current_version}, but version {latest_version} is available.
    
    To update, run: `pip install --upgrade pdstools`
    """
    )
