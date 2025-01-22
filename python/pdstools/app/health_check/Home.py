import shutil

import streamlit as st

import pdstools
from pdstools.utils.cdh_utils import setup_logger
from pdstools.utils.streamlit_utils import st_get_latest_pdstools_version

st.set_page_config(
    page_title="Home",
    menu_items={
        "Report a bug": "https://github.com/pegasystems/pega-datascientist-tools/issues",
        "Get help": "https://pegasystems.github.io/pega-datascientist-tools/Python/examples.html",
    },
)


if "log_buffer" not in st.session_state:
    logger, st.session_state.log_buffer = setup_logger()

"""
# Welcome to the Health Check app!

This app helps you analyze ADMDatamart data by allowing you to generate a Health Check document with visuals as well as excel
export for self exploration of the data.

To be up to date with changes in the app, make sure to regularly run a `pip install --upgrade pdstools`!
"""

pandoc_available = shutil.which("pandoc") is not None
quarto_available = shutil.which("quarto") is not None

if not (pandoc_available and quarto_available):
    missing_deps = []
    if not pandoc_available:
        missing_deps.append("Pandoc (https://pandoc.org/installing.html)")
    if not quarto_available:
        missing_deps.append("Quarto (https://quarto.org/docs/get-started/)")

    missing_deps_list = "\n    - ".join(missing_deps)
    message = (
        f"⚠️ The following required tools are not available on your system:\n"
        f"    - {missing_deps_list}\n"
        "The app will not function without these tools. Please install them before proceeding."
    )
    st.error(message)


current_version = pdstools.__version__
latest_version = st_get_latest_pdstools_version()

if latest_version and current_version != latest_version:
    st.warning(
        f"""
    ⚠️ You are using pdstools version {current_version}, but version {latest_version} is available.

    To update, run: `pip install --upgrade pdstools`
    """
    )
