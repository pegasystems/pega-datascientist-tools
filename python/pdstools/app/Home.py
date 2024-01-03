import streamlit as st

st.set_page_config(
    page_title="Home",
    menu_items={
        "Report a bug": "https://github.com/pegasystems/pega-datascientist-tools/issues",
        "Get help": "https://pegasystems.github.io/pega-datascientist-tools/Python/examples.html",
    },
)
"""
# Welcome to the pdstools app!

This app helps you analyze ADMDatamart data by allowing you to generate a Health Check document with visuals as well as excel
export for self exploration of the data. 

To be up to date with changes in the app, make sure to regularly run a `pip install --upgrade pdstools`!
"""
