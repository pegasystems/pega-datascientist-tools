import os
import streamlit as st
from pdstools.utils.streamlit_utils import convert_df
import polars as pl
from pathlib import Path
import traceback


"""
# Download Excel for Custom Analysis
You will see some tables we have generated out of the box in the Health Check. 
If you want to further analyze the data you can download the last snapshot of your 
ADMDatamart in an excel file for custom analysis. As well as our out-of-the-box summary
tables

"""
working_dir = Path("healthCheckDir")
st.write("#### Last Snasphot")
try:
    with st.expander("Downlaod Last Snapshot", expanded=True):
        name = st.text_input("File Name", "ADM Last Snapshot")
        if name == "":
            name = "LastSnapshot"

        if "last_snapshot" not in st.session_state:
            st.session_state["runID"] = 0
            st.session_state["last_snapshot"] = {0: {}}

        if st.button("Filter Last Snapshot"):
            st.session_state["runID"] = (
                max(list(st.session_state["last_snapshot"].keys())) + 1
            )
            st.session_state["last_snapshot"][st.session_state["runID"]] = {
                "name": f"{name}.xlsx"
            }
            with st.spinner("Creating Last Snapshot Excel..."):
                last_snapshot = (
                    st.session_state["dm"]
                    .applyGlobalQuery(st.session_state.get("filters", None))
                    .last(table="combinedData")
                    .filter(pl.col("PredictorName") != "Classifier")
                ).write_excel(
                    st.session_state["last_snapshot"][st.session_state["runID"]]["name"]
                )

                st.session_state["last_snapshot"][st.session_state["runID"]][
                    "last_snapshot"
                ] = open(
                    st.session_state["last_snapshot"][st.session_state["runID"]][
                        "name"
                    ],
                    "rb",
                )

                st.download_button(
                    label="Download Last Snapshot",
                    data=st.session_state["last_snapshot"][st.session_state["runID"]][
                        "last_snapshot"
                    ],
                    file_name=st.session_state["last_snapshot"][
                        st.session_state["runID"]
                    ]["name"],
                )


except Exception as e:
    st.error(f"""Error occured when creating last snapshot file: {e}""")
    traceback_str = traceback.format_exc()
    with open(working_dir / "log.txt", "a") as f:
        f.write(traceback_str)
    with open(working_dir / "log.txt", "rb") as f:
        btn = st.download_button(
            label="Download error log",
            data=f,
            file_name="errorlog.txt",
        )
    os.remove(st.session_state["last_snapshot"][st.session_state["runID"]]["name"])

st.write("#### Summary Tables")

try:
    with st.expander("Downlaod Summary Tables", expanded=True):
        name = "SummaryTables"
        name = st.text_input(
            "File Name",
            value=name,
        )

        if "ExportTables" not in st.session_state:
            st.session_state["runID"] = 0
            st.session_state["ExportTables"] = {0: {}}

        if st.button("Create Summary Tables"):
            st.session_state["runID"] = (
                max(list(st.session_state["ExportTables"].keys())) + 1
            )
            st.session_state["ExportTables"][st.session_state["runID"]] = {
                "name": f"{name}.xlsx"
            }
            with st.spinner("Creating Summary Tables..."):
                last_snapshot = (
                    st.session_state["dm"].applyGlobalQuery(
                        st.session_state.get("filters", None)
                    )
                ).exportTables(
                    st.session_state["ExportTables"][st.session_state["runID"]]["name"]
                )

                st.session_state["ExportTables"][st.session_state["runID"]][
                    "tables"
                ] = open(
                    st.session_state["ExportTables"][st.session_state["runID"]]["name"],
                    "rb",
                )

                st.download_button(
                    label="Download Summary Tables",
                    data=st.session_state["ExportTables"][st.session_state["runID"]][
                        "tables"
                    ],
                    file_name=st.session_state["ExportTables"][
                        st.session_state["runID"]
                    ]["name"],
                )
except Exception as e:
    st.error(f"""Error occured when creating summary tables: {e}""")
    traceback_str = traceback.format_exc()
    with open(working_dir / "log.txt", "a") as f:
        f.write(traceback_str)
    with open(working_dir / "log.txt", "rb") as f:
        btn = st.download_button(
            label="Download error log",
            data=f,
            file_name="errorlog.txt",
        )
    os.remove(st.session_state["ExportTables"][st.session_state["runID"]]["name"])
