import os
from pathlib import Path

import streamlit as st
import traceback

if "dm" not in st.session_state:
    st.warning("Please configure your files in the `data import` tab.")
    st.stop()

"""Time to start the Health Check. """
with st.expander("Health Check options"):
    name = st.text_input("Customer name")
    if name == "":
        name = None
    output_type = st.selectbox("Select output type", ["html"], index=0)
    working_dir = Path(st.text_input("Change working directory", "healthCheckDir"))
    delete_temp_files = st.checkbox("Remove temporary files", True)
    include_tables = st.checkbox(
        "Include tables in document",
        False,
        help="""
        Whether to include the overview tables embedded in the document itself
        or to separately recieve these in a tabbed Excel file.""",
    )
outfile = ""

if "run" not in st.session_state:
    st.session_state["runID"] = 0
    st.session_state["run"] = {0: {}}
try:
    if st.button("Generate Health Check"):
        st.session_state["runID"] = max(list(st.session_state["run"].keys())) + 1

        with st.spinner("Generating Health Check..."):
            outfile = (
                st.session_state["dm"]
                .applyGlobalQuery(st.session_state.get("filters", None))
                .generateHealthCheck(
                    name=name,
                    output_type=output_type,
                    working_dir=working_dir,
                    output_location=working_dir,
                    delete_temp_files=delete_temp_files,
                    include_tables=include_tables,
                    output_to_file=True,
                    modelData_only=st.session_state["modelhc"],
                    verbose=True,
                )
            )
            if os.path.isfile(outfile):
                file = open(outfile, "rb")

            st.session_state["run"][st.session_state["runID"]] = {
                "name": outfile,
                "file": file,
            }

            if not include_tables:
                tablename = Path(outfile).name.rsplit(".", 1)[0] + "_Tables.xlsx"
                tables = (
                    st.session_state["dm"]
                    .applyGlobalQuery(st.session_state.get("filters", None))
                    .exportTables(tablename)
                )
                st.session_state["run"][st.session_state["runID"]]["tables"] = tablename
                st.session_state["run"][st.session_state["runID"]]["tablefile"] = open(
                    tables, "rb"
                )

            if len(st.session_state["run"][st.session_state["runID"]]) == 0:
                st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        if "file" in st.session_state["run"][st.session_state["runID"]]:
            btn = st.download_button(
                label="Download Health Check",
                data=st.session_state["run"][st.session_state["runID"]]["file"],
                file_name=Path(
                    st.session_state["run"][st.session_state["runID"]]["name"]
                ).name,
            )
            if "tables" in st.session_state["run"][st.session_state["runID"]]:
                with col2:
                    btn = st.download_button(
                        label="Download additional tables",
                        data=st.session_state["run"][st.session_state["runID"]][
                            "tablefile"
                        ],
                        file_name=st.session_state["run"][st.session_state["runID"]][
                            "tables"
                        ],
                    )

except Exception as e:
    st.error(f"""Error occured when generating healthcheck: {e}""")
    traceback_str = traceback.format_exc()
    with open(working_dir / "log.txt", "a") as f:
        f.write(traceback_str)
    with open(working_dir / "log.txt", "rb") as f:
        btn = st.download_button(
            label="Download error log",
            data=f,
            file_name="errorlog.txt",
        )

    for filename in os.listdir(working_dir):
        file_path = os.path.join(working_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
