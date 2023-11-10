import os
from pathlib import Path

import streamlit as st
from pdstools.utils.streamlit_utils import model_selection_df
import traceback

if "dm" not in st.session_state:
    st.warning("Please configure your files in the `data import` tab.")
    st.stop()

if "data_is_cached" not in st.session_state:
    st.session_state["data_is_cached"] = False

health_check, model_report = st.tabs(
    [
        "Overall Health Check",
        "Individual Model Reports",
    ]
)

with health_check:
    """To begin monitoring your models, you can create a Health Check document that provides a summary of all models and predictors."""
    with st.expander("Health Check options"):
        name = st.text_input("Customer name")
        if name == "":
            name = None
        output_type = st.selectbox("Select output type", ["html"], index=0)
        working_dir = Path(st.text_input("Change working directory", "healthCheckDir"))
        delete_temp_files = st.checkbox("Remove temporary files", True)

    outfile = ""
    hc_col, table_col = st.columns([1, 1])
    if "run" not in st.session_state:
        st.session_state["runID"] = 0
        st.session_state["run"] = {0: {}}
    try:
        with hc_col:
            if st.button("Generate Health Check"):
                st.session_state["runID"] = (
                    max(list(st.session_state["run"].keys())) + 1
                )

                with st.spinner("Generating Health Check..."):
                    outfile = (
                        st.session_state["dm"]
                        .applyGlobalQuery(st.session_state.get("filters", None))
                        .generateReport(
                            name=name,
                            output_type=output_type,
                            working_dir=working_dir,
                            output_location=working_dir,
                            delete_temp_files=delete_temp_files,
                            output_to_file=True,
                            verbose=True,
                        )
                    )
                    if os.path.isfile(outfile):
                        file = open(outfile, "rb")

                    st.session_state["run"][st.session_state["runID"]] = {
                        "name": outfile,
                        "file": file,
                    }

                    if len(st.session_state["run"][st.session_state["runID"]]) == 0:
                        st.stop()
            if "file" in st.session_state["run"][st.session_state["runID"]]:
                btn = st.download_button(
                    label="Download Health Check",
                    data=st.session_state["run"][st.session_state["runID"]]["file"],
                    file_name=Path(
                        st.session_state["run"][st.session_state["runID"]]["name"]
                    ).name,
                )
        with table_col:
            if st.button("Create Tables"):
                with st.spinner("Creating Tables..."):
                    tablename = Path(outfile).name.rsplit(".", 1)[0] + "_Tables.xlsx"
                    tables = (
                        st.session_state["dm"]
                        .applyGlobalQuery(st.session_state.get("filters", None))
                        .exportTables(tablename)
                    )
                    st.session_state["run"][st.session_state["runID"]][
                        "tables"
                    ] = tablename
                    st.session_state["run"][st.session_state["runID"]][
                        "tablefile"
                    ] = open(tables, "rb")

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
        st.error(f"""An error occured: {e}""")
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


with model_report:
    if "working_dir" not in locals():
        working_dir = "healthCheckDir"
    if "model_selection_df" not in st.session_state:
        if "filters" in st.session_state:
            st.session_state["model_selection_df"] = model_selection_df(
                st.session_state["dm"]._apply_query(
                    st.session_state["dm"].combinedData, st.session_state["filters"]
                )
            )
        else:
            st.session_state["model_selection_df"] = model_selection_df(
                st.session_state["dm"].combinedData
            )

    st.write("Please choose the models for which you wish to generate a report")
    edited_df = st.data_editor(
        st.session_state["model_selection_df"],
        disabled=["Configuration", "Channel", "Name"],
    )

    st.session_state["selected_models"] = edited_df.loc[
        edited_df["Generate Report"] == True
    ]["ModelID"].to_list()
    st.write(f"{len(st.session_state['selected_models'])} models are selected")
    if len(st.session_state["selected_models"]) > 0:
        output_location = st.text_input("Path to save models", value=".")
        if st.button("Create Model Report(s) for selected model(s)"):
            row_count_bar = st.progress(0.0)
            with st.spinner("Running Model Reports..."):
                for i, modelid in enumerate(st.session_state["selected_models"]):
                    if i + 1 == len(st.session_state["selected_models"]):
                        del_cache = True
                    else:
                        del_cache = False
                    row_count_bar.progress(
                        value=i / len(st.session_state["selected_models"]),
                        text=f"Generating report for {modelid} ({i+1} / {len(st.session_state['selected_models'])})",
                    )

                    st.session_state["dm"].applyGlobalQuery(
                        st.session_state.get("filters", None)
                    ).generateReport(
                        name="",
                        modelid=modelid,
                        output_location=output_location,
                        working_dir=working_dir,
                        output_type="html",
                        allow_collect=True,
                        cached_data=st.session_state["data_is_cached"],
                        verbose=True,
                        output_to_file=True,
                        del_cache=del_cache,
                    )
                    st.session_state["data_is_cached"] = True
                row_count_bar.progress(value=1.0, text="Finished")
                st.success(
                    f"Your Reports are saved at ***{os.path.abspath(output_location)}*** :open_file_folder:"
                )
                st.balloons()
