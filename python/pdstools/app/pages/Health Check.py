import os
from pathlib import Path

import streamlit as st
import traceback
from pdstools.utils import streamlit_utils

intro, imports, filters, report = st.tabs(
    [
        "Introduction",
        "Data import",
        "Data filters",
        "Report configuration",
    ]
)

with intro:
    """# Introduction"""
    """This app allows you to easily generate the ADM Health Check."""

    """The Health Check gives a generic and global overview 
    of the Adaptive models and predictors using Pega Machine Learning.
    It is mostly written mostly towards a standard Next-Best-Action Designer
    setup within Pega Customer Decision Hub, but also works for non-standard
    setups and custom use-cases, though perhaps not as well."""

    """The Health Check is based on two files: the ADM Model Snapshot
    dataset and the ADM Predictor Binning dataset. These two datasets, 
    part of the so-called "datamart", give a snapshot of the internals of the
    adaptive model. For instructions on where and how to download these files,
    please [refer to the wiki](https://github.com/pegasystems/pega-datascientist-tools/wiki/How-to-export-and-use-the-ADM-Datamart)"""

    """We recommend following the instructions and not customizing the files too much,
    leaving types, names and timestamp formats as they are. 
    We can import the zipped file exports directly, no need to unzip them.
    If exporting the data through a Data Flow, we'd recommend using json as the 
    export file type, as csv uses a different timestamp convention."""

    """If for whatever reason you need to customize the files before using them here, 
    we'd recommend to use a strictly-typed format such as Parquet or Arrow, 
    storing timestamps as their strictly-typed type, but leaving the names as they are.
    """


with imports:
    "# Importing the data"
    st.write(
        """We currently support three ways to upload your own data,
    and one convenient way to test out the Health Check using a CDH Sample dataset. 
    Select an option below to get started, or first configure some advanded options
    in the expanding section below."""
    )

    opts = {}
    with st.expander("Configure advanced options"):
        col1, col2 = st.columns([4, 7])
        with col1:
            opts["extract_keys"] = st.checkbox(
                "Extract additional keys",
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

with filters:
    """# Add custom filters"""

    """You can easily add custom filters to specify the Health Check to your needs.
    In the selectbox below, simply select the columns you wish to filter.
    For each column, a new configuration screen will be added in which you can specify
    what values you want to keep in the Health Check.
    """

    if "dm" in st.session_state:
        st.session_state["filters"] = streamlit_utils.filter_dataframe(
            st.session_state["dm"].modelData
        )
    else:
        st.warning("Please configure your files in the `data import` tab.")


with report:
    "# Generating the Health Check"
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
            True,
            help="""
            Whether to include the overview tables embedded in the document itself
            or to separately recieve these in a tabbed Excel file.""",
        )
    outfile = ""

    if "run" not in st.session_state:
        st.session_state["runID"] = 0
        st.session_state["run"] = {0: {}}

    if st.button("Generate Health Check"):
        st.session_state["runID"] = max(list(st.session_state["run"].keys())) + 1

        with st.spinner("Generating Health Check..."):
            try:
                outfile = (
                    st.session_state["dm"]
                    .applyGlobalQuery(st.session_state["filters"])
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
                        .applyGlobalQuery(st.session_state["filters"])
                        .exportTables(tablename)
                    )
                    st.session_state["run"][st.session_state["runID"]][
                        "tables"
                    ] = tablename
                    st.session_state["run"][st.session_state["runID"]][
                        "tablefile"
                    ] = open(tables, "rb")

                if len(st.session_state["run"][st.session_state["runID"]]) == 0:
                    st.stop()

                col1, col2 = st.columns([1, 1])
                with col1:
                    if "file" in st.session_state["run"][st.session_state["runID"]]:
                        btn = st.download_button(
                            label="Download Health Check",
                            data=st.session_state["run"][st.session_state["runID"]][
                                "file"
                            ],
                            file_name=Path(
                                st.session_state["run"][st.session_state["runID"]][
                                    "name"
                                ]
                            ).name,
                        )
                        if (
                            "tables"
                            in st.session_state["run"][st.session_state["runID"]]
                        ):
                            with col2:
                                btn = st.download_button(
                                    label="Download additional tables",
                                    data=st.session_state["run"][
                                        st.session_state["runID"]
                                    ]["tablefile"],
                                    file_name=st.session_state["run"][
                                        st.session_state["runID"]
                                    ]["tables"],
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