import os
from pathlib import Path

import streamlit as st
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

    """The healthcheck gives a generic and global overview 
    of the Adaptive models and predictors using Pega Machine Learning.
    It is mostly written mostly towards a standard Next-Best-Action Designer
    setup within Pega Customer Decision Hub, but also works for non-standard
    setups and custom use-cases, though perhaps not as well."""

    """The healthcheck is based on two files: the ADM Model Snapshot
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
    and one convenient way to test out the healthcheck using a CDH Samle dataset. 
    Select an option below to get started, or first configure some advanded options
    in the expanding section below."""
    )

    opts = {}
    with st.expander("Configure advanced options"):
        col1, col2 = st.columns([4, 7])
        with col1:
            opts["extract_keys"] = st.checkbox(
                "Extract additional keys",
                False,
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

    """You can easily add custom filters to specify the healthcheck to your needs.
    In the selectbox below, simply select the columns you wish to filter.
    For each column, a new configuration screen will be added in which you can specify
    what values you want to keep in the healthcheck.
    """

    if "dm" in st.session_state:
        st.session_state["filters"] = streamlit_utils.filter_dataframe(
            st.session_state["dm"].modelData
        )


with report:
    "# Generating the Health Check"
    """Time to start the Health Check. """
    with st.expander("Health Check options"):
        name = st.text_input("Customer name")
        if name == "":
            name = None
        output_type = st.selectbox("Select output type", ["html"], index=0)
        working_dir = Path(st.text_input("Change working directory", "healthCheckDir"))
        delete_temp_files = st.checkbox("Remove temporary files", True)
    outfile = ""
    if st.button("Generate healthcheck"):
        if "file" in st.session_state:
            del st.session_state["file"]
        with st.spinner("Generating healthcheck..."):
            try:
                outfile = st.session_state["dm"].generateHealthCheck(
                    name=name,
                    output_type=output_type,
                    working_dir=working_dir,
                    output_location=working_dir,
                    delete_temp_files=delete_temp_files,
                    output_to_file=True,
                )
                if os.path.isfile(outfile):
                    st.session_state["file"] = open(outfile, "rb")
            except Exception as e:
                st.error(f"""Error occured when generating healthcheck: {e}""")
                with open(working_dir / "log.txt", "rb") as f:
                    btn = st.download_button(
                        label="Download error log",
                        data=f,
                        file_name="errorlog.txt",
                    )
    if "file" in st.session_state:
        btn = st.download_button(
            label="Download file",
            data=st.session_state["file"],
            file_name=Path(outfile).name,
        )
