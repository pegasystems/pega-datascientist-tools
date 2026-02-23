# python/pdstools/app/decision_analyzer/da_streamlit_utils.py
import os
from pathlib import Path
from typing import List, Optional

import polars as pl
import streamlit as st

from pdstools.decision_analyzer.data_read_utils import (
    read_data,
    read_nested_zip_files,
)
from pdstools.pega_io.File import read_ds_export

from pdstools.decision_analyzer.plots import plot_priority_component_distribution
from pdstools.utils.streamlit_utils import (
    ensure_session_data,
    get_current_index,  # noqa: F401 — re-exported for backward compat
    is_managed_deployment,
)

# Default sample data path for EC2 deployments. Override with
# PDSTOOLS_SAMPLE_DATA_PATH env var if needed.
_EC2_SAMPLE_PATH = os.environ.get(
    "PDSTOOLS_SAMPLE_DATA_PATH", "/s3-files/anonymized/anonymized"
)


def ensure_data():
    """Guard: stop if decision data is not loaded."""
    ensure_session_data("decision_data", "Please upload your data in the Home page.")


def ensure_funnel():
    if st.session_state.decision_data.extract_type == "explainability_extract":
        st.warning(
            "This page requires **Action Analysis (v2)** data with full stage "
            "pipeline information. Explainability Extract (v1) data only contains "
            "the arbitration stage and cannot show the decision funnel."
        )
        st.stop()


def ensure_getFilterComponentData():
    return (
        "Component Name"
        in st.session_state.decision_data.decision_data.collect_schema().names()
    )


# st.elements.utils._shown_default_value_warning = (
#     True  # to suppress default val+key warning in date filter
# )
polars_lazyframe_hashing = {
    pl.LazyFrame: lambda x: hash(x.explain(optimized=False)),
    pl.Expr: lambda x: str(x.inspect()),
}


def show_filtered_counts(statsBefore, statsAfter):
    keys = set().union(statsBefore.keys(), statsAfter.keys())
    for k in keys:
        st.progress(
            (statsAfter[k] / statsBefore[k]),
            text=f"{statsAfter[k]} {k} remaining from {statsBefore[k]}",
        )


def _clean_unselected_filters(to_filter_columns, filter_type):
    keys_to_remove = []
    for key in st.session_state.keys():
        if key.__contains__("selected_"):
            column_name = key.split("selected_", 1)[1]
            if column_name not in to_filter_columns:
                keys_to_remove.append(column_name)
    for column in keys_to_remove:
        selected_key = f"{filter_type}selected_{column}"
        _selected_key = f"{filter_type}_selected_{column}"
        regexselected_key = f"{filter_type}regexselected_{column}"
        regex_selected_key = f"{filter_type}regex_selected_{column}"
        categories_key = f"{filter_type}categories_{column}"
        for val in [
            selected_key,
            _selected_key,
            categories_key,
            regexselected_key,
            regex_selected_key,
        ]:
            if val in st.session_state:
                del st.session_state[val]


def get_data_filters(
    df: pl.LazyFrame, columns=None, queries=None, filter_type="local"
) -> List[
    pl.Expr
]:  # this one is way too complex, should be split up into probably 5 functions
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Parameters
    ----------
    df : pl.DataFrame
        Original dataframe
    """

    def _save_selected(
        filter_type, column, regex=""
    ):  ## see the issue on why we need to save a different session.state variable https://discuss.streamlit.io/t/session-state-is-not-preserved-when-navigating-pages/48787
        st.session_state[f"{filter_type}{regex}selected_{column}"] = st.session_state[
            f"{filter_type}{regex}_selected_{column}"
        ]

    def _save_multiselect():
        st.session_state[f"{filter_type}multiselect"] = st.session_state[
            f"{filter_type}_multiselect"
        ]

    if columns is None:
        columns = df.collect_schema().names()
    if queries is None:
        queries = []

    st.session_state[f"{filter_type}_multiselect"] = (
        st.session_state[f"{filter_type}multiselect"]
        if f"{filter_type}multiselect" in st.session_state
        else []
    )
    to_filter_columns = st.multiselect(
        "Filter data on",
        columns,
        key=f"{filter_type}_multiselect",
        on_change=_save_multiselect,
    )
    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        left.write("## ↳")

        # Treat columns with < 20 unique values as categorical
        col_dtype = df.collect_schema()[column]
        if (col_dtype == pl.Categorical) or (col_dtype == pl.Utf8):
            if f"{filter_type}categories_{column}" not in st.session_state.keys():
                st.session_state[f"{filter_type}categories_{column}"] = (
                    df.select(pl.col(column).unique()).collect().to_series().to_list()
                )
            if f"{filter_type}_selected_{column}" not in st.session_state.keys():
                st.session_state[f"{filter_type}_selected_{column}"] = (
                    st.session_state[f"{filter_type}categories_{column}"]
                    if f"{filter_type}selected_{column}" not in st.session_state
                    else st.session_state[f"{filter_type}selected_{column}"]
                )
            if len(st.session_state[f"{filter_type}categories_{column}"]) < 200:
                options = st.session_state[f"{filter_type}categories_{column}"]
                default_selected = (
                    st.session_state[f"{filter_type}selected_{column}"]
                    if f"{filter_type}selected_{column}" in st.session_state
                    else options
                )
                st.session_state[f"{filter_type}_selected_{column}"] = default_selected
                selected = right.multiselect(
                    f"Values for {column}",
                    options=options,
                    key=f"{filter_type}_selected_{column}",
                    on_change=_save_selected,
                    kwargs={"filter_type": filter_type, "column": column},
                )
                if selected != st.session_state[f"{filter_type}categories_{column}"]:
                    queries.append(
                        pl.col(column)
                        .cast(pl.Utf8)
                        .is_in(st.session_state[f"{filter_type}selected_{column}"])
                    )

            else:
                key_to_delete = f"{filter_type}_selected_{column}"
                if key_to_delete in st.session_state:
                    del st.session_state[key_to_delete]
                default_selected = (
                    st.session_state[f"{filter_type}regexselected_{column}"]
                    if f"{filter_type}regexselected_{column}" in st.session_state
                    else ""
                )
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    value=default_selected,
                    key=f"{filter_type}regex_selected_{column}",
                    on_change=_save_selected,
                    kwargs={
                        "filter_type": filter_type,
                        "column": column,
                        "regex": "regex",
                    },
                )
                if user_text_input:
                    queries.append(pl.col(column).str.contains(user_text_input))

        elif col_dtype.is_numeric():
            min_col, max_col = right.columns((1, 1))
            _min = float(df.select(pl.min(column)).collect().item())
            _max = float(df.select(pl.max(column)).collect().item())
            if f"{filter_type}selected_{column}" not in st.session_state:
                default_min, default_max = _min, _max
            else:
                default_min, default_max = st.session_state[
                    f"{filter_type}selected_{column}"
                ]
            if _max - _min <= 200:
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(default_min, default_max),
                )
            else:
                user_min = min_col.number_input(
                    label=f"Min value for {column} (Min:{_min})",
                    min_value=_min,
                    max_value=_max,
                    value=default_min,
                )
                user_max = max_col.number_input(
                    label=f"Max value for {column} (Max:{_max})",
                    min_value=_min,
                    max_value=_max,
                    value=default_max,
                )
                user_num_input = [user_min, user_max]
            st.session_state[f"{filter_type}selected_{column}"] = user_num_input
            if user_num_input[0] != _min or user_num_input[1] != _max:
                queries.append(pl.col(column).is_between(*user_num_input))
        elif col_dtype.is_temporal():
            value = (
                df.select(pl.min(column)).collect().item(),
                df.select(pl.max(column)).collect().item(),
            )
            st.session_state[f"{filter_type}_selected_{column}"] = (
                (st.session_state[f"{filter_type}selected_{column}"])
                if f"{filter_type}selected_{column}" in st.session_state
                else value
            )
            user_date_input = right.date_input(
                f"Values for {column}",
                value=value,
                key=f"{filter_type}_selected_{column}",
                on_change=_save_selected,
                kwargs={"filter_type": filter_type, "column": column},
            )
            if len(user_date_input) == 2:
                queries.append(pl.col(column).is_between(*user_date_input))
    _clean_unselected_filters(to_filter_columns, filter_type)
    return queries


def get_options() -> List[str]:
    """Data source options.

    'File path' is only shown in managed deployments where users need to
    reference server-side paths (e.g. S3-mounted directories). For local use,
    the file uploader's browse button is sufficient.
    """
    options = ["Sample data", "File upload"]
    if is_managed_deployment():
        options.append("File path")
    return options


def handle_sample_data() -> Optional[pl.LazyFrame]:
    """Load sample data, using a local S3 path in managed deployments."""
    if is_managed_deployment():
        return read_data(Path(_EC2_SAMPLE_PATH))

    # Prefer local file when available (e.g. during development), fall back
    # to downloading from GitHub for installed-package users.
    local_path = Path(__file__).resolve().parents[4] / "data" / "sample_eev2.parquet"
    if local_path.exists():
        return pl.scan_parquet(local_path)

    return read_ds_export(
        filename="sample_eev2.parquet",
        path="https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data",
    )


def _read_uploaded_zip(file_buffer) -> pl.LazyFrame:
    """Read a zip file, handling both nested-gzip (EEV2) and flat archive layouts."""
    import tempfile
    import zipfile

    with zipfile.ZipFile(file_buffer, "r") as zf:
        inner_names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
        inner_exts = {Path(n).suffix.lower() for n in inner_names if Path(n).suffix}

        # Reject archives with only unsupported file types
        supported_exts = {".csv", ".parquet", ".json", ".ndjson", ".arrow", ".zip"}
        if not inner_exts.intersection(supported_exts):
            raise ValueError(
                f"The uploaded archive does not contain recognizable data files. "
                f"Found: {', '.join(sorted(inner_exts))}. "
                f"Expected raw decision data in csv, parquet, json, or arrow format."
            )

        # If the zip contains .zip files, use the legacy gzipped-ndjson reader
        if ".zip" in inner_exts:
            file_buffer.seek(0)
            return read_nested_zip_files(file_buffer)

        # Otherwise extract to a temp directory and use read_data
        tmp_dir = tempfile.mkdtemp(prefix="da_upload_")
        zf.extractall(tmp_dir)

    return read_data(tmp_dir)


def _read_uploaded_tar(file_buffer) -> pl.LazyFrame:
    """Extract a tar/tar.gz/tgz archive to a temp directory and read its contents."""
    import tarfile
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="da_tar_")
    with tarfile.open(fileobj=file_buffer, mode="r:*") as tf:
        tf.extractall(tmp_dir, filter="data")
    return read_data(tmp_dir)


def handle_file_upload() -> Optional[pl.LazyFrame]:
    """Show file uploader accepting one or more files and return a LazyFrame, or None."""
    import tempfile

    uploaded_files = st.file_uploader(
        "Upload decision data",
        type=["zip", "parquet", "json", "csv", "arrow", "gz", "tgz", "tar"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        return None

    frames: list[pl.LazyFrame] = []
    for f in uploaded_files:
        name_lower = f.name.lower()
        suffix = Path(f.name).suffix.lower()
        if suffix == ".parquet":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
                tmp.write(f.getbuffer())
                frames.append(pl.scan_parquet(tmp.name))
        elif suffix == ".zip":
            frames.append(_read_uploaded_zip(f))
        elif (
            name_lower.endswith(".tar.gz")
            or name_lower.endswith(".tgz")
            or suffix == ".tar"
        ):
            frames.append(_read_uploaded_tar(f))
        elif suffix in {".json", ".ndjson"}:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getbuffer())
                frames.append(pl.scan_ndjson(tmp.name))
        elif suffix == ".csv":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(f.getbuffer())
                frames.append(pl.scan_csv(tmp.name))
        elif suffix == ".arrow":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".arrow") as tmp:
                tmp.write(f.getbuffer())
                frames.append(pl.scan_ipc(tmp.name))
        elif suffix == ".gz":
            # .gz that isn't .tar.gz — try as gzipped ndjson
            import gzip

            with tempfile.NamedTemporaryFile(delete=False, suffix=".ndjson") as tmp:
                tmp.write(gzip.decompress(f.getbuffer()))
                frames.append(pl.scan_ndjson(tmp.name))

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal", rechunk=True)


def handle_file_path() -> Optional[pl.LazyFrame]:
    """Show text input for a file/folder path and return a LazyFrame, or None."""
    st.write("Point the app to a file (zip, parquet, csv, …) or a partitioned folder.")
    path = st.text_input(
        "File or partitioned folder path",
        placeholder="/path/to/data",
    )
    if not path:
        return None
    p = Path(path)
    # Single zip file: extract first, then read the extracted contents
    if p.is_file() and p.suffix.lower() == ".zip":
        import tempfile
        import zipfile

        tmp_dir = tempfile.mkdtemp(prefix="da_path_")
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp_dir)
        return read_data(tmp_dir)
    return read_data(path)


@st.cache_resource
def load_decision_analyzer(
    _raw_data: pl.LazyFrame,
    level: str,
    sample_size: int,
    data_fingerprint: str = "",
):
    """Cache the DecisionAnalyzer instance so it persists across page navigations.

    Uses @st.cache_resource because DecisionAnalyzer is a stateful object
    (not serializable). The leading underscore on _raw_data tells Streamlit
    not to hash it (LazyFrames are not hashable by default).

    data_fingerprint is derived from the LazyFrame's query plan so the cache
    busts when different data is loaded.
    """
    from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer

    return DecisionAnalyzer(_raw_data, level=level, sample_size=sample_size)


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def st_priority_component_distribution(
    value_data: pl.LazyFrame, component, granularity
):
    return plot_priority_component_distribution(value_data, component, granularity)
