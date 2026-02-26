# python/pdstools/app/decision_analyzer/da_streamlit_utils.py
from __future__ import annotations
from pathlib import Path

import polars as pl
import streamlit as st

from pdstools.decision_analyzer.data_read_utils import (
    read_data,
    read_nested_zip_files,
)
from pdstools.pega_io.File import read_ds_export

from pdstools.decision_analyzer.plots import (
    plot_component_overview,
    plot_priority_component_distribution,
)
from pdstools.utils.streamlit_utils import (
    _apply_sidebar_logo,
    ensure_session_data,
    get_current_index,  # noqa: F401 — re-exported for backward compat
    get_data_path,
    is_managed_deployment,
)


def ensure_data():
    """Guard: stop if decision data is not loaded."""
    _apply_sidebar_logo()
    ensure_session_data("decision_data", "Please upload your data in the Home page.")


def _apply_stage_level():
    """Callback: sync the DA level with the radio widget value."""
    da = st.session_state.decision_data
    da.set_level(st.session_state["_stage_level_radio"])


def stage_level_selector():
    """Show a radio toggle for stage granularity when multiple levels are available.

    Calls ``set_level`` on the DecisionAnalyzer instance when the user
    switches. Only renders the widget for v2 data that has both
    "Stage Group" and "Stage" columns; for v1 data this is a no-op.
    """
    da = st.session_state.decision_data
    levels = da.available_levels
    if len(levels) <= 1:
        return
    current = da.level if da.level in levels else levels[0]
    st.radio(
        "Stage Granularity",
        options=levels,
        index=levels.index(current),
        horizontal=True,
        key="_stage_level_radio",
        on_change=_apply_stage_level,
        help="'Stage Group' shows high-level pipeline phases; 'Stage' shows individual strategy stages.",
    )
    # Also apply on first render (no callback fires on initial load)
    chosen = st.session_state.get("_stage_level_radio", current)
    if chosen != da.level:
        da.set_level(chosen)


def stage_selectbox(
    label: str = "Select Stage",
    key: str = "stage",
    default: str | None = None,
    **kwargs,
):
    """Render a stage selectbox that groups stages by their Stage Group.

    When the DecisionAnalyzer ``level`` is ``"Stage"`` and a stage→group
    mapping exists, each option is displayed as ``"GroupName / StageName"``
    so the user can see which group a stage belongs to.  At ``"Stage Group"``
    level (or when there is no mapping), options are shown as-is.

    Parameters
    ----------
    label : str
        Widget label.
    key : str
        Session-state key for the selected value.
    default : str, optional
        Preferred default value (e.g. ``"Arbitration"``).  Falls back to
        the first option when the default is not available.
    **kwargs
        Extra keyword arguments forwarded to ``st.selectbox``.
    """
    da = st.session_state.decision_data
    stage_options = da.getPossibleStageValues()
    mapping = da.stage_to_group_mapping  # empty dict when level != "Stage"

    if mapping:
        format_func = lambda s: f"{mapping.get(s, '?')}  ·  {s}"  # noqa: E731
    else:
        format_func = str

    # Determine index
    if default and key not in st.session_state and default in stage_options:
        index = stage_options.index(default)
    else:
        index = get_current_index(stage_options, key)

    st.selectbox(
        label,
        options=stage_options,
        index=index,
        key=key,
        format_func=format_func,
        **kwargs,
    )


def ensure_funnel():
    if st.session_state.decision_data.extract_type == "explainability_extract":
        st.warning(
            "This page requires **Action Analysis (v2)** data with full stage "
            "pipeline information. Explainability Extract (v1) data only contains "
            "the arbitration stage and cannot show the decision funnel."
        )
        st.stop()


def ensure_getFilterComponentData():
    return "Component Name" in st.session_state.decision_data.decision_data.collect_schema().names()


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


def _persist_widget_value(filter_type: str, column: str, regex: str = ""):
    """Copy widget value to a persistent session-state key.

    Streamlit clears keyed widget state on page navigation. We work around
    this by copying the value to a second key that survives navigation.
    See https://discuss.streamlit.io/t/session-state-is-not-preserved-when-navigating-pages/48787
    """
    src = f"{filter_type}{regex}_selected_{column}"
    dst = f"{filter_type}{regex}selected_{column}"
    st.session_state[dst] = st.session_state[src]


def reset_filter_state(filter_type: str):
    """Remove all session-state keys for the given filter type prefix."""
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(filter_type) and k != "filters"]
    for k in keys_to_remove:
        del st.session_state[k]


def _clean_unselected_filters(to_filter_columns: list[str], filter_type: str):
    """Remove session-state keys for columns no longer in the filter list."""
    keys_to_remove = []
    for key in st.session_state.keys():
        if "selected_" in key:
            column_name = key.split("selected_", 1)[1]
            if column_name not in to_filter_columns:
                keys_to_remove.append(column_name)
    for column in keys_to_remove:
        for key in [
            f"{filter_type}selected_{column}",
            f"{filter_type}_selected_{column}",
            f"{filter_type}regexselected_{column}",
            f"{filter_type}regex_selected_{column}",
            f"{filter_type}categories_{column}",
        ]:
            if key in st.session_state:
                del st.session_state[key]


def _render_column_selector(df: pl.LazyFrame, columns: list[str], filter_type: str) -> list[str]:
    """Render the multiselect widget for choosing which columns to filter on."""

    def _save_multiselect():
        st.session_state[f"{filter_type}multiselect"] = st.session_state[f"{filter_type}_multiselect"]

    st.session_state[f"{filter_type}_multiselect"] = st.session_state.get(f"{filter_type}multiselect", [])
    return st.multiselect(
        "Filter data on",
        columns,
        key=f"{filter_type}_multiselect",
        on_change=_save_multiselect,
    )


def _render_categorical_filter(
    df: pl.LazyFrame,
    column: str,
    container,
    filter_type: str,
    queries: list[pl.Expr],
):
    """Render filter UI for a categorical/string column.

    Uses a multiselect when fewer than 200 unique values exist, otherwise
    falls back to a regex text input.
    """
    categories_key = f"{filter_type}categories_{column}"
    if categories_key not in st.session_state:
        st.session_state[categories_key] = df.select(pl.col(column).unique()).collect().to_series().to_list()

    widget_key = f"{filter_type}_selected_{column}"
    persisted_key = f"{filter_type}selected_{column}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(persisted_key, st.session_state[categories_key])

    categories = st.session_state[categories_key]

    if len(categories) < 200:
        default = st.session_state.get(persisted_key, categories)
        st.session_state[widget_key] = default
        selected = container.multiselect(
            f"Values for {column}",
            options=categories,
            key=widget_key,
            on_change=_persist_widget_value,
            kwargs={"filter_type": filter_type, "column": column},
        )
        if selected != categories:
            queries.append(pl.col(column).cast(pl.Utf8).is_in(st.session_state[persisted_key]))
    else:
        # Too many unique values — use regex input instead
        if widget_key in st.session_state:
            del st.session_state[widget_key]
        regex_persisted = f"{filter_type}regexselected_{column}"
        default = st.session_state.get(regex_persisted, "")
        user_text_input = container.text_input(
            f"Substring or regex in {column}",
            value=default,
            key=f"{filter_type}regex_selected_{column}",
            on_change=_persist_widget_value,
            kwargs={
                "filter_type": filter_type,
                "column": column,
                "regex": "regex",
            },
        )
        if user_text_input:
            queries.append(pl.col(column).str.contains(user_text_input))


def _render_numeric_filter(
    df: pl.LazyFrame,
    column: str,
    container,
    filter_type: str,
    queries: list[pl.Expr],
):
    """Render filter UI for a numeric column (slider or dual number inputs)."""
    min_col, max_col = container.columns((1, 1))
    _min = float(df.select(pl.min(column)).collect().item())
    _max = float(df.select(pl.max(column)).collect().item())

    persisted_key = f"{filter_type}selected_{column}"
    if persisted_key not in st.session_state:
        default_min, default_max = _min, _max
    else:
        default_min, default_max = st.session_state[persisted_key]

    if _max - _min <= 200:
        user_num_input = container.slider(
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

    st.session_state[persisted_key] = user_num_input
    if user_num_input[0] != _min or user_num_input[1] != _max:
        queries.append(pl.col(column).is_between(*user_num_input))


def _render_temporal_filter(
    df: pl.LazyFrame,
    column: str,
    container,
    filter_type: str,
    queries: list[pl.Expr],
):
    """Render filter UI for a temporal (date/datetime) column."""
    value = (
        df.select(pl.min(column)).collect().item(),
        df.select(pl.max(column)).collect().item(),
    )
    widget_key = f"{filter_type}_selected_{column}"
    persisted_key = f"{filter_type}selected_{column}"
    st.session_state[widget_key] = st.session_state.get(persisted_key, value)

    user_date_input = container.date_input(
        f"Values for {column}",
        value=value,
        key=widget_key,
        on_change=_persist_widget_value,
        kwargs={"filter_type": filter_type, "column": column},
    )
    if len(user_date_input) == 2:
        queries.append(pl.col(column).is_between(*user_date_input))


def get_data_filters(df: pl.LazyFrame, columns=None, queries=None, filter_type="local") -> list[pl.Expr]:
    """Build filter expressions via interactive Streamlit widgets.

    Parameters
    ----------
    df : pl.LazyFrame
        Source data used to derive filter options.
    columns : list, optional
        Columns available for filtering. Defaults to all columns in *df*.
    queries : list, optional
        Pre-existing filter expressions to extend.
    filter_type : str
        Prefix for session-state keys, allowing independent filter sets
        (e.g. ``"global"`` vs ``"local"``).
    """
    if columns is None:
        columns = df.collect_schema().names()
    if queries is None:
        queries = []

    to_filter_columns = _render_column_selector(df, columns, filter_type)

    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        left.write("## ↳")

        col_dtype = df.collect_schema()[column]
        if col_dtype in (pl.Categorical, pl.Utf8):
            _render_categorical_filter(df, column, right, filter_type, queries)
        elif col_dtype.is_numeric():
            _render_numeric_filter(df, column, right, filter_type, queries)
        elif col_dtype.is_temporal():
            _render_temporal_filter(df, column, right, filter_type, queries)

    _clean_unselected_filters(to_filter_columns, filter_type)
    return queries


def get_options() -> list[str]:
    """Data source options.

    'File path' is only shown in managed deployments where users need to
    reference server-side paths (e.g. S3-mounted directories). For local use,
    the file uploader's browse button is sufficient.
    """
    options = ["Sample data", "File upload"]
    if is_managed_deployment():
        options.append("File path")
    return options


def handle_sample_data() -> pl.LazyFrame | None:
    """Load built-in sample data for demo purposes."""
    # Prefer local file when available (e.g. during development), fall back
    # to downloading from GitHub for installed-package users.
    local_path = Path(__file__).resolve().parents[4] / "data" / "sample_eev2.parquet"
    if local_path.exists():
        return pl.scan_parquet(local_path)

    return read_ds_export(
        filename="sample_eev2.parquet",
        path="https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data",
    )


def handle_data_path() -> pl.LazyFrame | None:
    """Load data from the ``--data-path`` CLI flag, if configured.

    Supports the same formats as the file upload: parquet, csv, json, arrow,
    zip archives, and partitioned directories.
    """
    data_path = get_data_path()
    if not data_path:
        return None

    p = Path(data_path)
    if not p.exists():
        st.error(f"Configured data path does not exist: `{data_path}`")
        return None

    # Single zip file: extract first, then read the extracted contents
    if p.is_file() and p.suffix.lower() == ".zip":
        import tempfile
        import zipfile

        tmp_dir = tempfile.mkdtemp(prefix="da_path_")
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp_dir)
        return read_data(tmp_dir)

    return read_data(data_path)


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


def handle_file_upload() -> pl.LazyFrame | None:
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
        elif name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz") or suffix == ".tar":
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


def handle_file_path() -> pl.LazyFrame | None:
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
def st_priority_component_distribution(value_data: pl.LazyFrame, component, granularity):
    return plot_priority_component_distribution(value_data, component, granularity)


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def st_component_overview(value_data: pl.LazyFrame, components, granularity):
    return plot_component_overview(value_data, components, granularity)
