# python/pdstools/app/decision_analyzer/da_streamlit_utils.py
from __future__ import annotations
from pathlib import Path

import polars as pl
import streamlit as st

from pdstools.decision_analyzer.data_read_utils import read_nested_zip_files
from pdstools.pega_io.File import _clean_artifacts, read_data, read_ds_export

from pdstools.decision_analyzer.plots import (
    plot_component_overview,
    plot_priority_component_distribution,
)
from pdstools.utils.streamlit_utils import (
    _apply_sidebar_logo,
    ensure_session_data,
    get_current_index,  # noqa: F401 — re-exported for backward compat
    get_data_path,
)


def show_cli_filter_banner():
    """Show active CLI --filter specs in the sidebar if any are set."""
    from pdstools.utils.streamlit_utils import get_filter_specs

    filter_specs = get_filter_specs()
    if not filter_specs:
        return
    with st.sidebar:
        st.caption("**Active CLI filters:**")
        for spec in filter_specs:
            st.caption(f"  `{spec}`")


def ensure_data():
    """Guard: stop if decision data is not loaded."""
    _apply_sidebar_logo()
    ensure_session_data("decision_data", "Please upload your data in the Home page.")
    show_cli_filter_banner()


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
    options: list[str] | None = None,
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
    options : list[str], optional
        Explicit list of stage values to show. When provided, overrides
        ``da.get_possible_stage_values()``.
    **kwargs
        Extra keyword arguments forwarded to ``st.selectbox``.
    """
    da = st.session_state.decision_data
    stage_options = options if options is not None else da.get_possible_stage_values()
    mapping = da.stage_to_group_mapping  # empty dict when level != "Stage"

    if mapping:
        format_func = lambda s: f"{mapping.get(s, '?')}  ·  {s}"  # noqa: E731
    else:
        format_func = str

    # If the stored value is no longer valid (e.g. after a level switch),
    # reset to the default so upstream code reading session state gets a
    # valid stage before the widget re-renders.
    if key in st.session_state and st.session_state[key] not in stage_options:
        if default and default in stage_options:
            st.session_state[key] = default
        else:
            st.session_state[key] = stage_options[0]

    # Only pass index when the key is not yet in session state;
    # Streamlit does not allow both a session-state value and an index.
    selectbox_kwargs: dict = dict(
        label=label,
        options=stage_options,
        key=key,
        format_func=format_func,
        **kwargs,
    )
    if key not in st.session_state:
        if default and default in stage_options:
            selectbox_kwargs["index"] = stage_options.index(default)
        else:
            selectbox_kwargs["index"] = 0

    st.selectbox(**selectbox_kwargs)


def ensure_funnel():
    if st.session_state.decision_data.extract_type == "explainability_extract":
        st.warning(
            "This page requires **Action Analysis (v2)** data with full stage "
            "pipeline information. Explainability Extract (v1) data only contains "
            "the arbitration stage and cannot show the decision funnel."
        )
        st.stop()


def ensure_get_filter_component_data():
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


def _render_column_selector(
    df: pl.LazyFrame,
    columns: list[str],
    filter_type: str,
    selector_label: str = "Filter data on",
    sort_columns: bool = True,
) -> list[str]:
    """Render the multiselect widget for choosing which columns to filter on."""

    def _save_multiselect():
        st.session_state[f"{filter_type}multiselect"] = st.session_state[f"{filter_type}_multiselect"]

    options = sorted(columns, key=str.casefold) if sort_columns else columns
    st.session_state[f"{filter_type}_multiselect"] = st.session_state.get(f"{filter_type}multiselect", [])
    return st.multiselect(
        selector_label,
        options,
        key=f"{filter_type}_multiselect",
        on_change=_save_multiselect,
    )


def _render_categorical_filter(
    df: pl.LazyFrame,
    column: str,
    container,
    filter_type: str,
    queries: list[pl.Expr],
    default_select_all_categories: bool,
):
    """Render filter UI for a categorical/string column.

    Uses a multiselect when fewer than 200 unique values exist, otherwise
    falls back to a regex text input.
    """
    categories_key = f"{filter_type}categories_{column}"
    if categories_key not in st.session_state:
        categories = df.select(pl.col(column).cast(pl.Utf8).drop_nulls().unique()).collect().to_series().to_list()
        st.session_state[categories_key] = sorted(categories, key=str.casefold)

    widget_key = f"{filter_type}_selected_{column}"
    persisted_key = f"{filter_type}selected_{column}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(persisted_key, st.session_state[categories_key])

    categories = st.session_state[categories_key]

    if len(categories) < 200:
        default_selection = categories if default_select_all_categories else []
        default = st.session_state.get(persisted_key, default_selection)
        st.session_state[widget_key] = default
        selected = container.multiselect(
            f"Values for {column}",
            options=categories,
            key=widget_key,
            on_change=_persist_widget_value,
            kwargs={"filter_type": filter_type, "column": column},
        )
        st.session_state[persisted_key] = selected
        if selected != categories:
            queries.append(pl.col(column).cast(pl.Utf8).is_in(selected))
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


def get_data_filters(
    df: pl.LazyFrame,
    columns=None,
    queries=None,
    filter_type="local",
    default_select_all_categories: bool = True,
    selector_label: str = "Filter data on",
    sort_columns: bool = True,
) -> list[pl.Expr]:
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
    default_select_all_categories : bool, default True
        Controls default categorical selection behavior. If True, all
        categories are selected initially (non-restrictive). If False,
        categorical selections start empty and users add values explicitly.
    selector_label : str, default "Filter data on"
        Label shown above the column selector multiselect.
    sort_columns : bool, default True
        If True, columns are sorted alphabetically. If False, the order
        of *columns* is preserved.
    """
    if columns is None:
        columns = df.collect_schema().names()
    if queries is None:
        queries = []

    to_filter_columns = _render_column_selector(
        df,
        columns,
        filter_type,
        selector_label,
        sort_columns=sort_columns,
    )

    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        left.write("## ↳")

        col_dtype = df.collect_schema()[column]
        if col_dtype in (pl.Categorical, pl.Utf8):
            _render_categorical_filter(
                df,
                column,
                right,
                filter_type,
                queries,
                default_select_all_categories,
            )
        elif col_dtype.is_numeric():
            _render_numeric_filter(df, column, right, filter_type, queries)
        elif col_dtype.is_temporal():
            _render_temporal_filter(df, column, right, filter_type, queries)

    _clean_unselected_filters(to_filter_columns, filter_type)
    return queries


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


def _is_tar_file(p: Path) -> bool:
    """Check if a path refers to a tar archive (plain or compressed)."""
    name = p.name.lower()
    return name.endswith((".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz"))


def handle_data_path() -> pl.LazyFrame | None:
    """Load data from the ``--data-path`` CLI flag, if configured.

    Supports the same formats as the file upload: parquet, csv, json, arrow,
    zip archives, tar archives, and partitioned directories.
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

        try:
            from pdstools.utils.progress_utils import (
                estimate_extraction_time,
                format_time_estimate,
            )

            file_size = p.stat().st_size
            min_time, max_time = estimate_extraction_time(file_size)
            time_msg = format_time_estimate(min_time, max_time)
            spinner_msg = f"Extracting archive... (estimated: {time_msg})"
        except Exception:
            spinner_msg = "Extracting archive..."

        tmp_dir = tempfile.mkdtemp(prefix="da_path_")
        with st.spinner(spinner_msg):
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tmp_dir)
        _clean_artifacts(tmp_dir)
        return read_data(tmp_dir)

    # Tar archive (plain or compressed): extract first, then read contents
    if p.is_file() and _is_tar_file(p):
        import tarfile
        import tempfile

        try:
            from pdstools.utils.progress_utils import (
                estimate_extraction_time,
                format_time_estimate,
            )

            file_size = p.stat().st_size
            min_time, max_time = estimate_extraction_time(file_size)
            time_msg = format_time_estimate(min_time, max_time)
            spinner_msg = f"Extracting archive... (estimated: {time_msg})"
        except Exception:
            spinner_msg = "Extracting archive..."

        tmp_dir = tempfile.mkdtemp(prefix="da_path_tar_")
        with st.spinner(spinner_msg):
            # Open file handle first to avoid macOS permission errors
            # (tarfile auto-detection re-opens by name for gzip probing)
            with open(p, "rb") as fh:
                with tarfile.open(fileobj=fh, mode="r:*") as tf:
                    tf.extractall(tmp_dir, filter="data")
        _clean_artifacts(tmp_dir)
        return read_data(tmp_dir)

    return read_data(data_path)


def _read_uploaded_zip(file_buffer) -> pl.LazyFrame:
    """Read a zip file, handling both Action Analysis exports and flat archive layouts."""
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

        # If the zip contains .zip files, treat as Pega Action Analysis export format
        # (nested archive where inner ".zip" files are actually gzipped NDJSON)
        if ".zip" in inner_exts:
            file_buffer.seek(0)
            return read_nested_zip_files(file_buffer)

        # Otherwise extract to a temp directory and use read_data
        try:
            from pdstools.utils.progress_utils import (
                estimate_extraction_time,
                format_time_estimate,
            )

            file_buffer.seek(0, 2)
            file_size = file_buffer.tell()
            file_buffer.seek(0)

            min_time, max_time = estimate_extraction_time(file_size)
            time_msg = format_time_estimate(min_time, max_time)
            spinner_msg = f"Extracting uploaded archive... (estimated: {time_msg})"
        except Exception:
            spinner_msg = "Extracting uploaded archive..."

        tmp_dir = tempfile.mkdtemp(prefix="da_upload_")
        with st.spinner(spinner_msg):
            zf.extractall(tmp_dir)

    _clean_artifacts(tmp_dir)
    return read_data(tmp_dir)


def _read_uploaded_tar(file_buffer) -> pl.LazyFrame:
    """Extract a tar/tar.gz/tgz archive to a temp directory and read its contents."""
    import tarfile
    import tempfile

    try:
        from pdstools.utils.progress_utils import (
            estimate_extraction_time,
            format_time_estimate,
        )

        file_buffer.seek(0, 2)
        file_size = file_buffer.tell()
        file_buffer.seek(0)

        min_time, max_time = estimate_extraction_time(file_size)
        time_msg = format_time_estimate(min_time, max_time)
        spinner_msg = f"Extracting uploaded archive... (estimated: {time_msg})"
    except Exception:
        spinner_msg = "Extracting uploaded archive..."

    tmp_dir = tempfile.mkdtemp(prefix="da_tar_")
    with st.spinner(spinner_msg):
        with tarfile.open(fileobj=file_buffer, mode="r:*") as tf:
            tf.extractall(tmp_dir, filter="data")
    _clean_artifacts(tmp_dir)
    return read_data(tmp_dir)


def handle_file_upload() -> tuple[pl.LazyFrame | None, dict | None]:
    """Show file uploader accepting one or more files and return a LazyFrame and metadata, or (None, None).

    Returns:
        Tuple of (LazyFrame, metadata dict) where metadata contains sample information if available.
        For single parquet file uploads, metadata is read from the file.
        For other cases, metadata is None.
    """
    import tempfile
    from pdstools.decision_analyzer.utils import _read_source_metadata

    uploaded_files = st.file_uploader(
        "Upload decision data",
        type=["zip", "parquet", "json", "csv", "arrow", "gz", "tgz", "tar"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        return None, None

    frames: list[pl.LazyFrame] = []
    temp_paths: list[str] = []  # Track temp file paths for parquet files

    for f in uploaded_files:
        name_lower = f.name.lower()
        suffix = Path(f.name).suffix.lower()
        if suffix == ".parquet":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
                tmp.write(f.getbuffer())
                temp_paths.append(tmp.name)
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
        return None, None

    # Try to read metadata for single parquet file uploads
    metadata = None
    if len(frames) == 1 and len(temp_paths) == 1:
        metadata = _read_source_metadata(temp_paths[0])

    if len(frames) == 1:
        return frames[0], metadata
    return pl.concat(frames, how="diagonal", rechunk=True), None


def get_available_channel_directions(sample_df: pl.LazyFrame) -> list[str]:
    """Get list of Channel/Direction combinations from the sample data.

    Handles both v2 data (with Direction) and v1 data (Channel only).
    Respects any filters already applied to the sample (e.g., global filters).

    Parameters
    ----------
    sample_df : pl.LazyFrame
        The sample data (after global filters applied).

    Returns
    -------
    list[str]
        Sorted list of "Channel/Direction" strings for v2 data,
        or sorted list of "Channel" strings for v1 data,
        or empty list if no channel data available.
    """
    schema = sample_df.collect_schema().names()

    if "Direction" in schema:
        combos = sample_df.select(pl.struct("Channel", "Direction")).unique().collect().to_series().to_list()
        return sorted([f"{c['Channel']}/{c['Direction']}" for c in combos])
    elif "Channel" in schema:
        channels = sample_df.select("Channel").unique().collect().to_series().to_list()
        return sorted(channels)
    else:
        return []


def _update_channel_filter():
    """Callback to update filter expression when channel selection changes."""
    selected = st.session_state._channel_direction_widget
    st.session_state.page_channel_filter = selected

    if selected == "Any":
        st.session_state.page_channel_expr = None
    else:
        if "/" in selected:
            channel, direction = selected.split("/", 1)
            st.session_state.page_channel_expr = (pl.col("Channel") == channel) & (pl.col("Direction") == direction)
        else:
            st.session_state.page_channel_expr = pl.col("Channel") == selected


def contextual_filters():
    """Render the universal contextual filter section in the sidebar.

    Adds a visual divider and section header, then renders all contextual
    filters (currently Channel/Direction). Every analysis page should call
    this as the last sidebar item to maintain a consistent layout.
    """
    st.divider()
    st.caption("**Filters**")
    channel_direction_selector()


def channel_direction_selector():
    """Render channel/direction selector in sidebar.

    Stores selection in st.session_state.page_channel_filter (UI value)
    and st.session_state.page_channel_expr (Polars filter expression).
    The filter is applied via DecisionAnalyzer.filtered_sample property.
    """
    da = st.session_state.decision_data

    available = get_available_channel_directions(da.sample)

    if not available:
        st.warning("No channel data available with current global filters.")
        st.session_state.page_channel_filter = "Any"
        st.session_state.page_channel_expr = None
        return

    current = st.session_state.get("page_channel_filter", "Any")
    if current != "Any" and current not in available:
        st.info(f"Previous selection '{current}' is no longer available. Reset to 'Any'.")
        current = "Any"
        st.session_state.page_channel_filter = "Any"
        st.session_state.page_channel_expr = None

    options = ["Any"] + available

    st.selectbox(
        "Channel / Direction",
        options=options,
        index=options.index(current),
        key="_channel_direction_widget",
        on_change=_update_channel_filter,
        help="Filter analysis to a specific channel. 'Any' shows all channels that pass global filters.",
    )


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
def st_priority_component_distribution(value_data: pl.LazyFrame, component, granularity, color_discrete_map=None):
    return plot_priority_component_distribution(value_data, component, granularity, color_discrete_map)


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def st_component_overview(value_data: pl.LazyFrame, components, granularity):
    return plot_component_overview(value_data, components, granularity)
