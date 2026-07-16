"""Health Check-specific Streamlit helpers.

Mirrors the layout of ``da_streamlit_utils.py`` and ``ia_streamlit_utils.py``
so each app's data-loading glue lives next to the app it serves. Shared
widgets/helpers stay in :mod:`pdstools.utils.streamlit_utils`.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st

from pdstools import pega_io
from pdstools.adm.HealthCheckImport import (
    HealthCheckReadOptions,
    HealthCheckSource,
    SourceImportOptions,
    SourceNormalizationOptions,
    import_health_check_data,
    preview_health_check_columns,
    resolve_health_check_output_dir,
    save_health_check_parquet,
)
from pdstools.utils.streamlit_utils import (
    _apply_sidebar_logo,
    cached_datamart,
    cached_prediction_table,
    cached_sample,
    cached_sample_prediction,
    get_data_path,
)
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pdstools.adm.ADMDatamart import ADMDatamart

logger = logging.getLogger(__name__)

_UPLOAD_TYPES = [
    "json",
    "jsonl",
    "ndjson",
    "zip",
    "parquet",
    "csv",
    "tsv",
    "txt",
    "arrow",
    "feather",
    "ipc",
    "xlsx",
    "xls",
]
_REQUIRED_IMPORT_FIELDS = {
    "Model": (
        "pyModelID",
        "pyConfigurationName",
        "pySnapshotTime",
        "pyPositives",
        "pyNegatives",
        "pyResponseCount",
        "pyPerformance",
        "pyChannel",
        "pyDirection",
        "pyIssue",
        "pyGroup",
        "pyName",
    ),
    "Predictor": (
        "pyModelID",
        "pyPredictorName",
        "pyPerformance",
        "pyBinIndex",
        "pyBinPositives",
        "pyBinNegatives",
        "pySnapshotTime",
        "pyEntryType",
    ),
}
_DEFAULT_REQUIRED_REPAIRS = {
    "pyChannel": "NA",
    "pyDirection": "NA",
    "pyIssue": "NA",
    "pyGroup": "NA",
    "pyEntryType": "Active",
}
_DEFAULT_PREDICTOR_CATEGORIES = (
    "External Model=Propensity,Score,Class,Classifier,Classification,Probability,Prediction,Predicted,ModelScore"
)


def _invalidate_manual_import() -> None:
    """Clear imported objects after a source or parsing setting changes."""
    for key in (
        "dm",
        "prediction",
        "_hc_import_result",
        "_hc_import_warnings",
        "_hc_output_dir",
        "_hc_written_paths",
    ):
        st.session_state.pop(key, None)


def _source_extension(source: HealthCheckSource | None) -> str:
    if source is None:
        return ""
    name = getattr(source, "name", os.fspath(source) if not isinstance(source, BytesIO) else "")
    return Path(name).suffix.casefold()


def _parse_assignments(text: str, *, list_values: bool = False) -> dict:
    """Parse newline-separated ``name=value`` settings from a widget."""
    parsed: dict = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Line {line_number} must use name=value syntax")
        name, value = (part.strip() for part in line.split("=", 1))
        if not name or not value:
            raise ValueError(f"Line {line_number} must contain both a name and value")
        parsed[name] = [item.strip() for item in value.split(",") if item.strip()] if list_values else value
    return parsed


def _parse_columns(text: str) -> tuple[str, ...]:
    return tuple(column.strip() for column in text.split(",") if column.strip())


def _column_key(column: str) -> str:
    key = column.casefold()
    return key[2:] if key.startswith(("py", "px")) else key


def _column_keys(columns: tuple[str, ...]) -> set[str]:
    keys = set()
    for column in columns:
        folded = column.casefold()
        keys.add(folded)
        keys.add(_column_key(column))
    return keys


def _required_field_repairs(role: str, source_columns: tuple[str, ...]) -> str:
    required = _REQUIRED_IMPORT_FIELDS.get(role, ())
    if not required:
        return ""

    existing = _column_keys(source_columns)
    repairs = []
    for field in required:
        if field.casefold() in existing or _column_key(field) in existing:
            continue
        if field == "pyNegatives" and {"responsecount", "positives"}.issubset(existing):
            repairs.append("pyNegatives=pyResponseCount-pyPositives")
        elif field in _DEFAULT_REQUIRED_REPAIRS:
            repairs.append(f"{field}={_DEFAULT_REQUIRED_REPAIRS[field]}")
    return "\n".join(repairs)


def _looks_like_derived_expression(value: str, source_columns: tuple[str, ...]) -> bool:
    stripped = value.strip()
    if stripped.startswith("concat(") and stripped.endswith(")"):
        return True
    if re.fullmatch(r"[A-Za-z_][\w.]*\s*-\s*[A-Za-z_][\w.]*", stripped):
        return True
    return _column_key(stripped) in _column_keys(source_columns)


def _split_field_repairs(
    text: str,
    source_columns: tuple[str, ...],
) -> tuple[dict, dict, dict]:
    fill_null_values: dict = {}
    derived_columns: dict = {}
    constant_columns: dict = {}
    existing = _column_keys(source_columns)
    for target, value in _parse_assignments(text).items():
        if _looks_like_derived_expression(value, source_columns):
            derived_columns[target] = value
        elif target.casefold() in existing or _column_key(target) in existing:
            fill_null_values[target] = value
        else:
            constant_columns[target] = value
    return fill_null_values, derived_columns, constant_columns


def _field_repairs_text_area(
    role: str,
    prefix: str,
    source: HealthCheckSource,
    read_options: HealthCheckReadOptions,
) -> tuple[str, tuple[str, ...]]:
    try:
        source_columns = preview_health_check_columns(source, read_options)
    except Exception:
        source_columns = ()
    default_repairs = _required_field_repairs(role, source_columns)
    key = f"{prefix}_field_repairs"
    default_key = f"{key}_default"
    previous_default = st.session_state.get(default_key)
    if key not in st.session_state or st.session_state[key] == previous_default:
        st.session_state[key] = default_repairs
    st.session_state[default_key] = default_repairs
    return (
        st.text_area(
            "Missing required field repairs",
            key=key,
            placeholder="pyChannel=NA\npyNegatives=pyResponseCount-pyPositives",
            on_change=_invalidate_manual_import,
        ),
        source_columns,
    )


def _source_import_options(
    role: str,
    source: HealthCheckSource,
    *,
    show_schema_inference: bool = True,
) -> SourceImportOptions:
    """Render and return independent parsing settings for one source."""
    prefix = f"_hc_{role.lower()}"
    extension = _source_extension(source)
    source_label = {
        "Model": "Model Snapshot Data",
        "Predictor": "Predictor Binning Data",
        "Prediction": "Prediction Data",
    }.get(role, role)
    st.markdown(f"#### Settings for the {source_label}")

    delimiter = None
    quote_char: str | None = '"'
    encoding = "utf8"
    has_header = True
    skip_rows = 0
    infer_schema_length = None
    ignore_errors = False
    null_values = None
    excel_sheet_name = None
    excel_sheet_id = None
    excel_header_row = None

    if extension in {".csv", ".tsv", ".txt"}:
        default_delimiter = r"\t" if extension in {".tsv", ".txt"} else ","
        delimiter_text = st.text_input(
            "Delimiter",
            value=default_delimiter,
            key=f"{prefix}_delimiter",
            on_change=_invalidate_manual_import,
        )
        delimiter = "\t" if delimiter_text == r"\t" else delimiter_text
        quote_text = st.text_input(
            "Quote character",
            value='"',
            key=f"{prefix}_quote_char",
            on_change=_invalidate_manual_import,
        )
        quote_char = quote_text or None
        encoding_options = ["utf8", "utf8-lossy"]
        encoding = cast(
            str,
            st.selectbox(
                "Encoding",
                options=encoding_options,
                index=encoding_options.index("utf8"),
                key=f"{prefix}_encoding",
                on_change=_invalidate_manual_import,
            ),
        )
        has_header = st.checkbox(
            "Has a header row",
            value=True,
            key=f"{prefix}_has_header",
            on_change=_invalidate_manual_import,
        )
        skip_rows = int(
            st.number_input(
                "Rows to skip",
                min_value=0,
                value=0,
                key=f"{prefix}_skip_rows",
                on_change=_invalidate_manual_import,
            )
        )
        if show_schema_inference:
            infer_schema_length = int(
                st.number_input(
                    "Schema inference length",
                    min_value=0,
                    value=10000,
                    step=10000,
                    key=f"{prefix}_infer_schema_length",
                    on_change=_invalidate_manual_import,
                )
            )
        ignore_errors = st.checkbox(
            "Ignore parsing errors",
            value=False,
            key=f"{prefix}_ignore_errors",
            on_change=_invalidate_manual_import,
        )
        null_text = st.text_input(
            "Null values",
            value=",NA,N/A,NULL",
            key=f"{prefix}_null_values",
            on_change=_invalidate_manual_import,
        )
        null_values = tuple(value.strip() for value in null_text.split(","))
    elif extension in {".xlsx", ".xls"}:
        excel_sheet_name = (
            st.text_input(
                "Excel sheet name",
                key=f"{prefix}_sheet_name",
                on_change=_invalidate_manual_import,
            ).strip()
            or None
        )
        excel_header_row = int(
            st.number_input(
                "Excel header row",
                min_value=0,
                value=0,
                key=f"{prefix}_header_row",
                on_change=_invalidate_manual_import,
            )
        )
        if show_schema_inference:
            infer_schema_length = int(
                st.number_input(
                    "Schema inference length",
                    min_value=0,
                    value=10000,
                    step=10000,
                    key=f"{prefix}_infer_schema_length",
                    on_change=_invalidate_manual_import,
                )
            )
    elif extension in {".json", ".jsonl", ".ndjson", ".zip", ".gz"}:
        if show_schema_inference:
            infer_schema_length = int(
                st.number_input(
                    "Schema inference length",
                    min_value=0,
                    value=10000,
                    step=10000,
                    key=f"{prefix}_infer_schema_length",
                    on_change=_invalidate_manual_import,
                )
            )

    read_options = HealthCheckReadOptions(
        delimiter=delimiter,
        quote_char=quote_char,
        encoding=encoding,
        has_header=has_header,
        skip_rows=skip_rows,
        infer_schema_length=infer_schema_length,
        ignore_errors=ignore_errors,
        null_values=null_values,
        excel_sheet_name=excel_sheet_name,
        excel_sheet_id=excel_sheet_id,
        excel_header_row=excel_header_row,
    )
    field_repairs_text, source_columns = _field_repairs_text_area(role, prefix, source, read_options)
    fill_null_values, derived_columns, constant_columns = _split_field_repairs(field_repairs_text, source_columns)
    timestamp_column = (
        st.text_input(
            "Timestamp column",
            key=f"{prefix}_timestamp_column",
            on_change=_invalidate_manual_import,
        ).strip()
        or None
    )
    timestamp_format = (
        st.text_input(
            "Timestamp format",
            key=f"{prefix}_timestamp_format",
            placeholder="%Y-%m-%d %H:%M:%S",
            on_change=_invalidate_manual_import,
        ).strip()
        or None
    )
    fallback_text = st.text_input(
        "Timestamp fallback",
        key=f"{prefix}_timestamp_fallback",
        placeholder="2026-01-01T00:00:00",
        on_change=_invalidate_manual_import,
    ).strip()
    timestamp_fallback = datetime.fromisoformat(fallback_text) if fallback_text else None

    return SourceImportOptions(
        read=read_options,
        normalize=SourceNormalizationOptions(
            fill_null_values=fill_null_values,
            derived_columns=derived_columns,
            timestamp_column=timestamp_column,
            timestamp_format=timestamp_format,
            timestamp_fallback=timestamp_fallback,
            constant_columns=constant_columns,
        ),
    )


def _detected_sources(directory: str) -> tuple[str | None, str | None, str | None]:
    """Return standard files auto-detected in a pasted directory path."""
    if not directory:
        return None, None, None
    path = Path(directory).expanduser()
    if not path.exists():
        st.error(f"Directory not found: {directory}")
        return None, None, None
    if not path.is_dir():
        st.error(f"Not a directory: {directory}")
        return None, None, None
    return (
        pega_io.get_latest_file(str(path), target="model_data"),
        pega_io.get_latest_file(str(path), target="predictor_data"),
        pega_io.get_latest_file(str(path), target="prediction_data"),
    )


def _path_sources() -> tuple[str | None, str | None, str | None]:
    st.text_input(
        "Data folder for automatic file detection",
        key="_hc_import_directory",
        on_change=_invalidate_manual_import,
    )
    detected = _detected_sources(st.session_state.get("_hc_import_directory", ""))
    model_path = st.text_input(
        "Model Snapshot path",
        key="_hc_model_path",
        on_change=_invalidate_manual_import,
    ).strip()
    predictor_path = st.text_input(
        "Predictor Binning snapshot path (optional)",
        key="_hc_predictor_path",
        on_change=_invalidate_manual_import,
    ).strip()
    prediction_path = st.text_input(
        "Prediction Table path (optional)",
        key="_hc_prediction_path",
        on_change=_invalidate_manual_import,
    ).strip()
    sources = (
        model_path or detected[0],
        predictor_path or detected[1],
        prediction_path or detected[2],
    )
    for label, source in zip(("Model", "Predictor", "Prediction"), sources, strict=True):
        if source:
            st.caption(f"{label}: {source}")
    return sources


def _upload_sources():
    model = st.file_uploader(
        "Upload Model Snapshot",
        type=_UPLOAD_TYPES,
        key="_hc_model_upload",
        on_change=_invalidate_manual_import,
    )
    predictor = st.file_uploader(
        "Upload Predictor Binning snapshot (optional)",
        type=_UPLOAD_TYPES,
        key="_hc_predictor_upload",
        on_change=_invalidate_manual_import,
    )
    prediction = st.file_uploader(
        "Upload Prediction Table (optional)",
        type=_UPLOAD_TYPES,
        key="_hc_prediction_upload",
        on_change=_invalidate_manual_import,
    )
    return model, predictor, prediction


def render_import_ui() -> None:
    """Render the Health Check import workflow."""
    st.write("### Data import")
    upload_sources = _upload_sources()
    with st.expander("File paths (optional)"):
        path_sources = _path_sources()

    model_source, predictor_source, prediction_source = (
        uploaded or path for uploaded, path in zip(upload_sources, path_sources, strict=True)
    )
    source_mode = "Direct file upload" if upload_sources[0] is not None else "Direct file path"

    model_options = predictor_options = prediction_options = None
    predictor_categorization = None
    categorization_uses_regex = False
    extract_pyname_keys = st.session_state.get("extract_pyname_keys", True)
    keep_parquet = False
    output_parent: Path | None = None
    configuration_error = False

    with st.expander("Predictor categorization (optional)"):
        st.info(
            "Add custom predictor categories as `Category=pattern1,pattern2`, one category per line. "
            "Patterns are matched against `PredictorName`; by default they are literal substring matches, "
            "for example `External Model=Score` matches `Customer.Score`. Enable regular expressions to "
            "pass the patterns to `ADMDatamart.apply_predictor_categorization(..., use_regexp=True)`. "
            "Predictors with no match keep their existing category."
        )
        if predictor_source is None:
            st.caption("Select a Predictor Binning snapshot to configure predictor categories.")
        else:
            try:
                st.session_state.setdefault(
                    "_hc_predictor_categories",
                    _DEFAULT_PREDICTOR_CATEGORIES,
                )
                category_text = st.text_area(
                    "Predictor category mappings",
                    key="_hc_predictor_categories",
                    on_change=_invalidate_manual_import,
                )
                predictor_categorization = _parse_assignments(category_text, list_values=True) or None
            except ValueError as error:
                configuration_error = True
                st.error(str(error))
            categorization_uses_regex = st.checkbox(
                "Use regular expressions for predictor mappings",
                key="_hc_predictor_category_regex",
                on_change=_invalidate_manual_import,
            )

    with st.expander("Processed parquet cache (optional)"):
        keep_parquet = st.checkbox(
            "Keep processed parquet files",
            key="_hc_keep_parquet",
        )
        if keep_parquet:
            if isinstance(model_source, str):
                use_smart_location = st.checkbox(
                    "Use source folder for HC cache",
                    value=True,
                    key="_hc_smart_cache_location",
                )
            else:
                use_smart_location = False
            if use_smart_location:
                output_dir = resolve_health_check_output_dir(
                    model_source,
                    predictor_source,
                    prediction_source,
                )
                output_parent = output_dir.parent
            else:
                output_parent_text = st.text_input(
                    "Output folder",
                    value=str(Path.cwd()),
                    key="_hc_output_parent",
                )
                output_parent = Path(output_parent_text).expanduser()
                output_dir = resolve_health_check_output_dir(output_parent=output_parent)
            st.caption(f"Processed parquet destination: {output_dir}")

    with st.expander("Advanced import settings (optional)"):
        extract_pyname_keys = st.checkbox(
            "Extract Treatments",
            value=extract_pyname_keys,
            key="_hc_extract_pyname_keys",
            on_change=_invalidate_manual_import,
        )
        st.session_state["extract_pyname_keys"] = extract_pyname_keys
        try:
            if model_source is not None:
                model_options = _source_import_options("Model", model_source)
            if predictor_source is not None:
                predictor_options = _source_import_options("Predictor", predictor_source)
            if prediction_source is not None:
                prediction_options = _source_import_options(
                    "Prediction",
                    prediction_source,
                    show_schema_inference=False,
                )
        except ValueError as error:
            configuration_error = True
            st.error(str(error))

    if model_source is None:
        st.info("Select a Model Snapshot to enable Import.")

    if st.button(
        "Import",
        type="primary",
        disabled=model_source is None or configuration_error,
        key="_hc_import_button",
    ):
        _invalidate_manual_import()
        if model_source is None:
            st.error("Select a Model Snapshot to enable Import.")
            st.stop()
        try:
            with st.spinner("Importing Health Check data..."):
                result = import_health_check_data(
                    model_source,
                    predictor_source,
                    prediction_source,
                    model_options=model_options,
                    predictor_options=predictor_options,
                    prediction_options=prediction_options,
                    extract_pyname_keys=extract_pyname_keys,
                    predictor_categorization=predictor_categorization,
                    predictor_categorization_uses_regex=categorization_uses_regex,
                )
                st.session_state["data_source"] = source_mode
                st.session_state["dm"] = result.datamart
                st.session_state["_hc_import_result"] = result
                if result.prediction is not None:
                    st.session_state["prediction"] = result.prediction
                else:
                    st.session_state.pop("prediction", None)
                st.session_state["_hc_import_warnings"] = result.warnings
                if keep_parquet and output_parent is not None:
                    paths = save_health_check_parquet(result, output_parent)
                    st.session_state["_hc_output_dir"] = str(
                        resolve_health_check_output_dir(output_parent=output_parent)
                    )
                    st.session_state["_hc_written_paths"] = tuple(str(path) for path in paths.values())
                else:
                    st.session_state.pop("_hc_output_dir", None)
        except Exception as error:
            _invalidate_manual_import()
            st.error(f"Import failed: {error}")

    if "_hc_import_result" in st.session_state:
        st.success("Import Successful!")
        for warning in st.session_state.get("_hc_import_warnings", ()):
            st.warning(warning)
        for path in st.session_state.get("_hc_written_paths", ()):
            st.caption(f"Wrote {path}")
    elif "dm" in st.session_state:
        st.info("Sample or configured data is loaded. Upload files or enter file paths to replace it.")


def ensure_dm_loaded(*, show_toast: bool = False) -> bool:
    """Ensure ``st.session_state["dm"]`` is populated, auto-loading a default.

    Used both by the Health Check home page (first-run UX) and by data-required
    sub-pages (deep-link / launcher flow) so a user who lands on a sub-page
    without visiting Home first still gets a working app.

    Priority chain:

    1. If ``dm`` is already in session_state, return ``True`` immediately.
    2. Try the ``--data-path`` CLI flag via :func:`handle_data_path_hc`.
    3. Fall back to the bundled CDH sample.

    Parameters
    ----------
    show_toast : bool, default False
        When True and a load actually happened (i.e. session_state was empty
        on entry), surface a non-modal toast describing what was loaded.
        The home page handles its own toasts to also drive ``st.rerun()``,
        so it leaves this False; sub-pages set it True.

    Returns
    -------
    bool
        ``True`` when ``dm`` is in session_state on return, ``False`` when
        every load attempt failed (no CLI path resolves, no sample bundled,
        etc.). Callers fall back to a "please upload" warning on ``False``.
    """
    if "dm" in st.session_state:
        return True

    configured_path = get_data_path()
    if configured_path:
        with st.spinner(f"Loading data from `{configured_path}`…"):
            dm = handle_data_path_hc()
        if dm is not None:
            if show_toast:
                st.toast(f"Loaded data from `{configured_path}`.", icon="📂")
            return True

    try:
        with st.spinner("Loading sample data…"):
            st.session_state["dm"] = cached_sample()
            st.session_state["prediction"] = cached_sample_prediction()
            st.session_state["data_source"] = "CDH Sample"
    except Exception:
        logger.exception("Failed to auto-load CDH sample for Health Check")
        return False

    if show_toast:
        st.toast("Loaded sample data — upload your own to replace it.", icon="📊")
    return True


def ensure_dm() -> ADMDatamart:
    """Sub-page guard: auto-load ``dm`` or warn-and-stop.

    Re-applies sidebar branding (sub-page navigation drops it) and tries
    :func:`ensure_dm_loaded` first so deep-linked users don't hit a dead
    "please configure your files" warning.
    """
    _apply_sidebar_logo()
    if not ensure_dm_loaded(show_toast=True):
        st.warning("Please configure your files on the Home page.")
        st.stop()
    return st.session_state["dm"]


def _load_from_directory(dir_path: str) -> ADMDatamart | None:
    """Auto-detect HC datamart files in ``dir_path`` and populate session state.

    Detects a required model snapshot, optional predictor binning, and
    optional prediction table.

    Returns the loaded :class:`ADMDatamart` or ``None`` when no model
    snapshot is found in the directory.
    """
    model_match = pega_io.get_latest_file(dir_path, target="model_data")
    if model_match is None:
        st.error(
            f"Could not find an ADM model snapshot in `{dir_path}`. "
            "Health Check needs at least a model snapshot file in the directory."
        )
        return None

    predictor_match = pega_io.get_latest_file(dir_path, target="predictor_data")

    dm = cached_datamart(
        base_path=dir_path,
        model_filename=Path(model_match).name,
        predictor_filename=Path(predictor_match).name if predictor_match else None,
        extract_pyname_keys=st.session_state.get("extract_pyname_keys", True),
        infer_schema_length=st.session_state.get("infer_schema_length", 10000),
    )
    if dm is None:
        return None

    st.session_state["dm"] = dm

    prediction_match = pega_io.get_latest_file(dir_path, target="prediction_data")
    if prediction_match is not None:
        prediction = cached_prediction_table(
            predictions_filename=prediction_match,
            infer_schema_length=st.session_state.get("infer_schema_length", 10000),
        )
        if prediction is not None:
            st.session_state["prediction"] = prediction

    return dm


def handle_data_path_hc() -> ADMDatamart | None:
    """Load HC data from the ``--data-path`` CLI flag, if configured.

    Accepts either:

    - a **directory** containing the standard Pega ADM Datamart export
      files (model snapshot + predictor binning, optional prediction
      table). Auto-detected via :func:`pega_io.get_latest_file`.
    - a single **zip** file containing the same; extracted to a temp
      directory and treated as the directory case.

    Returns the loaded :class:`ADMDatamart` on success, or ``None`` when
    no path is configured, the path doesn't exist, or the path is an
    unsupported file type. The caller is expected to fall back to the
    bundled sample in the latter cases.
    """
    data_path = get_data_path()
    if not data_path:
        return None

    p = Path(data_path)
    if not p.exists():
        return None

    if p.is_dir():
        dm = _load_from_directory(str(p))
        if dm is not None:
            st.session_state["data_source"] = "Direct file path"
        return dm

    if p.is_file() and p.suffix.lower() == ".zip":
        tmp_dir = tempfile.mkdtemp(prefix="hc_path_")
        with st.spinner("Extracting archive..."):
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tmp_dir)
        dm = _load_from_directory(tmp_dir)
        if dm is not None:
            st.session_state["data_source"] = "Direct file path"
        return dm

    st.error(
        f"Health Check expects a directory or zip archive for `--data-path`, "
        f"got `{data_path}` ({p.suffix or 'no extension'}). "
        "Provide a directory containing a model snapshot (and optionally a "
        "predictor binning + prediction table), or a zip file containing the same."
    )
    return None
