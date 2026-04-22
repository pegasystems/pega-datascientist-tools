"""Health Check-specific Streamlit helpers.

Mirrors the layout of ``da_streamlit_utils.py`` and ``ia_streamlit_utils.py``
so each app's data-loading glue lives next to the app it serves. Shared
widgets/helpers stay in :mod:`pdstools.utils.streamlit_utils`.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from pdstools import pega_io
from pdstools.adm.ADMDatamart import ADMDatamart
from pdstools.utils.streamlit_utils import (
    cached_datamart,
    cached_prediction_table,
    get_data_path,
)

logger = logging.getLogger(__name__)


def _load_from_directory(dir_path: str) -> ADMDatamart | None:
    """Auto-detect HC datamart files in ``dir_path`` and populate session state.

    Mirrors ``streamlit_utils.from_file_path`` auto-detection: required
    model snapshot, required predictor binning (with model-only fallback),
    and optional prediction table.

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
