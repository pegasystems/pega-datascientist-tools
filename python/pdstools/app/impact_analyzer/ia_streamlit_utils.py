# python/pdstools/app/impact_analyzer/ia_streamlit_utils.py
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import streamlit as st

from pdstools import ImpactAnalyzer
from pdstools.utils.streamlit_utils import (
    _apply_sidebar_logo,
    ensure_session_data,
    get_data_path,
)

SAMPLE_PDC_URL = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/ia/CDH_Metrics_ImpactAnalyzer.json"


def ensure_impact_analyzer() -> ImpactAnalyzer:
    """Guard: stop if Impact Analyzer data is not loaded.

    Re-applies sidebar branding on sub-pages.
    """
    _apply_sidebar_logo()
    ensure_session_data("impact_analyzer", "Please upload your data in the Home page.")
    return st.session_state["impact_analyzer"]


def _write_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".json"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _write_uploaded_files(uploaded_files: Iterable) -> list[str]:
    return [_write_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]


def _resolve_sample_pdc_path() -> Path:
    local_path = Path(__file__).resolve().parents[4] / "data" / "ia" / "CDH_Metrics_ImpactAnalyzer.json"
    if local_path.exists():
        return local_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        with urllib.request.urlopen(SAMPLE_PDC_URL) as response:
            tmp.write(response.read())
        return Path(tmp.name)


@st.cache_resource
def load_sample_pdc() -> ImpactAnalyzer:
    sample_path = _resolve_sample_pdc_path()
    return ImpactAnalyzer.from_pdc(str(sample_path))


@st.cache_resource
def load_pdc_from_paths(paths: tuple[str, ...]) -> ImpactAnalyzer:
    cleaned_paths = [path for path in paths if path]
    if len(cleaned_paths) == 1:
        return ImpactAnalyzer.from_pdc(cleaned_paths[0])
    return ImpactAnalyzer.from_pdc(cleaned_paths)


def load_pdc_from_uploads(uploaded_files: Iterable) -> ImpactAnalyzer:
    paths = _write_uploaded_files(uploaded_files)
    return load_pdc_from_paths(tuple(paths))


def _detect_file_format(uploaded_file) -> str:
    """Detect if file is PDC or VBD format by examining content.

    Returns: "pdc", "vbd", or "unknown"
    """
    import json

    # Read first few lines to detect format
    uploaded_file.seek(0)
    first_bytes = uploaded_file.read(500)
    uploaded_file.seek(0)

    try:
        first_line = first_bytes.decode("utf-8", errors="ignore").split("\n")[0]

        # Try to parse first line as JSON
        first_obj = json.loads(first_line)

        # PDC format has pxResults at top level
        if "pxResults" in first_obj:
            return "pdc"

        # VBD format (unzipped) has columns like OutcomeTime, MktValue, Channel, etc.
        vbd_indicators = {"OutcomeTime", "MktValue", "Channel", "AggregateCount", "Outcome"}
        if vbd_indicators.intersection(first_obj.keys()):
            return "vbd"

    except (json.JSONDecodeError, UnicodeDecodeError, IndexError):
        pass

    return "unknown"


def load_from_upload_auto(uploaded_file) -> ImpactAnalyzer | None:
    """Load Impact Analyzer data with automatic format detection."""
    format_type = _detect_file_format(uploaded_file)

    if format_type == "pdc":
        return load_pdc_from_uploads([uploaded_file])
    elif format_type == "vbd":
        return load_vbd_from_upload(uploaded_file)
    else:
        # Fall back to extension-based detection
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".zip":
            return load_vbd_from_upload(uploaded_file)
        else:
            # Try PDC as default for .json/.ndjson
            return load_pdc_from_uploads([uploaded_file])


@st.cache_resource
def load_vbd_from_path(path: str) -> ImpactAnalyzer | None:
    return ImpactAnalyzer.from_vbd(path)


def load_vbd_from_upload(uploaded_file) -> ImpactAnalyzer | None:
    path = _write_uploaded_file(uploaded_file)
    return load_vbd_from_path(path)


def handle_data_path_ia() -> ImpactAnalyzer | None:
    """Load IA data from the ``--data-path`` CLI flag.

    Auto-detects format (JSON/NDJSON for PDC, ZIP for VBD).
    """
    data_path = get_data_path()
    if not data_path:
        return None

    p = Path(data_path)
    if not p.exists():
        st.error(f"Configured data path does not exist: `{data_path}`")
        return None

    suffix = p.suffix.lower()
    if suffix in {".json", ".ndjson"}:
        return load_pdc_from_paths((str(p),))
    elif suffix == ".zip":
        return load_vbd_from_path(str(p))
    else:
        st.error(f"Unsupported file type: {suffix}. Use JSON/NDJSON (PDC) or ZIP (VBD).")
        return None


def prepare_and_save_random(
    data,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str = ".",
    source_path: str | None = None,
):
    """Apply simple random sampling and save to parquet.

    Parameters
    ----------
    data : pl.LazyFrame
        Input data to sample
    n : int, optional
        Absolute number of rows to sample
    fraction : float, optional
        Fraction of rows to sample (0.0 to 1.0)
    output_dir : str, default "."
        Directory to save sampled parquet file
    source_path : str, optional
        Original data source path (for metadata)

    Returns
    -------
    tuple[pl.LazyFrame, str | None]
        (sampled_data, output_path) where output_path is None if no sampling occurred
    """
    import polars as pl
    from datetime import datetime

    total_rows = data.select(pl.len()).collect().item()

    if n is not None:
        target_n = min(n, total_rows)
    elif fraction is not None:
        target_n = int(total_rows * fraction)
    else:
        raise ValueError("Must specify either n or fraction")

    if target_n >= total_rows:
        return data, None

    sampled = data.collect().sample(n=target_n, shuffle=True).lazy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = Path(source_path).stem if source_path else "data"
    output_filename = f"ia_sample_{source_name}_{target_n}_{timestamp}.parquet"
    output_path = Path(output_dir) / output_filename

    sampled.collect().write_parquet(output_path, use_pyarrow=True)

    return sampled, str(output_path)
