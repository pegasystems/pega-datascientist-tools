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


def load_from_upload_auto(uploaded_file, outcome_labels_json: str | None = None) -> ImpactAnalyzer | None:
    """Load Impact Analyzer data with automatic format detection."""
    format_type = _detect_file_format(uploaded_file)

    if format_type == "pdc":
        return load_pdc_from_uploads([uploaded_file])
    elif format_type == "vbd":
        return load_vbd_from_upload(uploaded_file, outcome_labels_json=outcome_labels_json)
    else:
        # Fall back to extension-based detection
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".zip":
            return load_vbd_from_upload(uploaded_file, outcome_labels_json=outcome_labels_json)
        else:
            # Try PDC as default for .json/.ndjson
            return load_pdc_from_uploads([uploaded_file])


@st.cache_resource
def load_vbd_from_path(path: str, outcome_labels_json: str | None = None) -> ImpactAnalyzer | None:
    """Load VBD data. outcome_labels_json is a JSON-serialized outcome_labels dict (hashable for cache)."""
    import json

    outcome_labels = json.loads(outcome_labels_json) if outcome_labels_json else None
    return ImpactAnalyzer.from_vbd(path, outcome_labels=outcome_labels)


def load_vbd_from_upload(uploaded_file, outcome_labels_json: str | None = None) -> ImpactAnalyzer | None:
    path = _write_uploaded_file(uploaded_file)
    return load_vbd_from_path(path, outcome_labels_json=outcome_labels_json)


def _outcome_aliases_filename(source_path: str) -> str:
    """Return the sidecar filename for persisted outcome aliases."""
    return Path(source_path).name + ".outcome_aliases.json"


def _outcome_aliases_candidates(source_path: str) -> list[Path]:
    """Return candidate sidecar paths: next to source first, then current dir."""
    filename = _outcome_aliases_filename(source_path)
    return [
        Path(source_path).parent / filename,
        Path.cwd() / filename,
    ]


def save_outcome_aliases(source_path: str, config: dict | None) -> Path | None:
    """Save outcome alias config to a sidecar JSON file.

    Tries to write next to the data source first; falls back to the current
    working directory if the source directory is not writable.

    If config is None, removes sidecar files (revert to defaults).

    Returns the path where the file was saved, or None if removed/failed.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    candidates = _outcome_aliases_candidates(source_path)

    if config is None:
        for candidate in candidates:
            candidate.unlink(missing_ok=True)
        return None

    for candidate in candidates:
        try:
            candidate.write_text(json.dumps(config, indent=2))
            logger.debug("Saved outcome aliases to %s", candidate)
            return candidate
        except OSError:
            continue

    logger.warning("Could not save outcome aliases for %s", source_path)
    return None


def load_outcome_aliases(source_path: str) -> dict | None:
    """Load outcome alias config from a sidecar JSON file if it exists.

    Checks next to the data source first, then the current working directory.
    """
    import json

    for candidate in _outcome_aliases_candidates(source_path):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def discover_vbd_outcomes(ia: ImpactAnalyzer) -> dict[str, list[str]]:
    """Return {channel: [unique outcome values]} from already-loaded VBD ia_data.

    Relies on the Outcome list column kept for debugging in from_vbd().
    Returns empty dict if Outcome column is absent (e.g., PDC data).
    """

    schema = ia.ia_data.collect_schema()
    if "Outcome" not in schema.names():
        return {}

    rows = (
        ia.ia_data.select("Channel", "Outcome")
        .explode("Outcome")
        .select("Channel", "Outcome")
        .unique()
        .sort("Channel", "Outcome")
        .collect()
    )
    result: dict[str, list[str]] = {}
    for channel, outcome in rows.iter_rows():
        result.setdefault(channel, []).append(outcome)
    return result


def show_outcome_alias_config(ia: ImpactAnalyzer, source_path: str | None = None) -> dict | None:
    """Show per-channel outcome alias configuration UI for VBD data.

    Returns the outcome_labels dict to pass to from_vbd(), or None if
    the user has not made any changes from the defaults.

    Only renders for VBD data (when Outcome column is present).

    Parameters
    ----------
    ia : ImpactAnalyzer
        The loaded ImpactAnalyzer instance.
    source_path : str or None, optional
        Path to the data source file. When provided, aliases are persisted
        to a sidecar JSON file alongside the data so they are automatically
        loaded on the next session.
    """
    import json

    outcomes_by_channel = discover_vbd_outcomes(ia)
    if not outcomes_by_channel:
        return None  # PDC data or no Outcome column — skip

    default_impressions = ImpactAnalyzer.outcome_labels["Impressions"]
    default_accepts = ImpactAnalyzer.outcome_labels["Accepts"]

    # Use persisted aliases (from sidecar file or session state) as widget defaults
    saved_config = st.session_state.get("ia_outcome_labels")

    with st.expander("Configure outcome aliases (VBD data)", expanded=False):
        st.caption(
            "Map channel-specific outcome values to **Impressions** and **Accepts**. "
            "Channels not configured here use the default mapping. "
            "Click **Apply** after making changes to reload the data."
        )

        per_channel_config: dict = {}
        any_non_default = False

        for channel, outcomes in outcomes_by_channel.items():
            st.markdown(f"**{channel}**")
            col1, col2 = st.columns(2)

            # Use saved config for defaults if available, otherwise fall back to class defaults
            if saved_config and channel in saved_config:
                default_imp = [o for o in saved_config[channel].get("Impressions", []) if o in outcomes]
                default_acc = [o for o in saved_config[channel].get("Accepts", []) if o in outcomes]
            else:
                default_imp = [o for o in outcomes if o in default_impressions]
                default_acc = [o for o in outcomes if o in default_accepts]

            with col1:
                impressions = st.multiselect(
                    "Impressions",
                    options=outcomes,
                    default=default_imp,
                    key=f"outcome_imp_{channel}",
                )
            with col2:
                accepts = st.multiselect(
                    "Accepts",
                    options=outcomes,
                    default=default_acc,
                    key=f"outcome_acc_{channel}",
                )

            # Compare against class defaults (not saved config) to detect non-default state
            class_default_imp = [o for o in outcomes if o in default_impressions]
            class_default_acc = [o for o in outcomes if o in default_accepts]
            if sorted(impressions) != sorted(class_default_imp) or sorted(accepts) != sorted(class_default_acc):
                any_non_default = True

            per_channel_config[channel] = {"Impressions": impressions, "Accepts": accepts}

        if st.button("Apply outcome aliases"):
            config = per_channel_config if any_non_default else None
            st.session_state["ia_outcome_labels"] = config
            st.session_state["ia_outcome_labels_json"] = json.dumps(config) if config else None
            # Persist to sidecar file if source path is known
            _path = source_path or st.session_state.get("ia_data_source_path")
            if _path:
                save_outcome_aliases(_path, config)
            # Clear existing IA so next upload/path reload uses the new config
            st.session_state.pop("impact_analyzer", None)
            st.session_state.pop("ia_is_sample_data", None)
            st.rerun()

    return st.session_state.get("ia_outcome_labels")


def handle_data_path_ia() -> ImpactAnalyzer | None:
    """Load IA data from the ``--data-path`` CLI flag.

    Auto-detects format (JSON/NDJSON for PDC, ZIP for VBD).
    Automatically loads persisted outcome aliases from sidecar file if present.
    """
    import json

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
        # Auto-load persisted outcome aliases from sidecar file
        persisted = load_outcome_aliases(str(p))
        outcome_labels_json = json.dumps(persisted) if persisted else None
        if persisted:
            st.session_state["ia_outcome_labels"] = persisted
            st.session_state["ia_outcome_labels_json"] = outcome_labels_json
        return load_vbd_from_path(str(p), outcome_labels_json=outcome_labels_json)
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
