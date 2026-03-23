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
    import json

    path = _write_uploaded_file(uploaded_file)
    # Use the original filename in cwd for sidecar persistence (not the temp path)
    logical_path = str(Path.cwd() / uploaded_file.name)
    st.session_state["ia_data_source_path"] = logical_path
    # Auto-load persisted aliases from a previous session
    if not outcome_labels_json:
        persisted = load_outcome_aliases(logical_path)
        if persisted:
            outcome_labels_json = json.dumps(persisted)
            st.session_state["ia_outcome_labels"] = persisted
            st.session_state["ia_outcome_labels_json"] = outcome_labels_json
            # Find which sidecar file was loaded for UI feedback
            loaded_from = next((p for p in _outcome_aliases_candidates(logical_path) if p.exists()), None)
            st.session_state["ia_outcome_aliases_loaded_from"] = str(loaded_from) if loaded_from else logical_path
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


def discover_vbd_outcomes(ia: ImpactAnalyzer) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Return outcome values and record counts per channel from VBD ia_data.

    Relies on the Outcome list column kept for debugging in from_vbd().
    Returns empty dicts if Outcome column is absent (e.g., PDC data).

    Returns
    -------
    tuple[dict[str, list[str]], dict[str, int]]
        (outcomes_by_channel, records_by_channel)
    """
    import polars as pl

    schema = ia.ia_data.collect_schema()
    if "Outcome" not in schema.names():
        return {}, {}

    rows = (
        ia.ia_data.select("Channel", "Outcome")
        .explode("Outcome")
        .select("Channel", "Outcome")
        .unique()
        .sort("Channel", "Outcome")
        .collect()
    )
    outcomes: dict[str, list[str]] = {}
    for channel, outcome in rows.iter_rows():
        outcomes.setdefault(channel, []).append(outcome)

    counts = ia.ia_data.group_by("Channel").agg(pl.len().alias("n")).sort("Channel").collect()
    records: dict[str, int] = dict(counts.iter_rows())

    return outcomes, records


def show_outcome_labels_section(ia: ImpactAnalyzer, source_path: str | None = None) -> dict | None:
    """Show outcome label transparency table and optional per-channel editor for VBD data.

    Always renders a summary table of the active Impressions/Accepts mapping.
    An edit section is available in a collapsed expander below the table.

    Only renders for VBD data (when outcome_labels_used is set on the instance).

    Parameters
    ----------
    ia : ImpactAnalyzer
        The loaded ImpactAnalyzer instance.
    source_path : str or None, optional
        Path to the data source file for sidecar persistence.
    """
    import json

    import polars as pl

    outcome_labels_used = getattr(ia, "outcome_labels_used", None)
    if not outcome_labels_used:
        return None  # PDC data — no raw outcome labels

    outcomes_by_channel, records_by_channel = discover_vbd_outcomes(ia)

    loaded_from = st.session_state.pop("ia_outcome_aliases_loaded_from", None)
    if loaded_from:
        st.info(f"Applied previously stored outcome aliases from `{loaded_from}`")

    "### Outcome labels"
    st.caption(
        "These outcome values are counted as **Impressions** (denominator) and "
        "**Accepts** (numerator) for each channel. "
        "Verify they match your Pega configuration before analyzing results."
    )

    # Always-visible summary table
    rows = []
    for channel in sorted(outcome_labels_used):
        labels = outcome_labels_used[channel]
        imp_vals = labels.get("Impressions", [])
        acc_vals = labels.get("Accepts", [])
        imp_str = ", ".join(imp_vals) if imp_vals else "⚠ none matched"
        acc_str = ", ".join(acc_vals) if acc_vals else "⚠ none matched"
        rows.append(
            {
                "Channel": channel,
                "Records": records_by_channel.get(channel, 0),
                "Impressions": imp_str,
                "Accepts": acc_str,
            }
        )

    st.dataframe(
        pl.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    if any(
        not outcome_labels_used[ch].get("Impressions") or not outcome_labels_used[ch].get("Accepts")
        for ch in outcome_labels_used
    ):
        st.warning(
            "One or more channels have no matching outcome values for Impressions or Accepts. "
            "Those channels will show zero counts. Edit the labels below to fix this."
        )

    # Collapsed edit section
    saved_config = st.session_state.get("ia_outcome_labels")

    with st.expander("Edit outcome labels", expanded=False):
        st.caption(
            "Select which outcome values count as Impressions and Accepts for each channel. "
            "The selectable values are the distinct outcomes found in your data. "
            "The defaults shown next to each label are the Pega CDH standard outcomes "
            "for that channel type. If your data uses custom outcome names, "
            "select the appropriate values manually. "
            "Click **Apply** to reload with the new mapping."
        )

        # Upload a previously saved outcome definitions file
        uploaded_def = st.file_uploader(
            "Upload outcome definitions file",
            type=["json"],
            help=(
                "Upload a previously saved `.outcome_aliases.json` file to apply "
                "its definitions. Useful when the outcome mapping is client- or "
                "application-specific but the VBD export filename changes between downloads."
            ),
            key="ia_outcome_def_upload",
        )
        if uploaded_def is not None:
            # Guard: only process each upload once (file_id changes per upload)
            upload_id = f"{uploaded_def.name}_{uploaded_def.size}"
            if st.session_state.get("_ia_last_def_upload_id") != upload_id:
                try:
                    uploaded_config = json.loads(uploaded_def.getvalue())
                    if isinstance(uploaded_config, dict) and all(isinstance(v, dict) for v in uploaded_config.values()):
                        st.session_state["_ia_last_def_upload_id"] = upload_id
                        st.session_state["ia_outcome_labels"] = uploaded_config
                        st.session_state["ia_outcome_labels_json"] = json.dumps(uploaded_config)
                        _path = source_path or st.session_state.get("ia_data_source_path")
                        if _path:
                            save_outcome_aliases(_path, uploaded_config)
                        st.session_state.pop("impact_analyzer", None)
                        st.session_state.pop("ia_is_sample_data", None)
                        st.rerun()
                    else:
                        st.error(
                            "Invalid format. Expected a JSON object mapping channels to "
                            '`{"Impressions": [...], "Accepts": [...]}`.'
                        )
                except json.JSONDecodeError:
                    st.error("Could not parse the uploaded file as JSON.")

        st.divider()

        from pdstools.utils.pega_outcomes import get_channel_defaults

        per_channel_config: dict = {}

        for channel, outcomes in outcomes_by_channel.items():
            n = records_by_channel.get(channel, 0)
            st.markdown(f"**{channel}** *({n:,} records)*")
            col1, col2 = st.columns(2)

            current = (saved_config or outcome_labels_used).get(channel, {})
            default_imp = [o for o in current.get("Impressions", []) if o in outcomes]
            default_acc = [o for o in current.get("Accepts", []) if o in outcomes]

            ch_defaults = get_channel_defaults(channel)
            imp_hint = ", ".join(ch_defaults["Impressions"])
            acc_hint = ", ".join(ch_defaults["Accepts"])
            imp_label = f"Impressions (default: {imp_hint})" if imp_hint else "Impressions"
            acc_label = f"Accepts (default: {acc_hint})" if acc_hint else "Accepts"

            with col1:
                impressions = st.multiselect(
                    imp_label,
                    options=outcomes,
                    default=default_imp,
                    key=f"outcome_imp_{channel}",
                )
            with col2:
                accepts = st.multiselect(
                    acc_label,
                    options=outcomes,
                    default=default_acc,
                    key=f"outcome_acc_{channel}",
                )

            per_channel_config[channel] = {"Impressions": impressions, "Accepts": accepts}

        if st.button("Apply outcome labels"):
            config = per_channel_config
            st.session_state["ia_outcome_labels"] = config
            st.session_state["ia_outcome_labels_json"] = json.dumps(config)
            _path = source_path or st.session_state.get("ia_data_source_path")
            if _path:
                saved_to = save_outcome_aliases(_path, config)
                if saved_to:
                    st.session_state["ia_outcome_aliases_saved_to"] = str(saved_to)
                else:
                    st.session_state.pop("ia_outcome_aliases_saved_to", None)
            st.session_state.pop("impact_analyzer", None)
            st.session_state.pop("ia_is_sample_data", None)
            st.rerun()

        _path = source_path or st.session_state.get("ia_data_source_path")
        if _path:
            existing = _outcome_aliases_candidates(_path)
            saved_file = next((p for p in existing if p.exists()), None)
            if saved_file:
                st.caption(f"Aliases saved to `{saved_file}`")
            else:
                target = existing[0]
                st.caption(f"Aliases will be saved to `{target}`")
        elif st.session_state.get("ia_outcome_aliases_saved_to"):
            st.caption(f"Aliases saved to `{st.session_state['ia_outcome_aliases_saved_to']}`")

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
        st.session_state["ia_data_source_path"] = str(p)
        persisted = load_outcome_aliases(str(p))
        outcome_labels_json = json.dumps(persisted) if persisted else None
        if persisted:
            st.session_state["ia_outcome_labels"] = persisted
            st.session_state["ia_outcome_labels_json"] = outcome_labels_json
            loaded_from = next((c for c in _outcome_aliases_candidates(str(p)) if c.exists()), None)
            st.session_state["ia_outcome_aliases_loaded_from"] = str(loaded_from) if loaded_from else str(p)
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
