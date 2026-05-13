# python/pdstools/app/impact_analyzer/_home_page.py
"""Home page for the Impact Analyzer app.

Exposes a single ``home_page()`` callable that
``_navigation.build_navigation()`` registers as the default page in
``st.navigation()``. Kept as a function (not a script) so navigation
routes to it without re-executing top-level side-effects.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

VBD_REQUIRED_COLS = {
    "OutcomeTime",
    "Channel",
    "Direction",
    "AggregateCount",
    "Outcome",
    "Value",
    "MktValue",
}


def _load_with_warning(loader, label: str, *, expected_input_cols=None):
    required_cols = {
        "SnapshotTime",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "ValuePerImpression",
        "Channel",
    }
    try:
        ia = loader()
        if ia is None:
            return None
        try:
            schema_names = set(ia.ia_data.collect_schema().names())
        except Exception as exc:
            st.warning(f"Unable to validate {label} data schema. {exc}")
            return ia
        missing = sorted(required_cols.difference(schema_names))
        if missing:
            st.warning(
                f"{label} data is missing required columns: {', '.join(missing)}.",
            )
        return ia
    except Exception as exc:
        message = str(exc)
        if "Missing required columns" in message:
            st.warning(
                f"{label} data is missing required columns: {', '.join(sorted(required_cols))}. {message}",
            )
        elif "unable to find column" in message:
            # Extract the missing column name from the error message
            import re

            match = re.search(r'unable to find column "([^"]+)"', message)
            missing_col = match.group(1) if match else "unknown"

            # Use expected_input_cols if provided, otherwise default to VBD required columns
            required_cols_list = expected_input_cols or VBD_REQUIRED_COLS
            required_cols_str = ", ".join(f"`{col}`" for col in sorted(required_cols_list))

            st.error(
                f"**Cannot load this data** — missing required column: `{missing_col}`\n\n"
                "Impact Analyzer requires VBD data from **Scenario Planner Actuals** exports, "
                "which include control group information (`MktValue`). Standard VBD Actuals exports "
                "don't contain this data.\n\n"
                f"**Required columns:** {required_cols_str}"
            )
        else:
            st.warning(f"Unable to load {label} data. {message}")
        return None


def _banner_prefix(source_kind: str, source_label: str | None) -> str:
    """Build the banner prefix from source kind and label.

    Parameters
    ----------
    source_kind : str
        One of ``"sample"``, ``"upload"``, ``"cli"``.
    source_label : str or None
        Friendly source identifier — filename, path, or descriptor.

    Returns
    -------
    str
        Banner prefix, e.g. ``"Data loaded from `foo.zip`"``.
    """
    if source_kind == "sample":
        return "Sample data loaded"
    if source_kind in ("upload", "cli"):
        return f"Data loaded from `{source_label}`" if source_label else "Data loaded successfully"
    return "Data loaded successfully"


def _show_data_summary(
    ia,
    *,
    source_kind: str | None = None,
    source_label: str | None = None,
):
    """Display a summary banner for the loaded ImpactAnalyzer.

    Renders at most once per (source_kind, source_label, row count) combination
    so that navigation reruns do not re-emit the banner.

    Parameters
    ----------
    ia : ImpactAnalyzer
        The loaded analyzer.
    source_kind : str or None
        Tri-state source identifier (``"sample"``, ``"upload"``,
        ``"cli"``). Falls back to session state ``ia_data_source_kind``.
    source_label : str or None
        Friendly label (filename or path). Falls back to session state
        ``ia_data_source_label``.
    """
    import polars as pl

    if source_kind is None:
        source_kind = st.session_state.get("ia_data_source_kind", "sample")
    if source_label is None:
        source_label = st.session_state.get("ia_data_source_label")

    try:
        schema_names = set(ia.ia_data.collect_schema().names())
        # Check for VBD-specific columns after from_vbd() processing
        # (MktValue becomes ControlGroup, OutcomeTime becomes SnapshotTime,
        # and VBD data has Application, Value, Outcome columns)
        has_vbd_markers = {"Application", "Value", "Outcome"}.issubset(schema_names)

        format_label = "**VBD Scenario Planner**" if has_vbd_markers else "**PDC Export**"
    except (pl.exceptions.PolarsError, AttributeError, KeyError):
        format_label = "**Unknown format**"
        schema_names = set()

    try:
        rows = ia.ia_data.select(pl.len()).collect().item()
    except (pl.exceptions.PolarsError, AttributeError, KeyError):
        rows = None

    # Render-once guard: skip if we've already shown the banner for this exact
    # data fingerprint on this rerun cycle.
    fingerprint = (source_kind, source_label, rows)
    if st.session_state.get("_ia_banner_shown_for") == fingerprint:
        return
    st.session_state["_ia_banner_shown_for"] = fingerprint

    prefix = _banner_prefix(source_kind, source_label)

    if rows is None:
        st.success(f"{prefix}. Detected format: {format_label}")
        return

    try:
        channels = (
            ia.ia_data.select(pl.col("Channel").n_unique()).collect().item() if "Channel" in schema_names else "N/A"
        )
    except (pl.exceptions.PolarsError, AttributeError, KeyError):
        channels = "N/A"

    st.success(f"{prefix}. Detected format: {format_label}\n\n**{rows:,}** rows · **{channels}** channels")


def home_page() -> None:
    """Render the Impact Analyzer home page."""
    from pdstools.app.impact_analyzer.ia_streamlit_utils import (
        handle_data_path_ia,
        load_from_upload_auto,
        load_pdc_from_uploads,
        load_sample,
        prepare_and_save_random,
        show_outcome_labels_section,
    )
    from pdstools.utils.streamlit_utils import (
        get_data_path,
        get_sample_limit,
        get_temp_dir,
        parse_sample_spec,
        set_active_app as _set_active_app,
        show_sidebar_branding,
        show_version_header,
        standard_page_config,
    )

    standard_page_config(page_title="Impact Analyzer")
    show_sidebar_branding("Impact Analyzer")
    _set_active_app("ia")

    show_version_header()

    st.markdown(
        """
# Impact Analyzer

Analyze A/B test experiments from your decisioning monitoring exports or
scenario-planner data. Multiple input formats are supported — the format
is auto-detected on upload:

| | **Monitoring Export (JSON)** | **Monitoring Export (Excel)** | **Scenario Planner Export** | **Interaction History** |
|---|---|---|---|---|
| Format | JSON/NDJSON | XLSX | ZIP archive | TBD - future |

All charts are interactive ([Plotly](https://plotly.com/graphing-libraries/)) — pan,
zoom, and hover for details.

The bundled Infinity demo dataset is loaded by default — upload your own
file below to replace it.

### Data import
"""
    )

    impact_analyzer = None
    data_source_path = None
    data_source_kind: str | None = None
    data_source_label: str | None = None
    is_sample_data = False

    # ── File upload (always shown — uploads replace whatever is loaded) ──
    uploaded_files = st.file_uploader(
        "Upload Impact Analyzer data",
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Clear any old data when new upload is attempted
        if "impact_analyzer" in st.session_state:
            del st.session_state["impact_analyzer"]
        for key in (
            "ia_is_sample_data",
            "ia_data_source_kind",
            "ia_data_source_label",
            "_ia_banner_shown_for",
        ):
            st.session_state.pop(key, None)

        # Filter out unwanted files (e.g., MANIFEST.mf from unzipped Pega exports)
        valid_suffixes = {".json", ".ndjson", ".zip", ".xlsx"}
        filtered_files = [
            f for f in uploaded_files if Path(f.name).suffix.lower() in valid_suffixes and "META-INF" not in f.name
        ]

        if not filtered_files:
            st.error(
                "No valid data files found. Upload JSON/NDJSON (PDC), XLSX (PDC Excel), or ZIP (VBD) files. "
                "If you dragged a folder, try uploading just the `data.json` file instead."
            )
        else:
            suffixes = {Path(f.name).suffix.lower() for f in filtered_files}

            # Single file: use auto-detection
            if len(filtered_files) == 1:
                _outcome_labels_json = st.session_state.get("ia_outcome_labels_json")
                with st.spinner("Loading data (auto-detecting format)..."):
                    impact_analyzer = _load_with_warning(
                        lambda: load_from_upload_auto(filtered_files[0], outcome_labels_json=_outcome_labels_json),
                        "uploaded",
                        expected_input_cols=VBD_REQUIRED_COLS if ".zip" in suffixes else None,
                    )
                if impact_analyzer is not None:
                    data_source_kind = "upload"
                    data_source_label = filtered_files[0].name
            # Multiple JSON files: treat as PDC
            elif suffixes.issubset({".json", ".ndjson"}):
                with st.spinner("Loading PDC data"):
                    impact_analyzer = _load_with_warning(
                        lambda: load_pdc_from_uploads(filtered_files),
                        "PDC",
                    )
                if impact_analyzer is not None:
                    data_source_kind = "upload"
                    data_source_label = f"{len(filtered_files)} PDC files"
            else:
                st.error("Upload a single file (JSON/NDJSON/ZIP) or multiple JSON/NDJSON files (PDC).")

    # --- Autoload sample on first visit (no data yet, no upload) ---
    if impact_analyzer is None and "impact_analyzer" not in st.session_state:
        with st.spinner("Loading sample data"):
            impact_analyzer = _load_with_warning(load_sample, "Sample")
        if impact_analyzer is not None:
            is_sample_data = True
            data_source_kind = "sample"
            st.toast("Loaded bundled sample data — upload your own to replace it.", icon="📊")

    # --- Handle --data-path CLI flag (works with any source tab) ---
    has_existing_data = "impact_analyzer" in st.session_state
    configured_path = get_data_path()

    if impact_analyzer is None and configured_path and not has_existing_data:
        with st.spinner(f"Loading data from configured path: {configured_path}"):
            impact_analyzer = _load_with_warning(
                lambda: handle_data_path_ia(),
                "CLI",
            )
            data_source_path = configured_path
        if impact_analyzer is not None:
            data_source_kind = "cli"
            data_source_label = configured_path
            st.info(f"Loaded data from configured path: `{configured_path}`")

    # Pre-ingestion sampling (only for CLI paths)
    sample_limit_raw = get_sample_limit() if data_source_path else None
    if impact_analyzer is not None and sample_limit_raw:
        try:
            sample_kwargs = parse_sample_spec(sample_limit_raw)
        except ValueError as e:
            st.error(f"Invalid --sample value: {e}")
            st.stop()

        if "fraction" in sample_kwargs:
            sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the rows..."
        elif "n" in sample_kwargs:
            sampling_msg = f"Sampling {sample_kwargs['n']:,} rows..."
        else:
            sampling_msg = "Sampling rows..."

        with st.spinner(sampling_msg):
            sampled_data, sample_path = prepare_and_save_random(
                impact_analyzer.ia_data,
                n=sample_kwargs.get("n"),
                fraction=sample_kwargs.get("fraction"),
                output_dir=get_temp_dir() or ".",
                source_path=data_source_path,
            )

            if sample_path is not None:
                impact_analyzer.ia_data = sampled_data

        label = sample_limit_raw.strip()
        if sample_path is not None:
            st.info(f"Pre-ingestion sampling applied: keeping **{label}** rows. Sampled data saved to `{sample_path}`.")
        else:
            st.info(f"Sampling requested (**{label}**) but data already within limit — using full dataset.")

    # Store data and show summary
    if impact_analyzer is not None:
        st.session_state["impact_analyzer"] = impact_analyzer
        st.session_state["ia_is_sample_data"] = is_sample_data
        if data_source_kind is not None:
            st.session_state["ia_data_source_kind"] = data_source_kind
        if data_source_label is not None:
            st.session_state["ia_data_source_label"] = data_source_label
        if data_source_path:
            st.session_state["ia_data_source_path"] = data_source_path
        # Force banner to render once for this newly loaded data.
        st.session_state.pop("_ia_banner_shown_for", None)
        _show_data_summary(
            impact_analyzer,
            source_kind=data_source_kind,
            source_label=data_source_label,
        )
    elif "impact_analyzer" in st.session_state:
        # Show summary for previously loaded data (gated by render-once fingerprint).
        _show_data_summary(st.session_state["impact_analyzer"])

    # For VBD data: show outcome labels and handle reload
    _active_ia = impact_analyzer or st.session_state.get("impact_analyzer")
    if _active_ia is not None:
        show_outcome_labels_section(_active_ia)
