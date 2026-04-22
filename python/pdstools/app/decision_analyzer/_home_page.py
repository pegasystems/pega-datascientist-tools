# python/pdstools/app/decision_analyzer/_home_page.py
"""Home page for the Decision Analysis app.

Exposes a single ``render()`` callable that ``_navigation.build_navigation()``
registers as the default page in ``st.navigation()``. Kept as a function
(not a script) so navigation routes to it without re-executing top-level
side-effects, and so AppTest can exercise it in isolation.
"""

from __future__ import annotations

from typing import cast

import streamlit as st

_LARGE_DATA_HINT = (
    "**Your data may be too large for the default polars build.** "
    "Standard polars uses 32-bit indexing and cannot handle more than "
    "~2 billion elements.\n\n"
    "To fix this, install the 64-bit runtime extra:\n\n"
    "```\nuv pip install 'polars[rt64]'\n```\n\n"
    "Or reduce the data before loading with the `--sample` CLI flag:\n\n"
    "```\npdstools da --sample 500000 --data-path /path/to/data\n```"
)


def _is_capacity_error(exc: BaseException) -> bool:
    """Detect polars index-capacity or memory-related errors."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in ("capacity overflow", "index exceeds", "out of memory", "alloc"))


def _show_data_summary(da):
    """Display a summary banner for the loaded DecisionAnalyzer."""
    import polars as pl

    extract_type = da.extract_type
    format_label = (
        "**Explainability Extract (v1)** — arbitration stage only"
        if extract_type == "explainability_extract"
        else "**Action Analysis / EEV2 (v2)** — full pipeline"
    )

    overview = da.overview_stats
    rows = da.decision_data.select(pl.len()).collect().item()
    summary = (
        f"Data loaded successfully. Detected format: {format_label}\n\n"
        f"**{rows:,}** rows · "
        f"**{overview['Decisions']:,}** decisions · "
        f"**{overview['Customers']:,}** customers · "
        f"**{overview['Actions']}** actions · "
        f"**{overview['Channels']}** channels · "
        f"**{overview['Duration'].days}** days "
        f"(from {overview['StartDate']})"
    )
    st.success(summary)

    sample_metadata = st.session_state.get("sample_metadata")
    if sample_metadata:
        sample_pct = sample_metadata["sample_percentage"]
        source_file = sample_metadata.get("source_file", "unknown")

        # Skip the "100% of original" message when filter was applied —
        # the filter info banner already explains what happened, and
        # "100%" is misleading (it's 100% of the filtered data, not the original).
        if sample_pct < 100.0:
            st.info(
                f"📊 This data represents **{sample_pct:.2f}%** of the original dataset. "
                f"Original source: `{source_file}`"
            )


def _show_locked_pages_hint(locked_titles: list[str]) -> None:
    """Show users which analysis pages will unlock once data is loaded.

    Only rendered when no ``decision_data`` is in session state — gives
    discoverability of features that ``st.navigation()`` would otherwise
    hide entirely until prerequisites are met.
    """
    if not locked_titles:
        return
    pretty = "\n".join(f"- {title}" for title in locked_titles)
    st.info(
        f"📤 **Upload data above to unlock these analysis pages:**\n\n{pretty}",
    )


def render(locked_page_titles: list[str] | None = None) -> None:
    """Render the Decision Analysis home page.

    Parameters
    ----------
    locked_page_titles : list[str] or None
        Titles of pages currently hidden from navigation because no data
        is loaded. Surfaced as a discoverability hint so users know what
        will appear after they upload.
    """
    import polars as pl

    from pdstools.app.decision_analyzer.da_streamlit_utils import (
        handle_data_path,
        handle_file_upload,
        handle_sample_data,
        load_decision_analyzer,
    )
    from pdstools.decision_analyzer.DecisionAnalyzer import DEFAULT_SAMPLE_SIZE
    from pdstools.decision_analyzer.utils import (
        _read_source_metadata,
        parse_filter_specs,
        parse_sample_flag,
        prepare_and_save,
        should_cache_source,
    )
    from pdstools.utils.streamlit_utils import (
        get_data_path,
        get_filter_specs,
        get_sample_limit,
        get_temp_dir,
        show_sidebar_branding,
        show_version_header,
        standard_page_config,
    )

    standard_page_config(page_title="Decision Analysis")
    show_sidebar_branding("Decision Analysis")
    pl.enable_string_cache()

    show_version_header()

    st.markdown(
        """
# Decision Analysis

Visualize and diagnose your NBA decision pipeline. Two data formats are supported — the
format is auto-detected on upload:

| | **Explainability Extract (v1)** | **Action Analysis / EEV2 (v2)** |
|---|---|---|
| Scope | Arbitration stage only | Full decision pipeline |
| Stages | Synthetic (Arbitration → Output) | Real stages from strategy framework |
| Filter components | Not available | Available (which rule filtered what) |
| Use case | Quick arbitration analysis | Deep pipeline diagnostics |

All charts are interactive ([Plotly](https://plotly.com/graphing-libraries/)) — pan,
zoom, and hover for details.

### Data import
"""
    )

    level = "Stage Group"

    sample_size = st.number_input(
        "Distribution Overview Sample Size",
        min_value=1000,
        value=DEFAULT_SAMPLE_SIZE,
        step=1000,
        help=(
            "Number of interactions used to compute distribution overviews on certain pages. "
            "Affects distribution plots and histograms, not aggregate metrics or counts. "
            "This is an in-app calculation setting, unrelated to the `--sample` CLI flag "
            "(which reduces data before loading). "
            "Decrease for faster rendering; increase for smoother distributions."
        ),
    )

    raw_data, uploaded_metadata = handle_file_upload()
    data_source_path = None
    sample_path = None

    has_existing_data = "decision_data" in st.session_state

    configured_path = get_data_path()
    configured_path_failed = False
    if raw_data is None and configured_path and not has_existing_data:
        try:
            with st.spinner(f"Loading data from configured path: {configured_path}"):
                raw_data = handle_data_path()
                data_source_path = configured_path
        except Exception as exc:
            st.error(f"🚫 Failed to load data: {type(exc).__name__}: {exc}")
            configured_path_failed = True
            raw_data = None
        if raw_data is not None:
            st.info(f"📂 Loaded data from configured path: `{configured_path}`")
        elif not configured_path_failed:
            configured_path_failed = True

    has_new_data = raw_data is not None

    # Fall back to sample data only when:
    # 1. Nothing was uploaded, AND
    # 2. No data is loaded yet, AND
    # 3. No configured --data-path was attempted (or it would have shown error above)
    if not has_new_data and not has_existing_data and not configured_path_failed:
        with st.spinner("Loading sample data"):
            raw_data = handle_sample_data()
        has_new_data = raw_data is not None
        st.info(
            "No file uploaded — using built-in sample data. Upload your own data above to analyze it.",
        )
    elif configured_path_failed:
        st.error(
            f"Failed to load data from configured path: `{configured_path}`. "
            "Please check the path and file format, or upload data using the file uploader above."
        )
        st.stop()

    sample_limit_raw = get_sample_limit() if data_source_path else None
    filter_specs_raw = get_filter_specs() if data_source_path else None
    if raw_data is not None:
        if filter_specs_raw:
            try:
                filter_expr = parse_filter_specs(
                    filter_specs_raw,
                    available_columns=set(raw_data.collect_schema().names()),
                )
            except ValueError as e:
                st.error(f"Invalid --filter value: {e}")
                st.stop()

            filter_desc = " AND ".join(filter_specs_raw)
            with st.spinner(f"Filtering data: {filter_desc}"):
                filtered_df = raw_data.filter(filter_expr).collect(engine="streaming")
                filter_row_count = len(filtered_df)
                if filter_row_count == 0:
                    st.warning(f"Filter matched 0 rows: {filter_desc}. Check your filter values.")
                    st.stop()
                raw_data = filtered_df.lazy()

            st.info(f"🔍 Pre-ingestion filter applied: **{filter_desc}**. **{filter_row_count:,}** rows matched.")

        if sample_limit_raw:
            try:
                sample_kwargs = parse_sample_flag(sample_limit_raw)
            except ValueError as e:
                st.error(f"Invalid --sample value: {e}")
                st.stop()

            if "fraction" in sample_kwargs:
                sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the interactions…"
            elif "n" in sample_kwargs:
                sampling_msg = f"Sampling {sample_kwargs['n']:,} interactions…"
            else:
                sampling_msg = "Sampling interactions…"

            try:
                from pdstools.utils.progress_utils import (
                    estimate_sampling_time,
                    format_time_estimate,
                )

                row_count = raw_data.select(pl.len()).collect().item()
                target_n = sample_kwargs.get("n", int(row_count * sample_kwargs.get("fraction", 1.0)))
                min_time, max_time = estimate_sampling_time(row_count, target_n)
                time_msg = format_time_estimate(min_time, max_time)
                sampling_msg += f" (estimated: {time_msg})"
            except (ImportError, AttributeError, ValueError):
                pass

            try:
                with st.spinner(sampling_msg):
                    raw_data, sample_path = prepare_and_save(
                        raw_data,
                        n=cast("int | None", sample_kwargs.get("n")),
                        fraction=cast("float | None", sample_kwargs.get("fraction")),
                        output_dir=get_temp_dir(),
                        source_path=data_source_path,
                    )
            except BaseException as exc:
                if _is_capacity_error(exc):
                    st.error(f"🚫 Sampling failed: {type(exc).__name__}: {exc}")
                    st.info(_LARGE_DATA_HINT)
                    st.stop()
                raise

            label = sample_limit_raw.strip()
            if sample_path is not None:
                st.info(
                    f"📉 Pre-ingestion sampling applied: keeping **{label}** interactions. "
                    f"Sampled data saved to `{sample_path}`."
                )
            else:
                st.info(f"📉 Sampling requested (**{label}**) but data already within limit — using full dataset.")

        elif filter_specs_raw or should_cache_source(data_source_path):
            with st.spinner("Caching data for faster reloading..."):
                raw_data, sample_path = prepare_and_save(
                    raw_data,
                    source_path=data_source_path,
                    output_dir=".",
                )

            if sample_path is not None:
                st.info(f"💾 Cached data saved to `{sample_path}` for faster reloading.")

    if has_new_data and raw_data is not None:
        try:
            with st.spinner("Reading Data"):
                data_id = str(hash(raw_data.explain(optimized=False)))
                da = load_decision_analyzer(
                    raw_data,
                    level=level,
                    sample_size=sample_size,
                    data_fingerprint=data_id,
                )
                st.session_state.decision_data = da
                st.session_state["filters"] = []

                # Priority: uploaded_metadata > sample_path > data_source_path
                sample_metadata = uploaded_metadata
                if sample_metadata is None:
                    metadata_path = sample_path or data_source_path
                    if metadata_path:
                        sample_metadata = _read_source_metadata(str(metadata_path))
                st.session_state["sample_metadata"] = sample_metadata

                del raw_data
        except BaseException as exc:
            if _is_capacity_error(exc):
                st.error(f"🚫 {type(exc).__name__}: {exc}")
                st.info(_LARGE_DATA_HINT)
                st.stop()
            msg = str(exc)
            if "critical columns missing" in msg.lower() or ("missing" in msg.lower() and "column" in msg.lower()):
                st.error(
                    f"🚫 Cannot load this data: {exc}\n\n"
                    "Make sure you are uploading a **Decision Analyzer export** — either an "
                    "**Explainability Extract (v1)** or **Action Analysis / EEV2 (v2)** file from "
                    "Pega Infinity. Other data formats (ADM model snapshots, IH data, etc.) are not "
                    "supported here."
                )
                st.stop()
            if isinstance(exc, ValueError):
                st.error(
                    f"🚫 Cannot parse this file: {exc}\n\n"
                    "Check that the file is a valid Decision Analyzer export and is not corrupted."
                )
                st.stop()
            raise

        # First-run auto-load: ``st.navigation()`` was already built
        # earlier in this script run with the data-gated pages hidden,
        # so the sidebar wouldn't reveal them until the next user
        # interaction triggered a rerun. Match the HC / IA UX and
        # rerun once now so the unlocked nav and the data summary
        # appear in the same user action. Guard on ``has_existing_data``
        # so we rerun exactly once on the load and not on every
        # subsequent script run. Skip ``_show_data_summary`` here —
        # the post-rerun pass will render it via the ``has_existing_data``
        # branch below.
        if not has_existing_data:
            st.rerun()

        _show_data_summary(da)
    elif has_existing_data:
        _show_data_summary(st.session_state.decision_data)
    else:
        # No data, no upload, no sample fallback — show what awaits.
        _show_locked_pages_hint(locked_page_titles or [])
