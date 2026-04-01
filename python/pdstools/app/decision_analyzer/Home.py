# python/pdstools/app/decision_analyzer/Home.py
import polars as pl
import streamlit as st

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    handle_data_path,
    handle_file_upload,
    handle_sample_data,
    load_decision_analyzer,
)
from pdstools.decision_analyzer.DecisionAnalyzer import DEFAULT_SAMPLE_SIZE
from pdstools.decision_analyzer.utils import (
    parse_filter_specs,
    parse_sample_flag,
    prepare_and_save,
    should_cache_source,
    _read_source_metadata,
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

# File upload — always visible. Drag-and-drop or use the Browse button.
raw_data, uploaded_metadata = handle_file_upload()
data_source_path = None  # Track the source path for metadata
sample_path = None  # Track the sampled file path if sampling is applied

# Check if we already have data loaded (used by all data loading logic below)
has_existing_data = "decision_data" in st.session_state

# If --data-path was provided, load from that path (takes priority over sample data)
configured_path = get_data_path()
configured_path_failed = False
if raw_data is None and configured_path and not has_existing_data:
    try:
        with st.spinner(f"Loading data from configured path: {configured_path}"):
            raw_data = handle_data_path()
            data_source_path = configured_path  # Capture the source
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


# Pre-ingestion data preparation (sampling or caching)
# Only apply --sample flag when loading from --data-path (not for file uploads)
sample_limit_raw = get_sample_limit() if data_source_path else None
filter_specs_raw = get_filter_specs() if data_source_path else None
if raw_data is not None:
    if filter_specs_raw:
        # Filter mode — apply before sampling
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
            # Collect filtered data so we can check for empty results and
            # avoid re-executing the filter scan during subsequent sampling.
            filtered_df = raw_data.filter(filter_expr).collect(streaming=True)
            filter_row_count = len(filtered_df)
            if filter_row_count == 0:
                st.warning(f"Filter matched 0 rows: {filter_desc}. Check your filter values.")
                st.stop()
            raw_data = filtered_df.lazy()

        st.info(f"🔍 Pre-ingestion filter applied: **{filter_desc}**. **{filter_row_count:,}** rows matched.")

    if sample_limit_raw:
        # Sampling mode
        try:
            sample_kwargs = parse_sample_flag(sample_limit_raw)
        except ValueError as e:
            st.error(f"Invalid --sample value: {e}")
            st.stop()

        # Build explicit sampling message based on parameters
        if "fraction" in sample_kwargs:
            sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the interactions…"
        elif "n" in sample_kwargs:
            sampling_msg = f"Sampling {sample_kwargs['n']:,} interactions…"
        else:
            sampling_msg = "Sampling interactions…"

        # Try to add time estimate to sampling message
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
        except Exception:
            pass  # Keep the simple message

        try:
            with st.spinner(sampling_msg):
                raw_data, sample_path = prepare_and_save(
                    raw_data,
                    n=sample_kwargs.get("n"),  # type: ignore[arg-type]
                    fraction=sample_kwargs.get("fraction"),  # type: ignore[arg-type]
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
        # Caching mode - save 100% of data from non-parquet sources
        with st.spinner("Caching data for faster reloading..."):
            raw_data, sample_path = prepare_and_save(
                raw_data,
                source_path=data_source_path,
                output_dir=".",  # Current working directory
            )

        if sample_path is not None:
            st.info(f"💾 Cached data saved to `{sample_path}` for faster reloading.")


def _show_data_summary(da):
    """Display a summary banner for the loaded DecisionAnalyzer."""
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

    # Check if data is sampled by looking for metadata in session state
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


if has_new_data and raw_data is not None:
    # New data provided — ingest, store, and reset filters
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

            # Check if the loaded data has sampling metadata
            # Priority: uploaded_metadata > sample_path > data_source_path
            sample_metadata = uploaded_metadata
            if sample_metadata is None:
                metadata_path = sample_path or data_source_path
                if metadata_path:
                    sample_metadata = _read_source_metadata(str(metadata_path))
            st.session_state["sample_metadata"] = sample_metadata

            del raw_data
            _show_data_summary(da)
    except BaseException as exc:
        if _is_capacity_error(exc):
            st.error(f"🚫 {type(exc).__name__}: {exc}")
            st.info(_LARGE_DATA_HINT)
            st.stop()
        raise  # re-raise unrelated errors
elif has_existing_data:
    # Returning to Home with data already loaded — just show the summary
    _show_data_summary(st.session_state.decision_data)
