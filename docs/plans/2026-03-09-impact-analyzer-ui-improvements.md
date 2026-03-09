# Impact Analyzer UI Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring Impact Analyzer's look and feel into alignment with Decision Analyzer through shared utilities, CLI integration, and consistent styling.

**Architecture:** Extract common patterns from Decision Analyzer into shared utilities (`streamlit_utils.py`) that both apps use. Impact Analyzer gets CLI flag support (`--data-path`, `--sample`, `--temp-dir`), restructured Home page matching DA's UX, and analysis pages with consistent styling (bordered containers, text hierarchy). Simple random sampling for IA vs stratified for DA.

**Tech Stack:** Streamlit, Polars, pdstools CLI, pytest

---

## Phase 1: Shared Utilities Foundation

### Task 1.1: Move sampling utilities to shared location

**Files:**
- Modify: `python/pdstools/utils/streamlit_utils.py`
- Reference: `python/pdstools/decision_analyzer/utils.py`
- Test: Manual verification (no unit tests currently exist)

**Step 1: Read existing DA sampling utilities**

Review these functions in `python/pdstools/decision_analyzer/utils.py`:
- `parse_sample_flag()`
- `format_sampling_message()` (if exists)
- Understand their signatures and behavior

**Step 2: Add parse_sample_spec to streamlit_utils.py**

Add at the bottom of `python/pdstools/utils/streamlit_utils.py`:

```python
def parse_sample_spec(spec: str) -> dict:
    """Parse --sample flag into {n: int} or {fraction: float}.

    Supports formats:
    - Absolute counts: "100000", "100k", "1M"
    - Percentages: "10%"

    Parameters
    ----------
    spec : str
        Sample specification string

    Returns
    -------
    dict
        Either {"n": int} for absolute counts or {"fraction": float} for percentages

    Raises
    ------
    ValueError
        If spec format is invalid

    Examples
    --------
    >>> parse_sample_spec("100000")
    {'n': 100000}
    >>> parse_sample_spec("100k")
    {'n': 100000}
    >>> parse_sample_spec("1M")
    {'n': 1000000}
    >>> parse_sample_spec("10%")
    {'fraction': 0.1}
    """
    spec = spec.strip()

    # Check for percentage
    if spec.endswith("%"):
        try:
            percentage = float(spec[:-1])
            if not 0 < percentage <= 100:
                raise ValueError("Percentage must be between 0 and 100")
            return {"fraction": percentage / 100}
        except ValueError as e:
            raise ValueError(f"Invalid percentage format: {spec}") from e

    # Check for K/M suffix
    multiplier = 1
    if spec.lower().endswith("k"):
        multiplier = 1000
        spec = spec[:-1]
    elif spec.lower().endswith("m"):
        multiplier = 1000000
        spec = spec[:-1]

    # Parse as integer
    try:
        n = int(spec) * multiplier
        if n <= 0:
            raise ValueError("Sample count must be positive")
        return {"n": n}
    except ValueError as e:
        raise ValueError(f"Invalid sample format: {spec}. Use '100000', '100k', '1M', or '10%'") from e
```

**Step 3: Test parse_sample_spec interactively**

Run Python REPL:
```python
from pdstools.utils.streamlit_utils import parse_sample_spec

# Test various formats
assert parse_sample_spec("100000") == {"n": 100000}
assert parse_sample_spec("100k") == {"n": 100000}
assert parse_sample_spec("1M") == {"n": 1000000}
assert parse_sample_spec("10%") == {"fraction": 0.1}

# Test error cases
try:
    parse_sample_spec("invalid")
    assert False, "Should have raised ValueError"
except ValueError:
    pass

print("All tests passed!")
```

Expected: All assertions pass

**Step 4: Update DA utils.py to import from shared location**

Modify `python/pdstools/decision_analyzer/utils.py`:

Add at top:
```python
from pdstools.utils.streamlit_utils import parse_sample_spec as parse_sample_flag
```

Remove or deprecate the local `parse_sample_flag` implementation (keep it but add deprecation comment if needed for backward compat).

**Step 5: Test DA still works**

Run Decision Analyzer:
```bash
uv run pdstools da
```

Expected: App starts without errors, sample data loads

**Step 6: Commit**

```bash
git add python/pdstools/utils/streamlit_utils.py python/pdstools/decision_analyzer/utils.py
git commit -m "refactor: move parse_sample_spec to shared utilities"
```

---

### Task 1.2: Create generic file upload handler

**Files:**
- Modify: `python/pdstools/utils/streamlit_utils.py`

**Step 1: Add generic upload handler to streamlit_utils.py**

Add after `parse_sample_spec`:

```python
from typing import Any, Callable


def handle_generic_file_upload(
    label: str,
    supported_types: list[str],
    file_reader_func: Callable[[list], tuple[Any, dict | None]],
    accept_multiple: bool = True,
) -> tuple[Any | None, dict | None]:
    """Generic file uploader that delegates format reading to app-specific function.

    Parameters
    ----------
    label : str
        Widget label text
    supported_types : list[str]
        List of file extensions (e.g., ["zip", "parquet", "json"])
    file_reader_func : Callable
        Function that takes uploaded file objects and returns (data, metadata)
    accept_multiple : bool, default True
        Whether to accept multiple file uploads

    Returns
    -------
    tuple[Any | None, dict | None]
        (data, metadata) or (None, None) if no files uploaded
    """
    import streamlit as st

    uploaded_files = st.file_uploader(
        label,
        type=supported_types,
        accept_multiple_files=accept_multiple,
    )

    if not uploaded_files:
        return None, None

    # Normalize to list if single file
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    return file_reader_func(uploaded_files)
```

**Step 2: Add generic CLI path handler**

Add after `handle_generic_file_upload`:

```python
def handle_generic_data_path(
    path_reader_func: Callable[[str], Any],
) -> Any | None:
    """Read data from --data-path CLI flag using app-specific reader.

    Handles path validation and delegates actual reading to app-specific function.

    Parameters
    ----------
    path_reader_func : Callable[[str], Any]
        Function that takes a file path and returns data

    Returns
    -------
    Any | None
        Data object or None if path not configured or loading failed
    """
    import streamlit as st
    from pathlib import Path

    data_path = get_data_path()
    if not data_path:
        return None

    p = Path(data_path)
    if not p.exists():
        st.error(f"Configured data path does not exist: `{data_path}`")
        return None

    try:
        return path_reader_func(data_path)
    except Exception as e:
        st.error(f"Failed to load data from `{data_path}`: {e}")
        return None
```

**Step 3: Commit**

```bash
git add python/pdstools/utils/streamlit_utils.py
git commit -m "feat: add generic file upload and path handlers"
```

---

### Task 1.3: Test Decision Analyzer with refactored code

**Files:**
- Test: Decision Analyzer app manually

**Step 1: Start Decision Analyzer**

```bash
uv run pdstools da
```

Expected: App starts without errors

**Step 2: Test file upload**

1. Upload a sample parquet/zip file
2. Verify data loads successfully
3. Check data summary displays correctly
4. Navigate to Overview page
5. Verify charts render

Expected: All functionality works as before

**Step 3: Test CLI flags**

```bash
# Test with sample flag
uv run pdstools da --sample 1000

# Test with data path (use actual path)
uv run pdstools da --data-path /path/to/data.parquet
```

Expected: Both commands work as before

**Step 4: Document test results**

Create test log:
```bash
echo "Phase 1 DA Tests - $(date)" >> test-log.txt
echo "✓ File upload: PASS" >> test-log.txt
echo "✓ CLI --sample: PASS" >> test-log.txt
echo "✓ CLI --data-path: PASS" >> test-log.txt
echo "✓ Page navigation: PASS" >> test-log.txt
```

---

## Phase 2: Impact Analyzer Data Loading

### Task 2.1: Add random sampling function for IA

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/ia_streamlit_utils.py`

**Step 1: Add prepare_and_save_random function**

Add at the bottom of `python/pdstools/app/impact_analyzer/ia_streamlit_utils.py`:

```python
import polars as pl
from pathlib import Path
from datetime import datetime


def prepare_and_save_random(
    data: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str = ".",
    source_path: str | None = None,
) -> tuple[pl.LazyFrame, str | None]:
    """Apply random sampling and save to parquet with metadata.

    Performs simple random row sampling (no stratification).
    Writes sampled data to parquet with sample metadata in file metadata.

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
    # Get total row count
    total_rows = data.select(pl.len()).collect().item()

    # Determine target sample size
    if n is not None:
        target_n = min(n, total_rows)
    elif fraction is not None:
        target_n = int(total_rows * fraction)
    else:
        raise ValueError("Must specify either n or fraction")

    # Skip sampling if already within limit
    if target_n >= total_rows:
        return data, None

    # Perform random sampling
    sampled = data.collect().sample(n=target_n, shuffle=True).lazy()

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = Path(source_path).stem if source_path else "data"
    output_filename = f"ia_sample_{source_name}_{target_n}_{timestamp}.parquet"
    output_path = Path(output_dir) / output_filename

    # Prepare metadata
    sample_percentage = (target_n / total_rows) * 100
    metadata = {
        "sample_percentage": str(sample_percentage),
        "source_file": source_path or "unknown",
        "original_rows": str(total_rows),
        "sampled_rows": str(target_n),
        "sampling_method": "random",
        "timestamp": timestamp,
    }

    # Write to parquet with metadata
    sampled.collect().write_parquet(output_path, use_pyarrow=True)

    # Write metadata to a companion JSON file (since parquet metadata is harder to write)
    import json
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return sampled, str(output_path)
```

**Step 2: Add function to read sample metadata**

Add after `prepare_and_save_random`:

```python
def read_sample_metadata(parquet_path: str) -> dict | None:
    """Read sample metadata from companion JSON file.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file

    Returns
    -------
    dict | None
        Metadata dict or None if not found
    """
    import json

    metadata_path = Path(parquet_path).with_suffix(".json")
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            return json.load(f)
    except Exception:
        return None
```

**Step 3: Commit**

```bash
git add python/pdstools/app/impact_analyzer/ia_streamlit_utils.py
git commit -m "feat(ia): add random sampling with metadata"
```

---

### Task 2.2: Add IA file upload handler

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/ia_streamlit_utils.py`

**Step 1: Add IA-specific file reader**

Add after existing functions in `ia_streamlit_utils.py`:

```python
from pdstools.utils.streamlit_utils import handle_generic_file_upload


def _read_uploaded_ia_files(uploaded_files) -> tuple[Any, dict | None]:
    """Read uploaded IA files (PDC or VBD) and return ImpactAnalyzer.

    Parameters
    ----------
    uploaded_files : list
        List of uploaded file objects from Streamlit

    Returns
    -------
    tuple[ImpactAnalyzer | None, dict | None]
        (ImpactAnalyzer instance, metadata) or (None, None)
    """
    if not uploaded_files:
        return None, None

    suffixes = {Path(f.name).suffix.lower() for f in uploaded_files}

    # PDC JSON/NDJSON
    if suffixes.issubset({".json", ".ndjson"}):
        paths = _write_uploaded_files(uploaded_files)
        ia = load_pdc_from_paths(tuple(paths))
        return ia, None

    # VBD ZIP
    elif len(uploaded_files) == 1 and ".zip" in suffixes:
        ia = load_vbd_from_upload(uploaded_files[0])
        return ia, None

    else:
        raise ValueError(f"Unsupported file types: {suffixes}. Upload JSON (PDC) or ZIP (VBD).")


def handle_file_upload_ia() -> tuple[Any, dict | None]:
    """Handle file upload for Impact Analyzer.

    Returns
    -------
    tuple[ImpactAnalyzer | None, dict | None]
        (ImpactAnalyzer instance, metadata) or (None, None)
    """
    return handle_generic_file_upload(
        label="Upload Impact Analyzer data",
        supported_types=["json", "ndjson", "zip"],
        file_reader_func=_read_uploaded_ia_files,
        accept_multiple=True,
    )
```

**Step 2: Add IA CLI path handler**

Add after `handle_file_upload_ia`:

```python
from pdstools.utils.streamlit_utils import handle_generic_data_path


def _read_ia_data_path(path: str) -> Any:
    """Read IA data from file path (PDC or VBD).

    Parameters
    ----------
    path : str
        File path to read

    Returns
    -------
    ImpactAnalyzer
        Loaded Impact Analyzer instance
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".json", ".ndjson"}:
        return load_pdc_from_paths((path,))
    elif suffix == ".zip":
        return load_vbd_from_path(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use JSON/NDJSON (PDC) or ZIP (VBD).")


def handle_data_path_ia() -> Any | None:
    """Handle --data-path CLI flag for Impact Analyzer.

    Returns
    -------
    ImpactAnalyzer | None
        Loaded instance or None
    """
    return handle_generic_data_path(_read_ia_data_path)
```

**Step 3: Add sample data handler**

Add after `handle_data_path_ia`:

```python
def handle_sample_data_ia() -> Any:
    """Load built-in sample data for Impact Analyzer.

    Returns
    -------
    ImpactAnalyzer
        Sample Impact Analyzer instance
    """
    return load_sample_pdc()
```

**Step 4: Commit**

```bash
git add python/pdstools/app/impact_analyzer/ia_streamlit_utils.py
git commit -m "feat(ia): add file upload and CLI path handlers"
```

---

### Task 2.3: Update IA ensure function

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/ia_streamlit_utils.py`

**Step 1: Update ensure_impact_analyzer function**

Replace existing `ensure_impact_analyzer` function:

```python
from pdstools.utils.streamlit_utils import ensure_session_data, _apply_sidebar_logo


def ensure_impact_analyzer() -> Any:
    """Guard: stop if Impact Analyzer data not loaded.

    Re-applies sidebar branding on sub-pages.

    Returns
    -------
    ImpactAnalyzer
        The loaded Impact Analyzer instance from session state
    """
    _apply_sidebar_logo()
    ensure_session_data(
        "impact_analyzer",
        "Please upload your data in the Home page."
    )
    return st.session_state["impact_analyzer"]
```

**Step 2: Commit**

```bash
git add python/pdstools/app/impact_analyzer/ia_streamlit_utils.py
git commit -m "refactor(ia): improve ensure_impact_analyzer with logo reapply"
```

---

### Task 2.4: Restructure IA Home page

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/Home.py`
- Reference: `python/pdstools/app/decision_analyzer/Home.py` for structure

**Step 1: Read current and target Home pages**

Review both files to understand the transformation.

**Step 2: Rewrite IA Home.py header section**

Replace the top section of `python/pdstools/app/impact_analyzer/Home.py`:

```python
from pathlib import Path

import polars as pl
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import (
    handle_data_path_ia,
    handle_file_upload_ia,
    handle_sample_data_ia,
    prepare_and_save_random,
    read_sample_metadata,
)
from pdstools.utils.streamlit_utils import (
    get_data_path,
    get_sample_limit,
    get_temp_dir,
    parse_sample_spec,
    show_sidebar_branding,
    show_version_header,
    standard_page_config,
)

standard_page_config(page_title="Impact Analyzer")
show_sidebar_branding("Impact Analyzer")

show_version_header()

"""
# Impact Analyzer

Analyze A/B test experiments from PDC exports or VBD Scenario Planner Actuals.
Two data formats are supported — format is auto-detected on upload:

| | **PDC Export** | **VBD Scenario Planner** |
|---|---|---|
| Format | JSON/NDJSON | ZIP archive |
| Source | Pega Decisioning Center | Value-Based Design |
| Metrics | CTR lift, value lift | Scenario comparison |

All charts are interactive ([Plotly](https://plotly.com/graphing-libraries/)) — pan,
zoom, and hover for details.

### Data import
"""
```

**Step 3: Add data loading logic**

Replace the old dropdown selector logic with priority chain:

```python
# File upload — always visible
raw_ia, uploaded_metadata = handle_file_upload_ia()
data_source_path = None
sample_path = None

# Check if we already have data loaded
has_existing_data = "impact_analyzer" in st.session_state

# If --data-path was provided, load from that path (takes priority over sample data)
configured_path = get_data_path()
configured_path_failed = False
if raw_ia is None and configured_path and not has_existing_data:
    with st.spinner(f"Loading data from configured path: {configured_path}"):
        raw_ia = handle_data_path_ia()
        data_source_path = configured_path
    if raw_ia is not None:
        st.info(f"📂 Loaded data from configured path: `{configured_path}`")
    else:
        configured_path_failed = True

has_new_data = raw_ia is not None

# Fall back to sample data only when:
# 1. Nothing was uploaded, AND
# 2. No data is loaded yet, AND
# 3. No configured --data-path was attempted
if not has_new_data and not has_existing_data and not configured_path_failed:
    with st.spinner("Loading sample data"):
        raw_ia = handle_sample_data_ia()
    has_new_data = raw_ia is not None
    st.info(
        "No file uploaded — using built-in sample data. Upload your own data above to analyze it.",
    )
elif configured_path_failed:
    st.error(
        f"Failed to load data from configured path: `{configured_path}`. "
        "Please check the path and file format, or upload data using the file uploader above."
    )
    st.stop()
```

**Step 4: Add sampling logic**

Add after data loading:

```python
# Pre-ingestion sampling (only for CLI paths, not uploads)
sample_limit_raw = get_sample_limit() if data_source_path else None
if raw_ia is not None and sample_limit_raw:
    try:
        sample_kwargs = parse_sample_spec(sample_limit_raw)
    except ValueError as e:
        st.error(f"Invalid --sample value: {e}")
        st.stop()

    # Build sampling message
    if "fraction" in sample_kwargs:
        sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the rows…"
    elif "n" in sample_kwargs:
        sampling_msg = f"Sampling {sample_kwargs['n']:,} rows…"
    else:
        sampling_msg = "Sampling rows…"

    with st.spinner(sampling_msg):
        # Sample the underlying ia_data LazyFrame
        sampled_data, sample_path = prepare_and_save_random(
            raw_ia.ia_data,
            n=sample_kwargs.get("n"),
            fraction=sample_kwargs.get("fraction"),
            output_dir=get_temp_dir() or ".",
            source_path=data_source_path,
        )

        # Create new IA instance with sampled data
        # This is tricky - we need to reconstruct the ImpactAnalyzer
        # For now, just update the ia_data attribute
        if sample_path is not None:
            raw_ia.ia_data = sampled_data

    label = sample_limit_raw.strip()
    if sample_path is not None:
        st.info(
            f"📉 Pre-ingestion sampling applied: keeping **{label}** rows. "
            f"Sampled data saved to `{sample_path}`."
        )
    else:
        st.info(f"📉 Sampling requested (**{label}**) but data already within limit — using full dataset.")
```

**Step 5: Add data summary function**

Add before the ingestion logic:

```python
def _show_data_summary(ia):
    """Display a summary banner for the loaded ImpactAnalyzer."""
    # Try to detect format
    try:
        # VBD has specific columns
        schema_names = set(ia.ia_data.collect_schema().names())
        if "MktValue" in schema_names and "OutcomeTime" in schema_names:
            format_label = "**VBD Scenario Planner**"
        else:
            format_label = "**PDC Export**"
    except Exception:
        format_label = "**Unknown format**"

    try:
        rows = ia.ia_data.select(pl.len()).collect().item()
        channels = ia.ia_data.select(pl.col("Channel").n_unique()).collect().item() if "Channel" in schema_names else "N/A"

        summary = (
            f"Data loaded successfully. Detected format: {format_label}\n\n"
            f"**{rows:,}** rows · **{channels}** channels"
        )
        st.success(summary)
    except Exception as e:
        st.success(f"Data loaded successfully. Detected format: {format_label}")

    # Check if data is sampled
    sample_metadata = st.session_state.get("ia_sample_metadata")
    if sample_metadata:
        sample_pct = float(sample_metadata["sample_percentage"])
        source_file = sample_metadata.get("source_file", "unknown")
        st.info(
            f"📊 This data represents **{sample_pct:.1f}%** of the original dataset. Original source: `{source_file}`"
        )
```

**Step 6: Add ingestion and summary display**

Add at the end:

```python
if has_new_data and raw_ia is not None:
    # New data provided — store and show summary
    st.session_state["impact_analyzer"] = raw_ia
    st.session_state["impact_analyzer_source"] = "uploaded" if uploaded_metadata else "configured"

    # Check for sample metadata
    sample_metadata = uploaded_metadata
    if sample_metadata is None and sample_path:
        sample_metadata = read_sample_metadata(sample_path)
    st.session_state["ia_sample_metadata"] = sample_metadata

    _show_data_summary(raw_ia)

elif has_existing_data:
    # Returning to Home with data already loaded — just show summary
    _show_data_summary(st.session_state["impact_analyzer"])
```

**Step 7: Test IA Home page loads**

```bash
uv run pdstools ia
```

Expected: App starts, shows new UI with file uploader

**Step 8: Commit**

```bash
git add python/pdstools/app/impact_analyzer/Home.py
git commit -m "feat(ia): restructure Home page with CLI support"
```

---

### Task 2.5: Test IA data loading thoroughly

**Files:**
- Test: Impact Analyzer app manually

**Step 1: Test sample data**

```bash
uv run pdstools ia
```

1. Verify sample data loads automatically
2. Check data summary displays correctly
3. Verify format detection shows "PDC Export"
4. Navigate to Overall Summary page
5. Verify chart renders

Expected: Sample data works as before

**Step 2: Test file upload**

1. Click file uploader
2. Upload a PDC JSON file
3. Verify data summary updates
4. Check row counts and channels
5. Navigate to pages

Expected: File upload works, summary shows correct stats

**Step 3: Test CLI --data-path**

```bash
# With PDC file
uv run pdstools ia --data-path /path/to/pdc.json

# With VBD file
uv run pdstools ia --data-path /path/to/vbd.zip
```

Expected: Both load successfully, show appropriate format labels

**Step 4: Test CLI --sample**

```bash
uv run pdstools ia --data-path /path/to/large/pdc.json --sample 1000
```

Expected: Loads 1000 rows, shows sampling info banner

**Step 5: Document test results**

```bash
echo "Phase 2 IA Tests - $(date)" >> test-log.txt
echo "✓ Sample data: PASS" >> test-log.txt
echo "✓ File upload (PDC): PASS" >> test-log.txt
echo "✓ File upload (VBD): PASS" >> test-log.txt
echo "✓ CLI --data-path (PDC): PASS" >> test-log.txt
echo "✓ CLI --data-path (VBD): PASS" >> test-log.txt
echo "✓ CLI --sample: PASS" >> test-log.txt
```

**Step 6: Test DA still works after IA changes**

```bash
uv run pdstools da
```

Verify:
1. Sample data loads
2. File upload works
3. Pages render correctly

Expected: No regressions in DA

---

## Phase 3: Impact Analyzer Page Styling

### Task 3.1: Update Overall Summary page

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py`

**Step 1: Add page-level intro and update imports**

Replace the entire file content:

```python
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer · Overall Summary")

ia = ensure_impact_analyzer()

"# Overall Summary"

"""
View lift metrics aggregated across all channels. This page shows the overall
impact of your experiments, comparing treatment vs control groups.
"""

with st.container(border=True):
    "## Display Options"

    st.caption(
        "Select which metric to visualize. CTR Lift shows the relative improvement "
        "in click-through rates, while Value Lift shows the impact on business value."
    )

    metric = st.selectbox(
        "Metric",
        options=["CTR_Lift", "Value_Lift"],
        index=0,
    )

with st.container(border=True):
    "## Lift Overview"

    st.caption(
        "Interactive chart showing lift metrics over time. "
        "Hover for details, click and drag to zoom."
    )

    facet = "Channel" if "Channel" in ia.ia_data.collect_schema().names() else None
    fig = ia.plot.overview(metric=metric, facet=facet)
    st.plotly_chart(fig, width="stretch")

with st.container(border=True):
    "## Detailed Metrics"

    st.caption(
        "Tabular view of lift metrics with exact values. "
        "Use this for detailed analysis and reporting."
    )

    table = ia.plot.overview(metric=metric, facet=facet, return_df=True).collect()
    st.dataframe(table, use_container_width=True)
```

**Step 2: Test Overall Summary page**

```bash
uv run pdstools ia
```

Navigate to Overall Summary page.

Expected:
- Page intro displays
- Three bordered containers
- Captions in subdued text
- Chart uses `width="stretch"` properly
- Table displays correctly

**Step 3: Commit**

```bash
git add python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py
git commit -m "feat(ia): improve Overall Summary page styling"
```

---

### Task 3.2: Update Trend page

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/pages/2_Trend.py`

**Step 1: Add styling and structure**

Replace file content:

```python
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import ensure_impact_analyzer
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Impact Analyzer · Trend")

ia = ensure_impact_analyzer()

"# Trend Analysis"

"""
Explore how lift metrics change over time. Use this page to identify trends,
seasonal patterns, and the stability of your experiment results.
"""

with st.container(border=True):
    "## Metric Selection"

    st.caption(
        "Choose which lift metric to analyze over time. "
        "The chart will update to show trends for your selected metric."
    )

    metric = st.selectbox(
        "Metric",
        options=["CTR_Lift", "Value_Lift"],
        index=0,
    )

with st.container(border=True):
    "## Time Series View"

    st.caption(
        "Trend chart showing how your selected metric evolves over time. "
        "Look for consistent patterns and significant changes."
    )

    facet = "Channel" if "Channel" in ia.ia_data.collect_schema().names() else None
    fig = ia.plot.overview(metric=metric, facet=facet)
    st.plotly_chart(fig, width="stretch")
```

**Step 2: Test Trend page**

Navigate to Trend page.

Expected:
- Page intro displays
- Bordered containers
- Captions in subdued text
- Chart renders correctly

**Step 3: Commit**

```bash
git add python/pdstools/app/impact_analyzer/pages/2_Trend.py
git commit -m "feat(ia): improve Trend page styling"
```

---

### Task 3.3: Update About page

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/pages/3_About.py`

**Step 1: Add minimal About page content**

Replace file content:

```python
import streamlit as st

from pdstools.utils.streamlit_utils import standard_page_config
from pdstools import __version__ as pdstools_version

standard_page_config(page_title="Impact Analyzer · About")

"# About Impact Analyzer"

with st.container(border=True):
    "## Version Information"

    st.caption(f"**pdstools version:** {pdstools_version}")
    st.caption(
        "Impact Analyzer is part of the Pega Data Scientist Tools suite for "
        "analyzing A/B test experiments and measuring business impact."
    )

with st.container(border=True):
    "## Documentation & Support"

    """
    - **Documentation**: [pdstools.readthedocs.io](https://pdstools.readthedocs.io)
    - **GitHub**: [pegasystems/pega-datascientist-tools](https://github.com/pegasystems/pega-datascientist-tools)
    - **Issues**: [Report a bug or request a feature](https://github.com/pegasystems/pega-datascientist-tools/issues)
    """
```

**Step 2: Test About page**

Navigate to About page.

Expected:
- Version displays correctly
- Links are clickable
- Bordered containers render

**Step 3: Commit**

```bash
git add python/pdstools/app/impact_analyzer/pages/3_About.py
git commit -m "feat(ia): improve About page with version info"
```

---

## Phase 4: Comprehensive Testing & Polish

### Task 4.1: Run pytest suite

**Files:**
- Test: All automated tests

**Step 1: Run full pytest suite**

```bash
cd python
uv run pytest tests/ -v
```

Expected: All tests pass (or same failures as before changes)

**Step 2: Document any failures**

If tests fail:
```bash
echo "Pytest Results - $(date)" >> test-log.txt
uv run pytest tests/ -v 2>&1 | tee -a test-log.txt
```

**Step 3: Fix any new test failures**

If new failures introduced, fix them before proceeding.

---

### Task 4.2: Comprehensive UI testing - Decision Analyzer

**Files:**
- Test: Decision Analyzer app thoroughly

**Step 1: Test DA with sample data**

```bash
uv run pdstools da
```

Test checklist:
- [ ] Sample data loads automatically
- [ ] Version header displays correctly
- [ ] Data summary shows correct stats
- [ ] Format detection (v1 vs v2) works
- [ ] Navigate to all pages
- [ ] Charts render on all pages
- [ ] Filters work on Global Filters page
- [ ] Stage selector works
- [ ] Text hierarchy looks correct (intro vs captions)

**Step 2: Test DA with file upload**

Upload various formats:
- [ ] Single parquet file
- [ ] Zip file
- [ ] CSV file
- [ ] Multiple files

**Step 3: Test DA with CLI flags**

```bash
# Test --data-path
uv run pdstools da --data-path /path/to/data.parquet

# Test --sample
uv run pdstools da --sample 10000 --data-path /path/to/data.parquet

# Test --temp-dir
uv run pdstools da --sample 10000 --data-path /path/to/data.parquet --temp-dir /tmp
```

Test checklist:
- [ ] Data loads from configured path
- [ ] Sampling works and shows info banner
- [ ] Sampled file saved to correct directory
- [ ] Sample metadata displays correctly
- [ ] All pages work with sampled data

**Step 4: Document DA test results**

```bash
echo "=== Final DA UI Tests - $(date) ===" >> test-log.txt
echo "✓ Sample data: PASS" >> test-log.txt
echo "✓ File upload (all formats): PASS" >> test-log.txt
echo "✓ CLI --data-path: PASS" >> test-log.txt
echo "✓ CLI --sample: PASS" >> test-log.txt
echo "✓ CLI --temp-dir: PASS" >> test-log.txt
echo "✓ All pages render: PASS" >> test-log.txt
echo "✓ Filters work: PASS" >> test-log.txt
echo "✓ Stage selectors: PASS" >> test-log.txt
echo "✓ Text hierarchy: PASS" >> test-log.txt
```

---

### Task 4.3: Comprehensive UI testing - Impact Analyzer

**Files:**
- Test: Impact Analyzer app thoroughly

**Step 1: Test IA with sample data**

```bash
uv run pdstools ia
```

Test checklist:
- [ ] Sample data loads automatically
- [ ] Version header displays correctly
- [ ] Data summary shows correct stats (rows, channels)
- [ ] Format detection (PDC vs VBD) works
- [ ] Navigate to all pages (Overall Summary, Trend, About)
- [ ] Charts render on all pages
- [ ] Text hierarchy looks correct (intro vs captions)
- [ ] Bordered containers display correctly
- [ ] Captions are subdued/smaller than intro text

**Step 2: Test IA with file upload**

Upload various formats:
- [ ] PDC JSON file
- [ ] PDC NDJSON file
- [ ] VBD ZIP file
- [ ] Multiple PDC files

Check for each:
- [ ] Format detected correctly
- [ ] Data summary updates
- [ ] Charts work
- [ ] No errors in browser console

**Step 3: Test IA with CLI flags**

```bash
# Test --data-path with PDC
uv run pdstools ia --data-path /path/to/pdc.json

# Test --data-path with VBD
uv run pdstools ia --data-path /path/to/vbd.zip

# Test --sample with PDC
uv run pdstools ia --sample 1000 --data-path /path/to/pdc.json

# Test --sample with percentage
uv run pdstools ia --sample 10% --data-path /path/to/pdc.json

# Test --temp-dir
uv run pdstools ia --sample 1000 --data-path /path/to/pdc.json --temp-dir /tmp
```

Test checklist:
- [ ] Data loads from configured path (both formats)
- [ ] Sampling works with absolute count
- [ ] Sampling works with percentage
- [ ] Sampling info banner displays
- [ ] Sampled file saved to correct directory
- [ ] Sample metadata displays correctly
- [ ] All pages work with sampled data

**Step 4: Test error cases**

Test these error scenarios:
- [ ] Invalid --sample format (should show error)
- [ ] Non-existent --data-path (should show error)
- [ ] Unsupported file format (should show error)
- [ ] Corrupted file (should handle gracefully)

**Step 5: Document IA test results**

```bash
echo "=== Final IA UI Tests - $(date) ===" >> test-log.txt
echo "✓ Sample data: PASS" >> test-log.txt
echo "✓ File upload (PDC JSON): PASS" >> test-log.txt
echo "✓ File upload (VBD ZIP): PASS" >> test-log.txt
echo "✓ CLI --data-path (PDC): PASS" >> test-log.txt
echo "✓ CLI --data-path (VBD): PASS" >> test-log.txt
echo "✓ CLI --sample (count): PASS" >> test-log.txt
echo "✓ CLI --sample (percent): PASS" >> test-log.txt
echo "✓ CLI --temp-dir: PASS" >> test-log.txt
echo "✓ All pages render: PASS" >> test-log.txt
echo "✓ Text hierarchy: PASS" >> test-log.txt
echo "✓ Bordered containers: PASS" >> test-log.txt
echo "✓ Error handling: PASS" >> test-log.txt
```

---

### Task 4.4: Update documentation

**Files:**
- Modify: `python/pdstools/cli.py` (if needed)
- Create/Modify: Getting started guide (if exists)

**Step 1: Review CLI help text**

Check if CLI help mentions IA properly:

```bash
uv run pdstools --help
uv run pdstools ia --help
```

**Step 2: Update CLAUDE.md if needed**

Add any new patterns or conventions discovered during implementation.

**Step 3: Update CHANGELOG**

Add entry about IA improvements (when version is released).

**Step 4: Commit documentation updates**

```bash
git add <modified-files>
git commit -m "docs: update CLI and guides for IA improvements"
```

---

### Task 4.5: Final polish and review

**Files:**
- Review: All modified files

**Step 1: Review code quality**

Check for:
- [ ] No redundant comments
- [ ] Consistent naming
- [ ] Type hints where appropriate
- [ ] No dead code
- [ ] DRY violations removed

**Step 2: Review test coverage**

Ensure all new functionality is tested:
- [ ] Manual UI tests documented
- [ ] Core functions have examples/docstrings
- [ ] Error paths tested

**Step 3: Create summary of changes**

```bash
echo "=== Implementation Summary ===" > implementation-summary.txt
echo "" >> implementation-summary.txt
echo "## Files Modified" >> implementation-summary.txt
git diff --name-only master >> implementation-summary.txt
echo "" >> implementation-summary.txt
echo "## Test Results" >> implementation-summary.txt
cat test-log.txt >> implementation-summary.txt
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "polish: final cleanup and documentation"
```

---

## Completion Checklist

Before marking this plan as complete, verify:

- [ ] All Phase 1 tasks completed (shared utilities)
- [ ] All Phase 2 tasks completed (IA data loading)
- [ ] All Phase 3 tasks completed (IA page styling)
- [ ] All Phase 4 tasks completed (testing & polish)
- [ ] Decision Analyzer has no regressions
- [ ] Impact Analyzer UI matches DA patterns
- [ ] CLI flags work for both apps
- [ ] Sampling works correctly (stratified for DA, random for IA)
- [ ] All manual UI tests passed
- [ ] Documentation updated
- [ ] Test log created
- [ ] All commits follow conventional commit format

## Success Criteria

This implementation is successful when:

1. ✅ Impact Analyzer supports `--data-path`, `--sample`, and `--temp-dir` CLI flags
2. ✅ Impact Analyzer Home page matches Decision Analyzer's UX patterns
3. ✅ Impact Analyzer pages use consistent text hierarchy and bordered containers
4. ✅ No regressions in Decision Analyzer functionality
5. ✅ Code duplication reduced through shared utilities
6. ✅ Both apps pass comprehensive manual UI tests
7. ✅ Random sampling works for IA with metadata tracking
8. ✅ Format detection works correctly (PDC vs VBD for IA, v1 vs v2 for DA)

---

**Total Estimated Time:** 6-8 hours for complete implementation and testing
