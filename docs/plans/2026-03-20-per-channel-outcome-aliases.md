# Per-Channel Outcome Aliases for Impact Analyzer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow `ImpactAnalyzer.from_vbd()` to accept per-channel outcome label mappings (e.g., "Sent" = Impressions for Email) and expose a configuration UI in the Streamlit app.

**Architecture:** Add an `outcome_labels` keyword parameter to `from_vbd()` that accepts either a global dict (current format) or a per-channel dict. A helper method builds the Polars filter expression. The Streamlit Home page discovers available outcomes from already-loaded data (the `Outcome` list column kept for debugging) and shows a UI to configure per-channel mappings that triggers a re-load.

**Tech Stack:** Python 3.10+, Polars, Streamlit, pytest

---

## Background

### Current outcome label structure (class attribute)

```python
# ImpactAnalyzer.outcome_labels (global, applies to all channels)
{
    "Impressions": ["Impression"],
    "Accepts": ["Accept", "Accepted", "Click", "Clicked"],
}
```

### New per-channel structure (optional override)

```python
{
    "Email/Outbound": {
        "Impressions": ["Sent"],
        "Accepts": ["Click", "Clicked"],
    },
    # Other channels not listed → fall back to class defaults
}
```

### Detection: if any value in the dict is itself a dict → per-channel mode.

### Key granularity: `Channel/Direction` not just `Channel`

The CDH Portal configures outcome aliases per **channel only** (e.g. `"Email"`). Our tool uses `Channel/Direction` combined strings (e.g. `"Email/Outbound"`) because that is how `from_vbd()` constructs the `Channel` column from VBD data. This is intentionally more granular — a user could configure different aliases for `Email/Inbound` vs `Email/Outbound`. The Streamlit discovery UI shows the actual `Channel/Direction` values present in the data so users know exactly what keys to use.

### Key insight: discovering outcomes

After `from_vbd()`, `ia_data` retains an `Outcome` list column (kept for debugging). We can use `.explode("Outcome").select("Channel", "Outcome").unique()` to find all outcome values present in the data — no need to re-read the raw file for the discovery UI.

---

## Task 1: Core library — `_build_outcome_filter()` helper and `from_vbd()` parameter

**Files:**
- Modify: `python/pdstools/impactanalyzer/ImpactAnalyzer.py`
- Test: `python/tests/test_ImpactAnalyzer.py`

### Step 1: Write the failing tests

Add to `python/tests/test_ImpactAnalyzer.py`:

```python
def _make_vbd_parquet(outcomes_by_channel: dict[str, list]) -> str:
    """Helper: write minimal VBD parquet for testing outcome label scenarios.

    outcomes_by_channel: {"Web/Inbound": ["Impression", "Accepted"], "Email/Outbound": ["Sent", "Click"]}
    Each channel gets 100 aggregated rows per outcome.
    """
    import tempfile

    rows = []
    for channel_dir, outcomes in outcomes_by_channel.items():
        channel, direction = channel_dir.split("/")
        for outcome in outcomes:
            rows.append({
                "pyOutcomeTime": "2024-01-01 10:00:00",
                "pyChannel": channel,
                "pyDirection": direction,
                "pxMktValue": None,
                "pyReason": "Test",
                "pxMktType": None,
                "pyApplication": "App1",
                "pxApplicationVersion": "1.0",
                "pyIssue": "Sales",
                "pyGroup": "Cards",
                "pyName": "GoldCard",
                "pyTreatment": "Default",
                "pyOutcome": outcome,
                "pxAggregateCount": 100,
                "pyValue": 0.0,
            })

    df = pl.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.write_parquet(f.name)
        return f.name


def test_from_vbd_default_outcome_labels():
    """Default outcome labels count standard outcomes correctly."""
    path = _make_vbd_parquet({
        "Web/Inbound": ["Impression", "Accepted", "UnknownOutcome"],
    })
    ia = ImpactAnalyzer.from_vbd(path)
    collected = ia.ia_data.collect()
    # Standard "Impression" → 100 impressions; "Accepted" → 100 accepts; "UnknownOutcome" ignored
    assert collected["Impressions"].sum() == 100
    assert collected["Accepts"].sum() == 100


def test_from_vbd_global_outcome_labels_override():
    """Custom global outcome_labels replace the class defaults entirely."""
    path = _make_vbd_parquet({
        "Web/Inbound": ["Sent", "Click"],
    })
    # Without override: Sent and Click are not in defaults → zero counts
    ia_default = ImpactAnalyzer.from_vbd(path)
    assert ia_default.ia_data.collect()["Impressions"].sum() == 0

    # With global override
    ia_custom = ImpactAnalyzer.from_vbd(
        path,
        outcome_labels={"Impressions": ["Sent"], "Accepts": ["Click"]},
    )
    collected = ia_custom.ia_data.collect()
    assert collected["Impressions"].sum() == 100
    assert collected["Accepts"].sum() == 100


def test_from_vbd_per_channel_outcome_labels():
    """Per-channel outcome_labels applied per channel; others fall back to class defaults."""
    path = _make_vbd_parquet({
        "Email/Outbound": ["Sent", "Click"],     # custom channel
        "Web/Inbound": ["Impression", "Accepted"],  # uses class defaults
    })
    per_channel = {
        "Email/Outbound": {
            "Impressions": ["Sent"],
            "Accepts": ["Click"],
        }
        # Web/Inbound not configured → class defaults apply
    }
    ia = ImpactAnalyzer.from_vbd(path, outcome_labels=per_channel)
    collected = ia.ia_data.collect()

    email_rows = collected.filter(pl.col("Channel") == "Email/Outbound")
    web_rows = collected.filter(pl.col("Channel") == "Web/Inbound")

    assert email_rows["Impressions"].sum() == 100   # "Sent" mapped
    assert email_rows["Accepts"].sum() == 100       # "Click" mapped
    assert web_rows["Impressions"].sum() == 100     # "Impression" via defaults
    assert web_rows["Accepts"].sum() == 100         # "Accepted" via defaults


def test_from_vbd_per_channel_fallback_for_unconfigured_channel():
    """A channel not in per-channel config uses class-level defaults."""
    path = _make_vbd_parquet({
        "SMS/Outbound": ["Impression", "Accepted"],
    })
    # Configure only Email, not SMS
    per_channel = {
        "Email/Outbound": {"Impressions": ["Sent"], "Accepts": ["Click"]},
    }
    ia = ImpactAnalyzer.from_vbd(path, outcome_labels=per_channel)
    sms_rows = ia.ia_data.collect().filter(pl.col("Channel") == "SMS/Outbound")
    # SMS falls back to class defaults: "Impression" and "Accepted"
    assert sms_rows["Impressions"].sum() == 100
    assert sms_rows["Accepts"].sum() == 100
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest python/tests/test_ImpactAnalyzer.py::test_from_vbd_default_outcome_labels python/tests/test_ImpactAnalyzer.py::test_from_vbd_global_outcome_labels_override python/tests/test_ImpactAnalyzer.py::test_from_vbd_per_channel_outcome_labels python/tests/test_ImpactAnalyzer.py::test_from_vbd_per_channel_fallback_for_unconfigured_channel -v
```

Expected: FAIL — `from_vbd()` does not accept `outcome_labels` parameter.

### Step 3: Add `_build_outcome_filter()` classmethod

In `python/pdstools/impactanalyzer/ImpactAnalyzer.py`, add before `from_vbd()`:

```python
@classmethod
def _build_outcome_filter(
    cls,
    metric: str,
    outcome_labels: dict | None,
) -> pl.Expr:
    """Return a boolean Polars expression for the given metric (Impressions or Accepts).

    Parameters
    ----------
    metric : str
        "Impressions" or "Accepts"
    outcome_labels : dict or None
        None → use class defaults.
        ``{"Impressions": [...], "Accepts": [...]}`` → global override.
        ``{"Channel/Dir": {"Impressions": [...], "Accepts": [...]}, ...}`` → per-channel.
        Channels absent from per-channel config fall back to the class defaults.
    """
    # Determine if per-channel mode
    is_per_channel = outcome_labels is not None and any(
        isinstance(v, dict) for v in outcome_labels.values()
    )

    if not is_per_channel:
        # Global mode: use provided labels or class defaults
        labels = (outcome_labels or cls.outcome_labels).get(metric, [])
        return pl.col("Outcome").is_in(labels)

    # Per-channel mode: start from class defaults as the fallback expression
    default_labels = cls.outcome_labels.get(metric, [])
    expr: pl.Expr = pl.col("Outcome").is_in(default_labels)

    for channel, channel_labels in outcome_labels.items():
        if metric in channel_labels:
            expr = (
                pl.when(pl.col("Channel") == channel)
                .then(pl.col("Outcome").is_in(channel_labels[metric]))
                .otherwise(expr)
            )

    return expr
```

### Step 4: Update `from_vbd()` signature and body

Replace the `from_vbd()` overloads and implementation. The `outcome_labels` parameter is keyword-only.

Update the first overload:
```python
@classmethod
@overload
def from_vbd(
    cls,
    vbd_source: os.PathLike | str,
    *,
    outcome_labels: dict | None = None,
    return_df: Literal[True],
) -> pl.LazyFrame | None: ...
```

Update the second overload:
```python
@classmethod
@overload
def from_vbd(
    cls,
    vbd_source: os.PathLike | str,
    *,
    outcome_labels: dict | None = None,
) -> Optional["ImpactAnalyzer"]: ...
```

Update the implementation signature:
```python
@classmethod
def from_vbd(
    cls,
    vbd_source: os.PathLike | str,
    *,
    outcome_labels: dict | None = None,
    return_df: bool = False,
) -> Union["ImpactAnalyzer", pl.LazyFrame, None]:
```

Update the docstring to document the new parameter:
```
outcome_labels : dict or None, optional
    Outcome value mappings for Impressions and Accepts. Accepts two formats:

    **Global override** — replaces class defaults for all channels::

        {"Impressions": ["Sent"], "Accepts": ["Click", "Clicked"]}

    **Per-channel** — overrides per channel; unconfigured channels fall back
    to the class-level defaults::

        {
            "Email/Outbound": {
                "Impressions": ["Sent"],
                "Accepts": ["Click", "Clicked"],
            }
        }

    Default is None (use :attr:`outcome_labels` class attribute).
```

In the `.agg()` call, replace the two hardcoded `cls.outcome_labels[...]` usages:

```python
# BEFORE
pl.col("AggregateCount")
    .filter(pl.col("Outcome").is_in(cls.outcome_labels["Impressions"]))
    .sum()
    .alias("Impressions"),
pl.col("AggregateCount")
    .filter(pl.col("Outcome").is_in(cls.outcome_labels["Accepts"]))
    .sum()
    .alias("Accepts"),
(
    pl.col("Value").filter(pl.col("Outcome").is_in(cls.outcome_labels["Accepts"])).sum()
    / (
        pl.col("AggregateCount")
        .filter(pl.col("Outcome").is_in(cls.outcome_labels["Impressions"]))
        .sum()
    )
).alias("ValuePerImpression"),

# AFTER — use helper; note the Channel column is available at this point
# because Channel is computed before group_by and is a group key
pl.col("AggregateCount")
    .filter(cls._build_outcome_filter("Impressions", outcome_labels))
    .sum()
    .alias("Impressions"),
pl.col("AggregateCount")
    .filter(cls._build_outcome_filter("Accepts", outcome_labels))
    .sum()
    .alias("Accepts"),
(
    pl.col("Value")
    .filter(cls._build_outcome_filter("Accepts", outcome_labels))
    .sum()
    / pl.col("AggregateCount")
    .filter(cls._build_outcome_filter("Impressions", outcome_labels))
    .sum()
).alias("ValuePerImpression"),
```

> **Important Polars note:** `Channel` is a group key in this `group_by`, so expressions that reference `pl.col("Channel")` inside `.agg()` work correctly — each group has a single unique Channel value.

### Step 5: Run tests to verify they pass

```bash
uv run pytest python/tests/test_ImpactAnalyzer.py -v
```

Expected: ALL PASS, including all pre-existing tests.

### Step 6: Commit

```bash
git add python/pdstools/impactanalyzer/ImpactAnalyzer.py python/tests/test_ImpactAnalyzer.py
git commit -m "feat(impact-analyzer): add per-channel outcome_labels to from_vbd()"
```

---

## Task 2: Streamlit — outcome discovery and configuration UI

**Files:**
- Modify: `python/pdstools/app/impact_analyzer/ia_streamlit_utils.py`
- Modify: `python/pdstools/app/impact_analyzer/Home.py`

This task adds no new library logic — only UI wiring.

### Step 1: Update `load_vbd_from_path` to accept outcome_labels

The function is `@st.cache_resource` — Streamlit hashes all arguments. Pass `outcome_labels` as a JSON string to make it hashable.

In `ia_streamlit_utils.py`, replace:

```python
@st.cache_resource
def load_vbd_from_path(path: str) -> ImpactAnalyzer | None:
    return ImpactAnalyzer.from_vbd(path)
```

With:

```python
@st.cache_resource
def load_vbd_from_path(path: str, outcome_labels_json: str | None = None) -> ImpactAnalyzer | None:
    """Load VBD data. outcome_labels_json is a JSON-serialized outcome_labels dict (hashable for cache)."""
    import json

    outcome_labels = json.loads(outcome_labels_json) if outcome_labels_json else None
    return ImpactAnalyzer.from_vbd(path, outcome_labels=outcome_labels)
```

Also update `load_vbd_from_upload` to forward the parameter:

```python
def load_vbd_from_upload(uploaded_file, outcome_labels_json: str | None = None) -> ImpactAnalyzer | None:
    path = _write_uploaded_file(uploaded_file)
    return load_vbd_from_path(path, outcome_labels_json=outcome_labels_json)
```

### Step 2: Add `discover_vbd_outcomes()` utility

Add to `ia_streamlit_utils.py`:

```python
def discover_vbd_outcomes(ia: "ImpactAnalyzer") -> dict[str, list[str]]:
    """Return {channel: [unique outcome values]} from already-loaded VBD ia_data.

    Relies on the Outcome list column kept for debugging in from_vbd().
    Returns empty dict if Outcome column is absent (e.g., PDC data).
    """
    import polars as pl

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
```

### Step 3: Add `show_outcome_alias_config()` UI component

Add to `ia_streamlit_utils.py`:

```python
def show_outcome_alias_config(ia: "ImpactAnalyzer") -> dict | None:
    """Show per-channel outcome alias configuration UI for VBD data.

    Returns the outcome_labels dict to pass to from_vbd(), or None if
    the user has not made any changes from the defaults.

    Only renders for VBD data (when Outcome column is present).
    """
    import json

    outcomes_by_channel = discover_vbd_outcomes(ia)
    if not outcomes_by_channel:
        return None  # PDC data or no Outcome column — skip

    from pdstools import ImpactAnalyzer as IA

    default_impressions = IA.outcome_labels["Impressions"]
    default_accepts = IA.outcome_labels["Accepts"]

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

            if sorted(impressions) != sorted(default_imp) or sorted(accepts) != sorted(default_acc):
                any_non_default = True

            per_channel_config[channel] = {"Impressions": impressions, "Accepts": accepts}

        if st.button("Apply outcome aliases"):
            config = per_channel_config if any_non_default else None
            st.session_state["ia_outcome_labels"] = config
            st.session_state["ia_outcome_labels_json"] = json.dumps(config) if config else None
            st.rerun()

    return st.session_state.get("ia_outcome_labels")
```

### Step 4: Wire outcome config into the VBD reload flow in `Home.py`

In `Home.py`, after a successful VBD load (inside the `if uploaded_files:` block and the CLI path block), call `show_outcome_alias_config()` and — if configuration changed — reload:

Add the following **after** the `if impact_analyzer is not None:` store block near the bottom of `Home.py`:

```python
# For VBD data: show outcome alias config and handle reload
_active_ia = impact_analyzer or st.session_state.get("impact_analyzer")
if _active_ia is not None:
    from pdstools.app.impact_analyzer.ia_streamlit_utils import show_outcome_alias_config
    show_outcome_alias_config(_active_ia)
```

Also, when loading from upload with VBD format detected, pass any existing `outcome_labels_json` from session state so the cached function produces the right result after "Apply":

In the `load_from_upload_auto` call path (inside `if uploaded_files:`), the reload happens via `st.rerun()` triggered by "Apply". On the next run, `uploaded_files` will be empty (the uploader resets), but `"impact_analyzer"` will be in session state. So the existing session-state persistence handles it — when "Apply" is clicked, session state gets `ia_outcome_labels_json`, the page reruns, and the existing impact_analyzer is displayed along with the config UI already showing the new defaults.

**For the re-load with new labels to take effect**, the "Apply" button must also clear the cached impact_analyzer from session state so the next uploader interaction picks up the new config. Update `show_outcome_alias_config()` button handler to additionally clear the old instance:

```python
if st.button("Apply outcome aliases"):
    config = per_channel_config if any_non_default else None
    st.session_state["ia_outcome_labels"] = config
    st.session_state["ia_outcome_labels_json"] = json.dumps(config) if config else None
    # Clear existing IA so next upload/path reload uses the new config
    st.session_state.pop("impact_analyzer", None)
    st.session_state.pop("ia_is_sample_data", None)
    st.rerun()
```

Then in `Home.py`, when loading from upload (single VBD file case), pass the stored `outcome_labels_json`:

```python
# Inside the single-file upload block, change:
impact_analyzer = _load_with_warning(
    lambda: load_from_upload_auto(filtered_files[0]),
    "uploaded",
    expected_input_cols=VBD_REQUIRED_COLS if ".zip" in suffixes else None,
)

# To:
_outcome_labels_json = st.session_state.get("ia_outcome_labels_json")
impact_analyzer = _load_with_warning(
    lambda: load_from_upload_auto(filtered_files[0], outcome_labels_json=_outcome_labels_json),
    "uploaded",
    expected_input_cols=VBD_REQUIRED_COLS if ".zip" in suffixes else None,
)
```

Also update `load_from_upload_auto` in `ia_streamlit_utils.py` to accept and forward `outcome_labels_json`:

```python
def load_from_upload_auto(uploaded_file, outcome_labels_json: str | None = None) -> ImpactAnalyzer | None:
    """Load Impact Analyzer data with automatic format detection."""
    format_type = _detect_file_format(uploaded_file)

    if format_type == "pdc":
        return load_pdc_from_uploads([uploaded_file])
    elif format_type == "vbd":
        return load_vbd_from_upload(uploaded_file, outcome_labels_json=outcome_labels_json)
    else:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".zip":
            return load_vbd_from_upload(uploaded_file, outcome_labels_json=outcome_labels_json)
        else:
            return load_pdc_from_uploads([uploaded_file])
```

### Step 5: Smoke test manually

```bash
pdstools impact_analyzer
```

1. Upload a VBD ZIP file → "Configure outcome aliases" expander should appear
2. Upload a PDC JSON file → expander should NOT appear
3. With VBD data: change an "Impressions" multiselect to something non-default, click "Apply" → page reruns, old session data is cleared; re-upload the VBD file → data loads with new config

### Step 6: Commit

```bash
git add python/pdstools/app/impact_analyzer/ia_streamlit_utils.py python/pdstools/app/impact_analyzer/Home.py
git commit -m "feat(impact-analyzer): add per-channel outcome alias config UI for VBD data"
```

---

## Task 3: Update the TODO backlog

**Files:**
- Modify: `docs/plans/impact-analyzer-TODO.md`

Add a new entry under **Core Library → Medium Priority** referencing this feature, then mark it complete once the PR merges. Also update the `**Total items:**` count.

Add to the TODO under `## Core Library`, `### Medium Priority`:

```markdown
- [x] **[P2] Per-channel outcome aliases** — `from_vbd()` now accepts `outcome_labels` for per-channel or global overrides. Streamlit UI exposes outcome discovery and mapping. Closes #598.
```

### Step 1: Edit the file, run tests, commit

```bash
uv run pytest python/tests/test_ImpactAnalyzer.py -v
git add docs/plans/impact-analyzer-TODO.md
git commit -m "docs: mark per-channel outcome aliases as done in TODO"
```

---

## Acceptance Criteria Checklist

From issue #598:

- [x] Per-channel configuration acceptance — `outcome_labels` parameter on `from_vbd()`
- [x] Channel-specific alias application during aggregation — `_build_outcome_filter()` helper
- [x] Streamlit UI exposure for VBD data — `show_outcome_alias_config()` in `ia_streamlit_utils.py`
- [x] Backward compatibility — `None` default falls through to class-level `outcome_labels`
- [x] Custom configuration test coverage — 4 new tests in `test_ImpactAnalyzer.py`
- [x] Persistent alias-to-datasource association — stored in `st.session_state["ia_outcome_labels"]`

---

## Notes for Implementer

- **Type hints**: Use `dict | None` not `Optional[dict]` (project uses Python 3.10+ style).
- **No `verbose` parameters**: Use `logger.debug()` if you need diagnostic output.
- **`from_ih()` is not implemented** — do not add `outcome_labels` to it (not needed yet).
- **PDC data**: The `outcome_labels` parameter only applies to `from_vbd()`. PDC data is pre-aggregated and the `Outcome` column is absent after loading.
- **Test file**: Tests live in `python/tests/test_ImpactAnalyzer.py` — add to the existing file, don't create a new one.
- **Run full suite before PR**: `uv run pytest python/tests/test_ImpactAnalyzer.py -v`
