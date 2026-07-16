from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from packaging.version import Version, InvalidVersion
import polars as pl
import polars.selectors as cs
import streamlit as st

from .. import __version__ as pdstools_version
from ..adm.ADMDatamart import ADMDatamart
from ..prediction.Prediction import Prediction
from ..utils import datasets
from . import cdh_utils

if TYPE_CHECKING:
    from ..utils.types import ANY_FRAME

# ---------------------------------------------------------------------------
# Shared Streamlit helpers — used by all pdstools apps
# ---------------------------------------------------------------------------

_MENU_ITEMS = {
    "Report a bug": "https://github.com/pegasystems/pega-datascientist-tools/issues",
    "Get help": "https://pegasystems.github.io/pega-datascientist-tools/latest/",
}

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "app" / "assets"


def _is_newer_version_available(installed: str, latest: str) -> bool:
    """Return True only when *latest* is strictly newer than *installed*.

    Uses PEP 440 parsing so that pre-release / dev versions of *installed*
    (e.g. ``4.6.0rc1``) are correctly recognised as newer than an older
    stable release on PyPI (e.g. ``4.5.2``).
    """
    try:
        return Version(latest) > Version(installed)
    except InvalidVersion:
        return False


def is_launcher_mode() -> bool:
    """True when the app is running inside the cross-app launcher.

    The CLI sets ``PDSTOOLS_LAUNCHER_MODE=1`` when invoking the launcher
    so per-app helpers (sidebar branding, navigation) can adapt their
    behaviour without each call site needing to know.
    """
    return os.environ.get("PDSTOOLS_LAUNCHER_MODE") == "1"


def set_active_app(app_key: str) -> None:
    """Mark the named app as currently active in the launcher sidebar.

    Each app's home page calls this on render. In launcher mode the
    cross-app entry script reads the flag to decide which app's
    sub-pages to register in ``st.navigation``, keeping the sidebar
    short until the user has actually entered an app. A no-op (other
    than the session_state write) in standalone tool launches.

    Triggers ``st.rerun()`` when the active app actually changes so
    the new sub-pages appear in the sidebar without a manual click.
    """
    if st.session_state.get("_active_app") == app_key:
        return
    st.session_state["_active_app"] = app_key
    if is_launcher_mode():
        st.rerun()


def _apply_sidebar_logo():
    """Re-apply the sidebar logo + brand from session state.

    Renders an optional subtitle (e.g. the active app name) under the
    main brand. In launcher mode the brand is always ``"pdstools"`` and
    the per-app title becomes the subtitle; standalone tools render a
    single-line brand only.
    """
    brand = st.session_state.get("_pdstools_brand")
    subtitle = st.session_state.get("_pdstools_brand_subtitle")
    if not brand and not subtitle:
        return
    logo_path = _ASSETS_DIR / "pega-logo.svg"
    if logo_path.exists():
        st.logo(str(logo_path), size="large")

    css_parts = ["<style>"]
    if brand:
        css_parts.append(
            "[data-testid='stSidebarNav']::before {"
            "  content: '" + brand.replace("'", "\\'") + "';"
            "  display: block;"
            "  font-size: 1.1rem;"
            "  font-weight: 500;"
            "  color: #5a5c63;"
            "  padding: 0 1rem 0.25rem;"
            "}"
        )
    if subtitle:
        css_parts.append(
            "[data-testid='stSidebarNav']::after {"
            "  content: '" + subtitle.replace("'", "\\'") + "';"
            "  display: block;"
            "  font-size: 0.85rem;"
            "  font-weight: 400;"
            "  color: #8a8d96;"
            "  padding: 0 1rem 0.75rem;"
            "}"
        )
    # Streamlit auto-collapses sidebar nav past ~10 items into a
    # "View N more" button. With three apps hosted side-by-side the
    # launcher easily exceeds that, hiding entire tools by default.
    # Force-expand by setting max-height on the nav list and hiding
    # the toggle. Targets stable testid attributes only — no JS.
    css_parts.append(
        "[data-testid='stSidebarNav'] ul {"
        "  max-height: none !important;"
        "}"
        "[data-testid='stSidebarNavViewButton'] {"
        "  display: none !important;"
        "}"
    )
    css_parts.append("</style>")
    st.html("".join(css_parts))


def standard_page_config(page_title: str, layout: Literal["centered", "wide"] = "wide", **kwargs):
    """Apply a consistent ``st.set_page_config`` across all pdstools apps.

    Idempotent: when called more than once in the same script run (e.g. an
    ``st.navigation()`` entry script calls it before routing, and the routed
    page calls it again), the second call is a no-op rather than the
    ``StreamlitAPIException`` Streamlit normally raises. This lets the same
    page module work both standalone (direct ``AppTest.from_file`` /
    ``streamlit run pages/X.py``) and inside a navigation router.

    Parameters
    ----------
    page_title : str
        The browser-tab title for the page.
    layout : str, default "wide"
        Streamlit layout mode.
    **kwargs
        Extra keyword arguments forwarded to ``st.set_page_config``.

    """
    from streamlit.errors import StreamlitAPIException

    kwargs.setdefault("menu_items", _MENU_ITEMS)
    logo_path = _ASSETS_DIR / "pega-logo.svg"
    if logo_path.exists():
        kwargs.setdefault("page_icon", str(logo_path))
    try:
        st.set_page_config(layout=layout, page_title=page_title, **kwargs)
    except StreamlitAPIException:
        pass
    _apply_sidebar_logo()


def show_sidebar_branding(title: str):
    """Display the Pega logo and an app title at the top of the sidebar.

    Uses ``st.logo`` for the logo and CSS injection for the title, so both
    render above the page navigation. Call once from the Home page; sub-pages
    re-apply automatically via ``standard_page_config`` or ``ensure_data``.

    In launcher mode (``PDSTOOLS_LAUNCHER_MODE=1``) the brand is forced to
    ``"pdstools"`` and the supplied *title* is rendered as a subtitle, so
    every page in the multi-app launcher shows a consistent top-level brand
    even when the active sub-app re-applies its own title.

    Parameters
    ----------
    title : str
        Application title shown below the logo in the sidebar.

    """
    if is_launcher_mode():
        st.session_state["_pdstools_brand"] = "pdstools"
        # Don't show "pdstools" twice when the launcher's own picker
        # page re-applies branding with the same title.
        st.session_state["_pdstools_brand_subtitle"] = title if title and title.lower() != "pdstools" else None
    else:
        st.session_state["_pdstools_brand"] = title
        st.session_state["_pdstools_brand_subtitle"] = None
    _apply_sidebar_logo()


def show_version_header(check_latest: bool = True):
    """Display the pdstools version, an upgrade hint, and optionally a staleness warning.

    Parameters
    ----------
    check_latest : bool, default True
        If *True*, queries PyPI for the latest version and shows an upgrade
        warning when the installed version is outdated.

    """
    st.caption(f"pdstools {pdstools_version}")

    if check_latest:
        latest = st_get_latest_pdstools_version()
        if latest and _is_newer_version_available(pdstools_version, latest):
            st.warning(
                f"A newer version of pdstools is available (**{latest}**, "
                f"you have {pdstools_version}). "
                "Run `uv pip install --upgrade pdstools` to update.",
            )


def ensure_session_data(key: str, message: str | None = None):
    """Guard that stops page execution when *key* is missing from session state.

    Parameters
    ----------
    key : str
        The ``st.session_state`` key to check.
    message : str or None
        Custom warning text. Falls back to a generic "Please load data on the Home page."

    """
    if key not in st.session_state:
        st.warning(message or "Please load data on the Home page.")
        st.stop()


def get_data_path() -> str | None:
    """Return the data path set via ``--data-path`` CLI flag.

    Returns ``None`` when no path was configured.
    """
    return os.environ.get("PDSTOOLS_DATA_PATH")


def get_sample_limit() -> str | None:
    """Return the raw sample limit string set via ``--sample`` CLI flag.

    Returns ``None`` when no sampling was requested.
    """
    return os.environ.get("PDSTOOLS_SAMPLE_LIMIT")


def get_filter_specs() -> list[str] | None:
    """Return the filter specs set via ``--filter`` CLI flags.

    Returns ``None`` when no filters were configured.
    """
    raw = os.environ.get("PDSTOOLS_FILTER")
    if raw is None:
        return None
    import json

    return json.loads(raw)


def get_temp_dir() -> str | None:
    """Return the temp directory set via ``--temp-dir`` CLI flag.

    Returns ``None`` when no temp directory was configured.
    """
    return os.environ.get("PDSTOOLS_TEMP_DIR")


def get_full_embed() -> bool | None:
    """Return the full-embed setting set via ``--full-embed`` / ``--no-full-embed`` CLI flag.

    Returns ``None`` when the flag was not provided (caller should apply its
    own default).
    """
    raw = os.environ.get("PDSTOOLS_FULL_EMBED")
    if raw is None:
        return None
    return raw.lower() in ("1", "true", "yes")


def parse_sample_spec(value: str) -> dict[str, int | float]:
    """Parse a ``--sample`` flag value into keyword arguments.

    Supports absolute counts (``"100000"``), percentages (``"10%"``),
    and human-readable notation (``"100k"``, ``"1M"``).

    Returns
    -------
    dict
        Either ``{"n": <int>}`` or ``{"fraction": <float>}``.
    """
    value = value.strip()
    if value.endswith("%"):
        pct = float(value[:-1])
        if not 0 < pct <= 100:
            raise ValueError(f"Percentage must be in (0, 100], got {pct}")
        return {"fraction": pct / 100.0}

    multiplier = None
    if value.lower().endswith("k"):
        multiplier = 1000
    elif value.lower().endswith("m"):
        multiplier = 1000000

    count = int(float(value[:-1]) * multiplier) if multiplier else int(value)

    if count <= 0:
        raise ValueError(f"Sample count must be positive, got {count}")
    return {"n": count}


def get_current_index(options, key, default=0):
    """Get index from session state if key exists and value is in options, else return default."""
    return (
        options.index(st.session_state[key])
        if key in st.session_state and st.session_state[key] in options
        else default
    )


@st.cache_resource
def cached_sample():
    """Cached sample."""
    return datasets.cdh_sample()


@st.cache_resource
def cached_datamart(**kwargs):
    """Load ADMDatamart with caching.

    Parameters
    ----------
    **kwargs
        Arguments passed to ADMDatamart.from_ds_export

    """
    with st.spinner("Loading datamart..."):
        try:
            datamart = ADMDatamart.from_ds_export(**kwargs)
            if datamart is not None:
                return datamart
            st.warning("Unable to load datamart.")
            return None
        except Exception as e:
            st.error(f"An error occurred while importing the datamart: {e!s}")
            return None


@st.cache_resource
def cached_sample_prediction():
    """Cached sample prediction."""
    return Prediction.from_mock_data(days=60)


@st.cache_resource
def cached_prediction_table(**kwargs):
    """Load Prediction with caching.

    Parameters
    ----------
    **kwargs
        Arguments passed to Prediction.from_ds_export

    """
    with st.spinner("Loading prediction table..."):
        try:
            prediction = Prediction.from_ds_export(**kwargs)
            if prediction is not None:
                return prediction
            st.warning("Unable to load prediction table.")
            return None
        except Exception as e:
            st.error(
                f"An error occurred while importing the prediction table: {e!s}",
            )
            return None


def model_selection_df(df: pl.LazyFrame, context_keys: list):
    """Model selection df."""
    return (
        df.select(["ModelID", "Configuration", *context_keys])
        .unique()
        .sort("Name")
        .select(pl.lit(False).alias("Generate Report"), pl.all())
        .collect()
    )


def filter_dataframe(
    df: pl.LazyFrame,
    schema: dict | None = None,
    queries: list[pl.Expr] | None = None,
) -> list[pl.Expr]:
    """Adds a UI on top of a dataframe to let viewers filter columns

    Parameters
    ----------
    df : pl.DataFrame
        Original dataframe

    Returns
    -------
    list[pl.Expr]
        Filter expressions collected from the UI.

    """
    if queries is None:
        queries = []
    to_filter_columns = st.multiselect(
        "Filter dataframe on",
        df.collect_schema().names(),
        key="multiselect",
    )
    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        left.write("## ↳")
        col_dtype = df.collect_schema()[column]
        # Treat columns with < 20 unique values as categorical
        if (col_dtype == pl.Categorical) or (col_dtype == pl.Utf8):
            if f"categories_{column}" not in st.session_state.keys():
                st.session_state[f"categories_{column}"] = (
                    df.select(pl.col(column).unique()).collect().to_series().to_list()
                )
            if f"selected_{column}" not in st.session_state.keys():
                st.session_state[f"selected_{column}"] = st.session_state[f"categories_{column}"]
            if len(st.session_state[f"categories_{column}"]) < 200:
                options = st.session_state[f"categories_{column}"]
                selected = right.multiselect(
                    f"Values for {column}",
                    options,
                    key=f"selected_{column}",
                )
                if selected != st.session_state[f"categories_{column}"]:
                    queries.append(
                        pl.col(column).cast(pl.Utf8).is_in(st.session_state[f"selected_{column}"]),
                    )
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    queries.append(pl.col(column).str.contains(user_text_input))

        elif df.select(cs.numeric()).collect_schema().get(column) is not None:
            min_col, max_col = right.columns((1, 1))
            _min = float(df.select(pl.min(column)).collect().item())
            _max = float(df.select(pl.max(column)).collect().item())
            if _max - _min <= 200:
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                )
            else:
                user_min = min_col.number_input(
                    label=f"Min value for {column} (Min:{_min})",
                    min_value=_min,
                    max_value=_max,
                    value=_min,
                )
                user_max = max_col.number_input(
                    label=f"Max value for {column} (Max:{_max})",
                    min_value=_min,
                    max_value=_max,
                    value=_max,
                )
                user_num_input = (user_min, user_max)
            if user_num_input[0] != _min or user_num_input[1] != _max:
                queries.append(pl.col(column).is_between(*user_num_input))
        elif df.select(cs.temporal()).collect_schema().get(column) is not None:
            user_date_input = right.date_input(
                f"Values for {column}",
                value=(
                    df.select(pl.min(column)).collect().item(),
                    df.select(pl.max(column)).collect().item(),
                ),
            )
            if len(user_date_input) == 2:
                queries.append(pl.col(column).is_between(*user_date_input))

    return queries


def model_and_row_counts(df: ANY_FRAME):
    """Returns unique model id count and row count from a dataframe

    Parameters
    ----------
    df: Union[pl.DataFrame, pl.LazyFrame]
        The input dataframe

    Returns
    -------
    Tuple[int, int]
        unique model count
        row count

    """
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    counts = df.select(
        unique_model_id_count=pl.approx_n_unique("ModelID"),
        row_count=pl.count("ModelID"),
    ).collect()

    unique_model_id_count = counts.get_column("unique_model_id_count")[0]
    row_count = counts.get_column("row_count")[0]

    return unique_model_id_count, row_count


@st.cache_data
def convert_df(df):
    """Convert df."""
    return df.write_csv().encode("utf-8")


@st.cache_data
def st_get_latest_pdstools_version():
    """St get latest pdstools version."""
    return cdh_utils.get_latest_pdstools_version()


def show_about_page():
    """Render a standardised About page with version and dependency information.

    Mirrors the Credits section of the Quarto ADM Health Check report.
    Call this from a Streamlit page to display pdstools version info,
    platform details, and an expandable dependency listing.
    """
    from ..utils.show_versions import show_versions

    st.markdown("# About")
    st.markdown(
        "This application is part of "
        "[Pega Data Scientist Tools](https://github.com/pegasystems/pega-datascientist-tools), "
        "an open-source toolkit for Pega decisioning analytics.",
    )

    st.markdown("### Version information")
    summary = show_versions(print_output=False, include_dependencies=False)
    st.code(summary, language=None)

    latest = st_get_latest_pdstools_version()
    if latest and _is_newer_version_available(pdstools_version, latest):
        st.warning(
            f"A newer version is available (**{latest}**). Run `uv pip install --upgrade pdstools` to update.",
        )

    with st.expander("Detailed dependency versions"):
        details = show_versions(print_output=False, include_dependencies=True)
        st.code(details, language=None)

    st.markdown(
        "For more information see the "
        "[documentation](https://pegasystems.github.io/pega-datascientist-tools/) "
        "or [report an issue](https://github.com/pegasystems/pega-datascientist-tools/issues).",
    )
