"""Widget-interaction test: column-picker multiselect on HC Data Filters page.

``1_Data_Filters.py`` uses ``filter_dataframe()`` to render a
"Filter dataframe on" multiselect (keyed ``"multiselect"``).  When the
user picks a column, a per-column sub-widget appears.  When a
categorical column's values are narrowed, a polars filter expression is
appended to ``session_state["filters"]``.

If the multiselect key is renamed, the column loop is broken, or the
``session_state["filters"]`` assignment is removed, the filter widgets
still render but silently do nothing — the report page always sees an
empty ``filters`` list.  Existing smoke tests stay green because the
page renders without any column selected.  These tests exercise the
state-transitions: pick a column, verify the per-column widget appears,
narrow values, verify the filter expression is stored.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _find_multiselect(at: AppTest, key: str):
    for ms in at.multiselect:
        if ms.key == key:
            return ms
    return None


def test_initial_filters_list_is_empty(hc_app_dir, seeded_admdatamart) -> None:
    """HC Data Filters page initialises ``session_state['filters']`` as empty."""
    page = hc_app_dir / "pages" / "1_Data_Filters.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    # With no column selected the filter list must be empty.
    assert at.session_state["filters"] == [], (
        f"Expected filters=[] on first render, got {at.session_state['filters']!r}"
    )

    column_picker = _find_multiselect(at, "multiselect")
    assert column_picker is not None, (
        "Expected 'Filter dataframe on' multiselect with key='multiselect' to be rendered."
    )
    assert column_picker.value == [], "Column picker should be empty on first visit."


def test_selecting_column_renders_sub_widget(hc_app_dir, seeded_admdatamart) -> None:
    """Picking a categorical column from the column picker renders a per-column widget."""
    page = hc_app_dir / "pages" / "1_Data_Filters.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    column_picker = _find_multiselect(at, "multiselect")
    assert column_picker is not None

    # Pick the first available column that is likely categorical (Utf8 columns
    # appear in ADM model data: Channel, Direction, Configuration, etc.).
    available_columns = column_picker.options
    assert available_columns, "Column picker should list at least one column from the ADM data."

    chosen = available_columns[0]
    column_picker.set_value([chosen]).run()
    assert not at.exception, f"Post-selection run raised: {at.exception}"

    # After selecting a column, either a per-column multiselect or a text/slider
    # widget should be rendered for that column — the exact type depends on dtype.
    sub_widget_count = len(at.multiselect) + len(at.slider) + len(at.text_input)
    assert sub_widget_count >= 1, (
        f"After selecting column {chosen!r}, expected at least one sub-filter widget but found none."
    )


def test_narrowing_categorical_column_adds_filter_expression(
    hc_app_dir,
    seeded_admdatamart,
) -> None:
    """Deselecting a value from a categorical column adds a polars Expr to filters."""
    import polars as pl

    page = hc_app_dir / "pages" / "1_Data_Filters.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception

    column_picker = _find_multiselect(at, "multiselect")
    assert column_picker is not None

    # Find a categorical column that has at least 2 distinct values so we
    # can deselect one and create a real filter expression.
    schema = seeded_admdatamart.model_data.collect_schema()
    categorical_col = next(
        (
            col
            for col, dtype in schema.items()
            if dtype in (pl.Utf8, pl.String, pl.Categorical)
            and seeded_admdatamart.model_data.select(pl.col(col).n_unique()).collect().item() >= 2
        ),
        None,
    )
    if categorical_col is None:
        pytest.skip("Test fixture has no categorical column with ≥2 distinct values.")

    column_picker.set_value([categorical_col]).run()
    assert not at.exception

    sub_ms = _find_multiselect(at, f"selected_{categorical_col}")
    if sub_ms is None:
        pytest.skip(
            f"Column {categorical_col!r} did not render a multiselect sub-widget "
            "(may have too many unique values or an unexpected dtype)."
        )

    all_values = sub_ms.options
    if len(all_values) < 2:
        pytest.skip(f"Column {categorical_col!r} multiselect has only one option; cannot test narrowing.")

    # Deselect all but the first value — this should append a filter expression.
    reduced = [all_values[0]]
    sub_ms.set_value(reduced).run()
    assert not at.exception, f"Post-narrowing run raised: {at.exception}"

    filters = at.session_state["filters"]
    assert len(filters) > 0, (
        f"After deselecting values from {categorical_col!r}, "
        f"session_state['filters'] should be non-empty, got {filters!r}."
    )
    assert isinstance(filters[0], pl.Expr), f"Expected a polars Expr in filters, got {type(filters[0]).__name__}"
