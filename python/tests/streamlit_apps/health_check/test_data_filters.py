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
    assert column_picker.options == seeded_admdatamart.model_data.collect_schema().names()
    assert column_picker.value == [], "Column picker should be empty on first visit."


def test_selecting_column_renders_sub_widget(hc_app_dir, seeded_admdatamart) -> None:
    """Picking a categorical column from the column picker renders a per-column widget."""
    page = hc_app_dir / "pages" / "1_Data_Filters.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    column_picker = _find_multiselect(at, "multiselect")
    assert "Channel" in column_picker.options
    column_picker.set_value(["Channel"]).run()
    assert not at.exception, f"Post-selection run raised: {at.exception}"

    assert [widget.key for widget in at.multiselect if widget.key == "selected_Channel"] == ["selected_Channel"]


def test_narrowing_categorical_column_adds_filter_expression(
    hc_app_dir,
    seeded_admdatamart,
) -> None:
    """Deselecting a value from a categorical column adds a polars Expr to filters."""
    page = hc_app_dir / "pages" / "1_Data_Filters.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception

    column_picker = _find_multiselect(at, "multiselect")
    assert "Channel" in column_picker.options
    column_picker.set_value(["Channel"]).run()
    assert not at.exception

    sub_ms = _find_multiselect(at, "selected_Channel")
    assert set(sub_ms.options) == {"Email", "SMS", "Web"}

    reduced = ["Email"]
    sub_ms.set_value(reduced).run()
    assert not at.exception, f"Post-narrowing run raised: {at.exception}"

    filters = at.session_state["filters"]
    filtered_channels = (
        seeded_admdatamart.model_data.filter(*filters).select("Channel").unique().collect().to_series().to_list()
    )
    assert filtered_channels == reduced
