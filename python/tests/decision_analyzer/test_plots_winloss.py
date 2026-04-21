"""Unit tests for ``decision_analyzer.plots._winloss.create_win_distribution_plot``.

These exercise the non-Action aggregation path (lines 146-226 of
``_winloss.py``) using small synthetic Polars frames so every expected value
can be traced back to the input.
"""

from __future__ import annotations

import polars as pl
import pytest
from plotly.graph_objs import Figure

from pdstools.decision_analyzer.plots._winloss import create_win_distribution_plot


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

WIN_COL = "new_win_count"


def _group_scope(selected_value: str = "Cards") -> dict:
    return {
        "level": "Group",
        "group_cols": ["Issue", "Group"],
        "x_col": "Group",
        "selected_value": selected_value,
        "plot_title_prefix": "Win Count by Group",
    }


def _issue_scope(selected_value: str = "Sales") -> dict:
    return {
        "level": "Issue",
        "group_cols": ["Issue"],
        "x_col": "Issue",
        "selected_value": selected_value,
        "plot_title_prefix": "Win Count by Issue",
    }


def _action_scope(selected_value: str = "GoldCard") -> dict:
    return {
        "level": "Action",
        "group_cols": ["Issue", "Group", "Action"],
        "x_col": "Action",
        "selected_value": selected_value,
        "plot_title_prefix": "Win Count by Action",
    }


@pytest.fixture
def mixed_data() -> pl.DataFrame:
    """Three regular actions across two groups + a 'No Winner' row.

    The no-winner row mirrors how ``get_win_distribution_data`` synthesises
    it upstream: the Issue/Group/Action triple are all literally
    ``"No Winner"`` so the orange-bar lookup at any aggregation level matches.
    """
    return pl.DataFrame(
        {
            "Issue": ["Sales", "Sales", "Sales", "No Winner"],
            "Group": ["Cards", "Cards", "Loans", "No Winner"],
            "Action": ["GoldCard", "SilverCard", "PersonalLoan", "No Winner"],
            WIN_COL: [10, 4, 7, 3],
        }
    )


# ---------------------------------------------------------------------------
# Action-level early return
# ---------------------------------------------------------------------------


def test_action_level_returns_data_unchanged(mixed_data):
    fig, plot_data = create_win_distribution_plot(
        mixed_data, WIN_COL, _action_scope("GoldCard"), "Current", "Win Count"
    )
    assert isinstance(fig, Figure)
    assert plot_data.equals(mixed_data)
    # Selected GoldCard => red, No Winner => orange, others => grey.
    colors = list(fig.data[0]["marker_color"])
    expected = ["grey"] * mixed_data.height
    expected[mixed_data["Action"].to_list().index("GoldCard")] = "#FF0000"
    expected[mixed_data["Action"].to_list().index("No Winner")] = "#FFA500"
    assert colors == expected
    # Action-level data exposes both Group and Issue, so the per-action
    # hover branch (customdata as list of (Group, Issue) tuples) is taken.
    assert len(fig.data[0]["customdata"]) == mixed_data.height
    assert all(len(row) == 2 for row in fig.data[0]["customdata"])


# ---------------------------------------------------------------------------
# Group level: regular_data + no_winner_data
# ---------------------------------------------------------------------------


def test_group_level_aggregates_and_appends_no_winner(mixed_data):
    fig, plot_data = create_win_distribution_plot(mixed_data, WIN_COL, _group_scope("Cards"), "Current", "Win Count")
    assert isinstance(fig, Figure)

    # Three rows expected: Cards (10+4=14), Loans (7), No Winner (3).
    rows = {row["Group"]: (row["Issue"], row[WIN_COL]) for row in plot_data.iter_rows(named=True)}
    assert rows == {
        "Cards": ("Sales", 14),
        "Loans": ("Sales", 7),
        "No Winner": ("No Winner", 3),
    }
    # Sort: regular rows are sorted by win count descending; the No Winner
    # row is appended at the end.
    assert plot_data["Group"].to_list() == ["Cards", "Loans", "No Winner"]

    # The Group x_col branch produces single-column customdata (Issue).
    assert list(fig.data[0]["customdata"]) == ["Sales", "Sales", "No Winner"]

    # Cards (selected) => red; the "No Winner" group => orange.
    assert list(fig.data[0]["marker_color"]) == ["#FF0000", "grey", "#FFA500"]
    assert fig.layout.xaxis.title.text == "Group"
    assert "Selected: Cards" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Group level: only regular data (no_winner branch skipped)
# ---------------------------------------------------------------------------


def test_group_level_without_no_winner(mixed_data):
    data = mixed_data.filter(pl.col("Action") != "No Winner")
    fig, plot_data = create_win_distribution_plot(data, WIN_COL, _group_scope("Loans"), "Current", "Win Count")
    assert plot_data["Group"].to_list() == ["Cards", "Loans"]
    assert plot_data[WIN_COL].to_list() == [14, 7]
    # Selected Loans => red; no orange (no "No Winner" row).
    assert list(fig.data[0]["marker_color"]) == ["grey", "#FF0000"]


# ---------------------------------------------------------------------------
# Group level: only "No Winner" data (regular_data empty)
# ---------------------------------------------------------------------------


def test_group_level_only_no_winner_data():
    data = pl.DataFrame(
        {
            "Issue": ["No Winner"],
            "Group": ["No Winner"],
            "Action": ["No Winner"],
            WIN_COL: [9],
        }
    )
    fig, plot_data = create_win_distribution_plot(data, WIN_COL, _group_scope("Cards"), "Current", "Win Count")
    # Only the no-winner-projected columns survive.
    assert plot_data.columns == ["Issue", "Group", WIN_COL]
    assert plot_data.height == 1
    assert plot_data[WIN_COL].to_list() == [9]
    # Selected value not present -> no red. No Winner present -> orange.
    assert list(fig.data[0]["marker_color"]) == ["#FFA500"]


# ---------------------------------------------------------------------------
# Group level: completely empty input
# ---------------------------------------------------------------------------


def test_group_level_all_empty_raises():
    """Both branches empty produces an unschemed DataFrame; downstream
    indexing raises ColumnNotFoundError. Documenting current behaviour."""
    schema = {"Issue": pl.Utf8, "Group": pl.Utf8, "Action": pl.Utf8, WIN_COL: pl.Int64}
    data = pl.DataFrame(schema=schema)
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        create_win_distribution_plot(data, WIN_COL, _group_scope("Cards"), "Current", "Win Count")


# ---------------------------------------------------------------------------
# Issue level aggregation
# ---------------------------------------------------------------------------


def test_issue_level_aggregates_across_groups(mixed_data):
    fig, plot_data = create_win_distribution_plot(
        mixed_data, WIN_COL, _issue_scope("Sales"), "After Lever", "New Win Count"
    )
    rows = dict(zip(plot_data["Issue"].to_list(), plot_data[WIN_COL].to_list(), strict=True))
    # Sales = 10 + 4 + 7, the No Winner row keeps Issue == "No Winner" with 3.
    assert rows == {"Sales": 21, "No Winner": 3}
    assert plot_data["Issue"].to_list() == ["Sales", "No Winner"]
    # No Group column at Issue level => default hover branch (no customdata).
    assert fig.data[0]["customdata"] is None
    # Selected Sales => red; the appended No Winner row => orange.
    assert list(fig.data[0]["marker_color"]) == ["#FF0000", "#FFA500"]
    assert fig.layout.yaxis.title.text == "New Win Count"
