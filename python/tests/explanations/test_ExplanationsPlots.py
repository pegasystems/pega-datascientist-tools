"""Test cases for Plots class that handles plotting of explanations data."""

import shutil
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import pytest
from pdstools.explanations import Explanations
from pdstools.explanations.ExplanationsUtils import _DEFAULT, _SPECIAL
from pdstools.explanations.Plots import Plots

basePath = Path(__file__).parent.parent.parent.parent

def clean_up(root_dir):
    _root_dir = Path(f'{basePath}/{root_dir}')
    if _root_dir.exists():
        for file in _root_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                # Remove subdirectories recursively
                shutil.rmtree(file)
        _root_dir.rmdir()

@pytest.fixture(scope="module")
def plots():
    """Fixture to serve as class to call functions from."""

    explanations = Explanations(
        data_folder=f'{basePath}/data/explanations',
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield explanations.plot

    # cleanup .tmp folder
    clean_up(explanations.root_dir)


def test_plot_context_table():
    """Test the _plot_context_table method."""

    mock_context_info = {"key1": "value1", "key2": "value2"}
    fig = Plots._plot_context_table(mock_context_info)
    # Check that the returned object is a plotly Figure
    assert isinstance(fig, go.Figure)

    # Check that the figure contains a table with the expected data
    table = fig.data[0]
    assert table.header.values == ("Model context key", "Model context value")
    assert list(table.cells.values[0]) == list(mock_context_info.keys())
    assert list(table.cells.values[1]) == list(mock_context_info.values())


def test_plot_contributions_for_overall_default_params(plots):
    """Test the plot_contributions_for_overall method with default parameters."""

    overall_fig, predictors_figs = plots.plot_contributions_for_overall()

    # Assertions
    assert isinstance(overall_fig, go.Figure)
    assert isinstance(predictors_figs, list)
    assert all(isinstance(fig, go.Figure) for fig in predictors_figs)

    # +1 for the remaining bar
    _assert_fig_bar_data_overall(overall_fig, _DEFAULT.TOP_N.value + 1, check_condition="le")
    _assert_fig_bar_data_predictors(predictors_figs, 1, check_condition="gt")


def test_plot_contributions_for_overall_custom_params(plots):
    """Test the plot_contributions_for_overall method with custom parameters."""

    top_n = 2
    top_k = 3

    # Call the method with custom parameters
    overall_fig, predictors_figs = plots.plot_contributions_for_overall(
        top_n=top_n,
        top_k=top_k,
    )

    # Assertions
    assert isinstance(overall_fig, go.Figure)
    assert isinstance(predictors_figs, list)
    assert all(isinstance(fig, go.Figure) for fig in predictors_figs)

    # Will also have remaining bar, so +1
    _assert_fig_bar_data_overall(overall_fig, top_n + 1)
    _assert_fig_bar_data_predictors(predictors_figs, top_k + 1, check_remaining=True)


def test_plot_contributions_for_overall_with_missing_for_age(plots):
    _, predictors_figs = plots.plot_contributions_for_overall()
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs, "Age", check_missing=True, exists=True
    )


def test_plot_contributions_for_overall_without_missing_for_age(plots):
    _, predictors_figs = plots.plot_contributions_for_overall(missing=False)
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs, "Age", check_missing=True, exists=False
    )


def test_plot_contributions_for_overall_with_remaining_for_eyecolor(plots):
    _, predictors_figs = plots.plot_contributions_for_overall(top_k=2, remaining=True)
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs, "EyeColor", check_remaining=True, exists=True
    )


def test_plot_contributions_for_overall_without_remaining_for_eyecolor(plots):
    _, predictors_figs = plots.plot_contributions_for_overall(top_k=2, remaining=False)
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs, "EyeColor", check_remaining=True, exists=False
    )

def test_plot_contributions_for_overall_with_invalid_contribution_type(plots):
    """Test the plot_contributions_for_overall method with an invalid contribution type."""

    # Call the method with an invalid contribution type
    with pytest.raises(ValueError, match="Invalid contribution type"):
        _, _, _ = plots.plot_contributions_for_overall(
            contribution_calculation="invalid"
        )


def test_plot_contributions_by_context_default_params(plots):
    selected_context = {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }
    # Call the method
    header_fig, overall_fig, predictors_figs = plots.plot_contributions_by_context(
        context=selected_context
    )

    # Assertions
    assert isinstance(header_fig, go.Figure)
    assert isinstance(overall_fig, go.Figure)
    assert isinstance(predictors_figs, list)
    assert all(isinstance(fig, go.Figure) for fig in predictors_figs)

    _assert_fig_bar_data_overall(overall_fig, 1, check_condition="gt")
    _assert_fig_bar_data_predictors(predictors_figs, 1, check_condition="gt")


def test_plot_contributions_by_context_limit_top(plots):
    selected_context = {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }
    top_n = 2
    top_k = 3

    # Call the method with custom parameters
    header_fig, overall_fig, predictors_figs = plots.plot_contributions_by_context(
        context=selected_context,
        top_n=top_n,
        top_k=top_k,
    )

    # Assertions
    assert isinstance(header_fig, go.Figure)
    assert isinstance(overall_fig, go.Figure)
    assert isinstance(predictors_figs, list)
    assert all(isinstance(fig, go.Figure) for fig in predictors_figs)

    _assert_fig_bar_data_overall(overall_fig, 1 , check_condition="gt")
    _assert_fig_bar_data_predictors(predictors_figs, 1, check_condition="gt")

def test_plot_contributions_by_context_with_invalid_contribution_type(plots):
    """Test the plot_contributions_by_context method with an invalid contribution."""
    selected_context = {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }
    # Call the method with an invalid contribution type
    with pytest.raises(ValueError, match="Invalid contribution type"):
        _, _, _ = plots.plot_contributions_by_context(
            context=selected_context,
            contribution_calculation="invalid"
        )


def _assert_fig_bar_data_predictors_special_bins(
    predictor_figs,
    predictor_name,
    check_remaining=False,
    check_missing=False,
    exists=True,
):
    for fig in predictor_figs:
        bar_data = _get_bar_data_from_fig(fig)
        if bar_data.name == predictor_name:
            if check_missing:
                if exists:
                    assert _SPECIAL.MISSING.value in bar_data.y
                else:
                    assert _SPECIAL.MISSING.value not in bar_data.y
            if check_remaining:
                if exists:
                    assert _SPECIAL.REMAINING.value in bar_data.y
                else:
                    assert _SPECIAL.REMAINING.value not in bar_data.y
            return


def _assert_fig_bar_data_predictors(
    figs, top_k, check_condition="eq", check_remaining=False
):
    for fig in figs:
        if _get_predictor_type_from_fig(fig) == "SYMBOLIC":
            _assert_fig_bar_data_overall(
                fig,
                top_k,
                check_condition=check_condition,
                check_remaining=check_remaining,
            )


def _assert_fig_bar_data_overall(
    fig, y_len, check_condition="eq", check_remaining=False, check_missing=False
):
    bar_data = _get_bar_data_from_fig(fig)

    if check_condition == "eq":
        assert len(bar_data.y) == y_len
    elif check_condition == "gt":
        assert len(bar_data.y) >= y_len
    elif check_condition == "lt":
        assert len(bar_data.y) < y_len
    elif check_condition == "le":
        assert len(bar_data.y) <= y_len

    if check_remaining:
        # Check if the remaining bar is present
        assert _SPECIAL.REMAINING.value in bar_data.y

    if check_missing:
        # Check if the missing bar is present
        assert _SPECIAL.MISSING.value in bar_data.y


def _get_bar_data_from_fig(fig):
    """Extracts bar data from a plotly figure."""
    if not fig.data or not isinstance(fig.data[0], go.Bar):
        raise ValueError("Figure does not contain bar data.")

    return fig.data[0]


def _get_predictor_type_from_fig(fig):
    return _get_bar_data_from_fig(fig).customdata[0]
