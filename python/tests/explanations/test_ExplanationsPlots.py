"""Test cases for Plots class that handles plotting of explanations data."""

from pathlib import Path
from unittest.mock import MagicMock

import plotly.graph_objects as go
import pytest
from pdstools.explanations import Explanations
from pdstools.explanations._constants import MISSING, REMAINING
from pdstools.explanations.Plots import Plots

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


@pytest.fixture(scope="module")
def plots():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations.from_aggregates(
        base_path=DATA_DIR,
        model_name="AdaptiveBoostCT",
    )
    explanations.filter = MagicMock()
    explanations.filter.is_context_selected.return_value = False
    yield explanations.plot


def test_context_table_figure():
    """Test the _context_table_figure method."""
    mock_context_info = {"key1": "value1", "key2": "value2"}
    fig = Plots._context_table_figure(mock_context_info)
    # Check that the returned object is a plotly Figure
    assert isinstance(fig, go.Figure)

    # Check that the figure contains a table with the expected data
    table = fig.data[0]
    assert table.header.values == ("Model context key", "Model context value")
    assert list(table.cells.values[0]) == list(mock_context_info.keys())
    assert list(table.cells.values[1]) == list(mock_context_info.values())


def test_contributions_overall_default_params(plots):
    """Test the contributions_overall method with default parameters."""
    overall_fig, predictors_figs = plots.contributions_overall()

    # Assertions
    assert isinstance(overall_fig, go.Figure)
    assert isinstance(predictors_figs, list)
    assert all(isinstance(fig, go.Figure) for fig in predictors_figs)

    # +1 for the remaining bar
    _assert_fig_bar_data_overall(
        overall_fig,
        20 + 1,
        check_condition="le",
    )
    _assert_fig_bar_data_predictors(predictors_figs, 1, check_condition="gt")


def test_contributions_overall_custom_params(plots):
    """Test the contributions_overall method with custom parameters."""
    top_n = 2
    top_k = 3

    # Call the method with custom parameters
    overall_fig, predictors_figs = plots.contributions_overall(
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


def test_contributions_overall_with_missing_for_age(plots):
    _, predictors_figs = plots.contributions_overall()
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs,
        "Age",
        check_missing=True,
        exists=True,
    )


def test_contributions_overall_without_missing_for_age(plots):
    _, predictors_figs = plots.contributions_overall(missing=False)
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs,
        "Age",
        check_missing=True,
        exists=False,
    )


def test_contributions_overall_with_remaining_for_eyecolor(plots):
    _, predictors_figs = plots.contributions_overall(top_k=2, remaining=True)
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs,
        "EyeColor",
        check_remaining=True,
        exists=True,
    )


def test_contributions_overall_without_remaining_for_eyecolor(plots):
    _, predictors_figs = plots.contributions_overall(top_k=2, remaining=False)
    _assert_fig_bar_data_predictors_special_bins(
        predictors_figs,
        "EyeColor",
        check_remaining=True,
        exists=False,
    )


def test_contributions_overall_with_invalid_contribution_type(plots):
    """Test the contributions_overall method with an invalid contribution type."""
    # Call the method with an invalid contribution type
    with pytest.raises(ValueError, match="Invalid contribution type"):
        _, _, _ = plots.contributions_overall(
            sort_by="invalid",
        )


def test_contributions_by_context_default_params(plots):
    selected_context = {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }
    # Call the method
    header_fig, overall_fig, predictors_figs = plots.contributions_by_context(
        context=selected_context,
    )

    # Assertions
    assert isinstance(header_fig, go.Figure)
    assert isinstance(overall_fig, go.Figure)
    assert isinstance(predictors_figs, list)
    assert all(isinstance(fig, go.Figure) for fig in predictors_figs)

    _assert_fig_bar_data_overall(overall_fig, 1, check_condition="gt")
    _assert_fig_bar_data_predictors(predictors_figs, 1, check_condition="gt")


def test_contributions_by_context_limit_top(plots):
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
    header_fig, overall_fig, predictors_figs = plots.contributions_by_context(
        context=selected_context,
        top_n=top_n,
        top_k=top_k,
    )

    # Assertions
    assert isinstance(header_fig, go.Figure)
    assert isinstance(overall_fig, go.Figure)
    assert isinstance(predictors_figs, list)
    assert all(isinstance(fig, go.Figure) for fig in predictors_figs)

    _assert_fig_bar_data_overall(overall_fig, 1, check_condition="gt")
    _assert_fig_bar_data_predictors(predictors_figs, 1, check_condition="gt")


def test_contributions_by_context_with_invalid_contribution_type(plots):
    """Test the contributions_by_context method with an invalid contribution."""
    selected_context = {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }
    # Call the method with an invalid contribution type
    with pytest.raises(ValueError, match="Invalid contribution type"):
        _, _, _ = plots.contributions_by_context(
            context=selected_context,
            sort_by="invalid",
        )


def test_contributions_overall_return_df(plots):
    """return_df=True must skip plotting and yield the underlying frames."""
    overall_df, predictors_df = plots.contributions_overall(
        top_n=5,
        top_k=5,
        return_df=True,
    )

    assert overall_df["predictor_name"].to_list() == [
        "pyName",
        "Age",
        "Occupation",
        "CustomerName",
        "NumX",
        REMAINING,
    ]
    assert predictors_df.group_by("predictor_name").len().sort("predictor_name").to_dict(as_series=False) == {
        "predictor_name": ["Age", "CustomerName", "NumX", "Occupation", "pyName"],
        # CustomerName and Occupation gain a row now that the forced MISSING bin is
        # actually appended; the others already had MISSING inside their top 5.
        "len": [6, 7, 6, 7, 6],
    }

    fig_overall, _ = plots.contributions_overall(top_n=5, top_k=5)
    fig_overall_y = list(fig_overall.data[0].y)

    # Predictor names plotted on the figure must match the names in the returned df.
    assert sorted(fig_overall_y) == sorted(overall_df["predictor_name"].to_list())


def test_contributions_by_context_return_df(plots):
    """return_df=True for the context variant returns (context_df, value_df)."""
    selected_context = {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }
    context_df, value_df = plots.contributions_by_context(
        context=selected_context,
        top_n=3,
        top_k=3,
        return_df=True,
    )

    assert context_df["predictor_name"].to_list() == ["Age", "Occupation", REMAINING]
    # Each predictor now also carries its forced MISSING bin, which the broken
    # _get_missing_predictor_values_df filter previously dropped.
    assert value_df.select("predictor_name", "bin_contents").to_dict(as_series=False) == {
        "predictor_name": [
            "Age",
            "Age",
            "Age",
            "Age",
            "Age",
            "Occupation",
            "Occupation",
            "Occupation",
            "Occupation",
            "Occupation",
        ],
        "bin_contents": [
            "MISSING",
            "remaining",
            "[25.000:32.000]",
            "[32.000:40.000]",
            "[40.000:45.000]",
            "remaining",
            "MISSING",
            "Geneticist, molecular",
            "Podiatrist",
            "Food technologist",
        ],
    }


def _assert_fig_bar_data_predictors_special_bins(
    predictor_figs,
    predictor_name,
    check_remaining=False,
    check_missing=False,
    exists=True,
):
    # An earlier version of this helper matched on trace.name (always None),
    # so it silently found nothing and asserted nothing. Look the figure up by
    # name so a miss fails here rather than passing vacuously.
    figs_by_predictor = {_get_predictor_name_from_fig(f): f for f in predictor_figs}
    assert predictor_name in figs_by_predictor, (
        f"No figure found for predictor {predictor_name!r}; got {sorted(figs_by_predictor)}"
    )
    bar_data = _get_bar_data_from_fig(figs_by_predictor[predictor_name])
    if check_missing:
        if exists:
            assert MISSING in bar_data.y
        else:
            assert MISSING not in bar_data.y
    if check_remaining:
        if exists:
            assert REMAINING in bar_data.y
        else:
            assert REMAINING not in bar_data.y


def _assert_fig_bar_data_predictors(
    figs,
    top_k,
    check_condition="eq",
    check_remaining=False,
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
    fig,
    y_len,
    check_condition="eq",
    check_remaining=False,
    check_missing=False,
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
        assert REMAINING in bar_data.y

    if check_missing:
        # Check if the missing bar is present
        assert MISSING in bar_data.y


def _get_bar_data_from_fig(fig):
    """Extracts bar data from a plotly figure."""
    if not fig.data or not isinstance(fig.data[0], go.Bar):
        raise ValueError("Figure does not contain bar data.")

    return fig.data[0]


def _get_predictor_type_from_fig(fig):
    """Extract predictor type from customdata (column index 1)."""
    return _get_bar_data_from_fig(fig).customdata[0][1]


def _get_predictor_name_from_fig(fig):
    """Extract predictor name from customdata (column index 0)."""
    return _get_bar_data_from_fig(fig).customdata[0][0]


# --- Tests for contributions() dispatcher ---
# NOTE: The `contributions()` dispatcher was removed from the Plots API.
# The validation and dispatch behaviours are covered by the
# contributions_overall / contributions_by_context tests above.


def test_contributions_overall_unknown_kwarg(plots):
    """Test contributions_overall() rejects unknown kwargs."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        plots.contributions_overall(unknown_param=True)


def test_contributions_overall_no_kwargs_uses_defaults(plots):
    """Calling with no filter kwargs should produce the same structure as passing explicit defaults."""
    _, figs_no_kwargs = plots.contributions_overall()
    _, figs_explicit = plots.contributions_overall(
        sort_by="contribution_abs",
        display_by="contribution",
        descending=True,
        missing=True,
        remaining=True,
        include_numeric_single_bin=False,
    )
    # Same number of predictor figures — same set of top predictors selected
    assert len(figs_no_kwargs) == len(figs_explicit)


def test_contributions_overall_with_kwargs_overrides_default(plots):
    """Passing filter kwargs should override the defaults and change the output."""
    _, figs_default = plots.contributions_overall()
    _, figs_no_remaining = plots.contributions_overall(remaining=False)
    # Without remaining bar the predictor figures should differ
    assert any(
        fig_a.to_json() != fig_b.to_json() for fig_a, fig_b in zip(figs_default, figs_no_remaining, strict=False)
    )


def test_contributions_overall_include_numeric_single_bin_default(plots):
    """Default (False) should produce same output as explicit False."""
    _, figs_default = plots.contributions_overall()
    _, figs_explicit = plots.contributions_overall(include_numeric_single_bin=False)
    assert len(figs_default) == len(figs_explicit)


def test_contributions_overall_include_numeric_single_bin_true(plots):
    """Passing include_numeric_single_bin=True should be accepted and may include extra predictors."""
    _, figs_default = plots.contributions_overall()
    _, figs_with_single = plots.contributions_overall(include_numeric_single_bin=True)
    # With single-bin numerics included, we should get at least as many predictor figures
    assert len(figs_with_single) >= len(figs_default)
