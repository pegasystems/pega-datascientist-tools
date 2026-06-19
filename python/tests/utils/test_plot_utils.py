import pandas as pd
import plotly.express as px

from pdstools.utils.plot_utils import fig_update_facet


def _make_faceted_scatter(n_facets: int, col_wrap: int = 2) -> "px.Figure":
    labels = [chr(65 + i) for i in range(n_facets)]  # A, B, C, ...
    df = pd.DataFrame(
        {
            "x": range(n_facets * 2),
            "y": range(n_facets * 2),
            "f": [label for label in labels for _ in range(2)],
        }
    )
    return px.scatter(df, x="x", y="y", facet_col="f", facet_col_wrap=col_wrap)


class TestFigUpdateFacet:
    def test_height_single_row(self):
        """1 facet → 1 row → base_height + 1 * step_height."""
        fig = _make_faceted_scatter(1)
        result = fig_update_facet(fig, n_cols=2, base_height=250, step_height=270)
        assert result.layout.height == 250 + 1 * 270

    def test_height_three_rows(self):
        """5 facets, col_wrap=2 → 3 rows."""
        fig = _make_faceted_scatter(5, col_wrap=2)
        result = fig_update_facet(fig, n_cols=2, base_height=250, step_height=270)
        assert result.layout.height == 250 + 3 * 270

    def test_vline_annotations_do_not_inflate_height(self):
        """add_vline with annotation_text creates one annotation per subplot.

        Before the fix, fig_update_facet counted ALL annotations to derive
        n_rows, so 4 facets + 2 vlines × 4 subplots = 12 annotations → n_rows=6
        → a massively inflated height. Now only facet-title annotations
        (those containing "=" in their text) are counted.
        """
        fig = _make_faceted_scatter(4, col_wrap=2)
        fig.add_vline(x=1, annotation_text="Line1")
        fig.add_vline(x=3, annotation_text="Line2")

        # Sanity-check: vlines added 2×4=8 extra annotations on top of 4 facet titles
        assert len(fig.layout.annotations) == 12

        result = fig_update_facet(fig, n_cols=2, base_height=250, step_height=270)
        assert result.layout.height == 250 + 2 * 270  # still 2 rows, not 6

    def test_facet_titles_simplified(self):
        """Facet annotations like 'f=A' are simplified to 'A'."""
        fig = _make_faceted_scatter(2)
        result = fig_update_facet(fig, n_cols=2)
        titles = [a.text for a in result.layout.annotations if a.text]
        assert all("=" not in t for t in titles)
        assert set(titles) == {"A", "B"}

    def test_custom_step_height(self):
        """step_height parameter is honoured."""
        fig = _make_faceted_scatter(4, col_wrap=2)
        result = fig_update_facet(fig, n_cols=2, base_height=200, step_height=400)
        assert result.layout.height == 200 + 2 * 400

    def test_returns_figure(self):
        """Return value is the same figure (for chaining)."""
        fig = _make_faceted_scatter(2)
        result = fig_update_facet(fig)
        assert result is fig
