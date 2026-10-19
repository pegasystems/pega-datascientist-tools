"""Shared ADM vocabulary."""

import polars as pl

MODEL_TECHNIQUE_LABELS: dict[str, str] = {
    "NaiveBayes": "Naive Bayes",
    "GradientBoost": "Adaptive Gradient Boosting",
}
"""User-facing names for the raw ``ModelTechnique`` values in the datamart."""

MODEL_TECHNIQUE_ORDER: list[str] = ["Naive Bayes", "Adaptive Gradient Boosting"]
"""Fixed display order of the model techniques in plots and reports."""


def model_technique_label(column: str = "ModelTechnique") -> pl.Expr:
    """Translate raw ``ModelTechnique`` values into user-facing names.

    The datamart only records the technique since Pega Infinity '24, the
    release that introduced Adaptive Gradient Boosting. A missing technique
    therefore means Naive Bayes. Unknown values are passed through unchanged.

    Parameters
    ----------
    column : str, default "ModelTechnique"
        Name of the column holding the raw technique.

    Returns
    -------
    pl.Expr
        String expression with the display name, keeping the column name.

    Examples
    --------
    >>> pl.DataFrame({"ModelTechnique": ["GradientBoost", None]}).select(model_technique_label())["ModelTechnique"].to_list()
    ['Adaptive Gradient Boosting', 'Naive Bayes']
    """
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .replace(MODEL_TECHNIQUE_LABELS)
        .fill_null(MODEL_TECHNIQUE_LABELS["NaiveBayes"])
        .alias(column)
    )
