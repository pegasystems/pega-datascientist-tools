"""Pega-style field-name normalisation and predictor categorisation."""

import re
from collections.abc import Iterable

import polars as pl


def _capitalize(
    fields: str | Iterable[str],
    extra_endwords: Iterable[str] | None = None,
) -> list[str]:
    """Applies automatic capitalization, aligned with the R counterpart.

    Parameters
    ----------
    fields : list
        A list of names

    Returns
    -------
    fields : list
        The input list, but each value properly capitalized

    Notes
    -----
    The capitalize_endwords list contains atomic word parts that are commonly
    found in Pega field names. Compound words (like "ResponseCount") don't need
    to be listed separately because the algorithm processes words by length,
    allowing shorter components ("Response", "Count") to handle them.

    """
    capitalize_endwords = [
        "Active",
        "Adjusted",
        "Bin",
        "Bound",
        "Cap",
        "Category",
        "Class",
        "Code",
        "Component",
        "Configuration",
        "Context",
        "Control",
        "Count",
        "Date",
        "Description",
        "Email",
        "Enabled",
        "Error",
        "Evidence",
        "Execution",
        "Group",
        "Hash",
        "ID",
        "Identifier",
        "Importance",
        "Index",
        "Issue",
        "Key",
        "Limit",
        "Lower",
        "Message",
        "Model",
        "Name",
        "Negative",
        "Number",
        "Offline",
        "Omni",
        "Outcome",
        "Paid",
        "Percentage",
        "Performance",
        "Positive",
        "Prediction",
        "Predictor",
        "Propensity",
        "Proposition",
        "Rate",
        "Ratio",
        "Reference",
        "Relevant",
        "Response",
        "SMS",
        "Stage",
        "Strategy",
        "Subject",
        "Symbol",
        "Technique",
        "Template",
        "Threshold",
        "Time",
        "ToClass",  # Keep as compound - "To" alone is too generic
        "Treatment",
        "Type",
        "Update",
        "Upper",
        "URL",
        "Value",
        "Variant",
        "Version",
        "Web",
        "Weight",
    ]

    if not isinstance(fields, list):
        fields = [str(fields)]
    fields = [re.sub("^p(x|y|z)", "", field.lower()) for field in fields]
    fields = list(
        map(lambda x: x.replace("configurationname", "configuration"), fields),
    )
    # Sort by length ascending so longer words are processed last and can
    # "fix" any incorrect replacements made by shorter substring matches.
    # E.g., "Ratio" might corrupt "configuration" to "configuRation", but
    # processing "Configuration" after will correct it back.
    for word in sorted(capitalize_endwords, key=len):
        fields = [re.sub(word, word, field, flags=re.IGNORECASE) for field in fields]
    fields = [field[:1].upper() + field[1:] for field in fields]
    return fields


def default_predictor_categorization(
    x: str | pl.Expr = pl.col("PredictorName"),
) -> pl.Expr:
    """Function to determine the 'category' of a predictor.

    It is possible to supply a custom function.
    This function can accept an optional column as input
    And as output should be a Polars expression.
    The most straight-forward way to implement this is with
    pl.when().then().otherwise(), which you can chain.

    By default, this function returns "Primary" whenever
    there is no '.' anywhere in the name string,
    otherwise returns the first string before the first period

    Parameters
    ----------
    x: str | pl.Expr, default = pl.col('PredictorName')
        The column to parse

    """
    if isinstance(x, str):
        x = pl.col(x)
    x = x.cast(pl.Utf8) if not isinstance(x, pl.Utf8) else x
    return (
        pl.when(x == "Classifier")
        .then(pl.lit(None).cast(pl.Utf8))
        .when(x.str.split(".").list.len() > 1)
        .then(x.str.split(".").list.get(0))
        .otherwise(pl.lit("Primary"))
    ).alias("PredictorCategory")
