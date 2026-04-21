"""Schema definitions for Interaction History data.

These classes mirror the convention used by :mod:`pdstools.adm.Schema` and
document the columns expected by :class:`pdstools.ih.IH.IH` after column
name normalisation (Pega ``py``/``px`` prefixes are stripped via
:func:`pdstools.utils.cdh_utils._polars_capitalize`).

The :data:`REQUIRED_IH_COLUMNS` tuple lists the minimum set of columns
that must be present on the LazyFrame passed to :class:`IH`. Any other
columns are optional and only used when present.
"""

from __future__ import annotations

import polars as pl


class IHInteraction:
    """Normalised Interaction History row schema.

    Required columns
    ----------------
    InteractionID : pl.Utf8
        Unique interaction identifier.
    Outcome : pl.Utf8
        Raw outcome label (e.g. ``"Impression"``, ``"Clicked"``).
    OutcomeTime : pl.Datetime
        Timestamp of the outcome. String values are parsed via
        :func:`pdstools.utils.cdh_utils.parse_pega_date_time_formats`.

    Optional but commonly present
    -----------------------------
    Channel, Direction, Issue, Group, Name, Treatment : pl.Utf8
        Action / context dimensions.
    Propensity : pl.Float64
        Model propensity emitted with the interaction.
    ModelTechnique : pl.Utf8
        Model family (e.g. ``"NaiveBayes"``, ``"GradientBoost"``).
    """

    InteractionID = pl.Utf8
    SubjectID = pl.Utf8
    Outcome = pl.Utf8
    OutcomeTime = pl.Datetime
    Channel = pl.Utf8
    Direction = pl.Utf8
    Issue = pl.Utf8
    Group = pl.Utf8
    Name = pl.Utf8
    Treatment = pl.Utf8
    Propensity = pl.Float64
    ModelTechnique = pl.Utf8


REQUIRED_IH_COLUMNS: tuple[str, ...] = (
    "InteractionID",
    "Outcome",
    "OutcomeTime",
)
"""Columns that must be present on the LazyFrame consumed by :class:`IH`."""
