"""Schema definitions for Impact Analyzer experiment data.

These classes mirror the convention used by ``pdstools.adm.Schema`` and document
the columns expected by :class:`ImpactAnalyzer`. Required columns are validated
in :meth:`ImpactAnalyzer._validate_ia_data`; optional columns (e.g. those
prefixed with ``Pega_``) are populated only by some data sources.
"""

from __future__ import annotations

import polars as pl


class ImpactAnalyzerData:
    """Normalised long-format experiment data consumed by :class:`ImpactAnalyzer`.

    Required columns
    ----------------
    SnapshotTime : pl.Datetime | pl.Date
        Snapshot date for the row.
    ControlGroup : pl.Utf8
        Control / test group identifier (e.g. ``"NBA"``, ``"PropensityPriority"``).
    Impressions : numeric
        Impression count.
    Accepts : numeric
        Accept count.
    ValuePerImpression : pl.Float64
        Average per-impression value (may be null for PDC data).
    Channel : pl.Utf8
        Channel name (or ``Channel/Direction`` for VBD data).

    Optional columns
    ----------------
    Pega_ValueLift, Pega_ValueLiftInterval : pl.Float64
        Pre-computed lift values present only in PDC exports.
    """

    SnapshotTime = pl.Datetime
    ControlGroup = pl.Utf8
    Impressions = pl.Float64
    Accepts = pl.Float64
    ValuePerImpression = pl.Float64
    Channel = pl.Utf8

    Pega_ValueLift = pl.Float64
    Pega_ValueLiftInterval = pl.Float64


REQUIRED_IA_COLUMNS: tuple[str, ...] = (
    "SnapshotTime",
    "ControlGroup",
    "Impressions",
    "Accepts",
    "ValuePerImpression",
    "Channel",
)
"""Columns that must be present on the LazyFrame passed to :class:`ImpactAnalyzer`."""
