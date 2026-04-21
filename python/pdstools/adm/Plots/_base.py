"""Shared base for the :class:`Plots` mixins.

Each mixin in this package references ``self.datamart``. The base class
declares that attribute so the mixins type-check independently of the
final composed :class:`Plots` class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..ADMDatamart import ADMDatamart


class _PlotsBase:
    """Common attribute surface used by every plot mixin."""

    datamart: "ADMDatamart"
