"""Seeded synthetic CDH demo data: one world, many views.

:func:`demo_data` returns the datamart, prediction and Interaction History
views of one synthetic world. The ``from_mock_data`` methods on
:class:`~pdstools.ADMDatamart`, :class:`~pdstools.Prediction`,
:class:`~pdstools.IH` and :class:`~pdstools.ImpactAnalyzer` return a
single view of the same world.
"""

from ._demo import DemoData, demo_data

__all__ = ["DemoData", "demo_data"]
