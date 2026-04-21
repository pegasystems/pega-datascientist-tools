"""Smoke tests for the v5 subpackage ``__init__`` surface.

These pin the contract documented in ``docs/migration-v4-to-v5.md`` so
regressions to the public-import paths are caught immediately.
"""

import pdstools
from pdstools import decision_analyzer, pega_io, valuefinder
from pdstools.utils import __all__ as utils_all


def test_valuefinder_reexports_class():
    from pdstools.valuefinder import ValueFinder

    assert ValueFinder is pdstools.ValueFinder
    assert valuefinder.__all__ == ["ValueFinder"]


def test_utils_namespace_is_empty():
    assert utils_all == []


def test_pega_io_does_not_publicly_export_private_helper():
    assert "_read_client_credential_file" not in pega_io.__all__
    # …but the private name is still importable for callers that opt in.
    from pdstools.pega_io import _read_client_credential_file  # noqa: F401


def test_decision_analyzer_init_surface():
    assert set(decision_analyzer.__all__) == {"DecisionAnalyzer", "plots", "utils"}
    assert "da_streamlit_utils" not in decision_analyzer.__all__
    assert not hasattr(decision_analyzer, "da_streamlit_utils")
    # Internal sys import must not leak either.
    assert not hasattr(decision_analyzer, "sys")
