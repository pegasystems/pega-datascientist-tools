"""Shared pytest fixtures and one-time setup for the pdstools test suite."""

import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    # These process-local defaults apply only inside pytest workers and their
    # subprocesses; they cannot affect normal pdstools or Polars usage.
    for variable in (
        "POLARS_MAX_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

import polars as pl


def pytest_xdist_auto_num_workers(config) -> int | None:
    """Cap local ``-n auto`` runs while leaving CI worker selection automatic."""
    if os.environ.get("CI") or config.getoption("numprocesses") != "auto":
        return None
    return 4


@pl.api.register_lazyframe_namespace("shape")
class _LazyShape:
    """Get the shape of a lazy dataframe.

    Registered once for the whole test session so that LazyFrames expose
    ``.shape`` interchangeably with eager DataFrames in assertions. Defined
    here (rather than per-test-module) to avoid the polars
    ``overriding existing custom namespace`` warning.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (
            ldf.select(pl.first().len()).collect().item(),
            len(ldf.collect_schema().names()),
        )
