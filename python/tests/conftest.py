"""Shared pytest fixtures and one-time setup for the pdstools test suite."""

import polars as pl


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
