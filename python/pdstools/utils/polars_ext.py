import polars as pl


@pl.api.register_lazyframe_namespace("pdstools")
class Sample:
    """Sample."""

    def __init__(self, ldf: pl.LazyFrame) -> None:
        self._ldf = ldf

    def sample(self, n):
        """Sample."""
        from functools import partial

        def sample_it(s: pl.Series, n) -> pl.Series:
            import numpy as np

            s_len = s.len()
            if s_len < n:
                return pl.Series(values=[True] * s_len, dtype=pl.Boolean)
            return pl.Series(
                values=np.random.binomial(1, n / s_len, s_len),
                dtype=pl.Boolean,
            )

        func = partial(sample_it, n=n)
        return (
            self._ldf.with_columns(
                pl.first().map_batches(func, return_dtype=pl.Boolean).alias("_sample"),
            )
            .filter(pl.col("_sample"))
            .drop("_sample")
        )

    def height(self):
        """Height."""
        return self._ldf.select(pl.first().len()).collect().item()

    def shape(self):
        """Shape."""
        return (self.height(), len(self._ldf.collect_schema().names()))

    def item(self):
        """Item."""
        if self.shape() == (1, 1):
            return self._ldf.collect().item()
