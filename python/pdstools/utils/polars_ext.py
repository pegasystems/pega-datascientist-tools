import polars as pl


@pl.api.register_lazyframe_namespace("pdstools")
class Sample:
    def __init__(self, ldf: pl.LazyFrame) -> pl.LazyFrame:
        self._ldf = ldf

    def sample(self, n):
        from functools import partial

        def sample_it(s: pl.Series, n) -> pl.Series:
            import numpy as np

            s_len = s.len()
            if s_len < n:
                return pl.Series(values=[True] * s_len, dtype=pl.Boolean)
            else:
                return pl.Series(
                    values=np.random.binomial(1, n / s_len, s_len),
                    dtype=pl.Boolean,
                )

        func = partial(sample_it, n=n)
        return (
            self._ldf.with_columns(pl.first().map(func).alias("_sample"))
            .filter(pl.col("_sample"))
            .drop("_sample")
        )

    def height(self):
        return self._ldf.select(pl.first().count()).collect().item()

    def shape(self):
        return (self.height(), len(self._ldf.columns))

    def item(self):
        if self.shape() == (1, 1):
            return self._ldf.collect().item()
