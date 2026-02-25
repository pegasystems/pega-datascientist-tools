from collections.abc import Iterable

import polars as pl
from typing_extensions import TypeAlias

ANY_FRAME: TypeAlias = pl.DataFrame | pl.LazyFrame
QUERY: TypeAlias = pl.Expr | Iterable[pl.Expr] | dict[str, list]
