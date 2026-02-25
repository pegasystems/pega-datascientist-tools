from collections.abc import Iterable
from typing import Union

import polars as pl
from typing_extensions import TypeAlias

ANY_FRAME: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]
QUERY: TypeAlias = Union[pl.Expr, Iterable[pl.Expr], dict[str, list]]
