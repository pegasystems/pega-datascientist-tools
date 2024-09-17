from typing import Dict, Iterable, List, Union

import polars as pl
from typing_extensions import TypeAlias

ANY_FRAME: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]
QUERY: TypeAlias = Union[pl.Expr, Iterable[pl.Expr], Dict[str, List]]
