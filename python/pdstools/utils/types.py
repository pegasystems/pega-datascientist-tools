from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import polars as pl

ANY_FRAME: TypeAlias = pl.DataFrame | pl.LazyFrame
QUERY: TypeAlias = pl.Expr | Iterable[pl.Expr] | dict[str, list]
