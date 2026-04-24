"""Shared internals used across the cdh_utils submodules."""

from __future__ import annotations

import logging
from typing import TypeVar

import polars as pl

logger = logging.getLogger("pdstools.utils.cdh_utils")

F = TypeVar("F", pl.DataFrame, pl.LazyFrame)
