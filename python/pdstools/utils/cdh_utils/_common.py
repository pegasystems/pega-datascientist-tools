"""Shared internals used across the cdh_utils submodules."""

import logging
from typing import TypeVar

import polars as pl

logger = logging.getLogger("pdstools.utils.cdh_utils")

F = TypeVar("F", pl.DataFrame, pl.LazyFrame)
