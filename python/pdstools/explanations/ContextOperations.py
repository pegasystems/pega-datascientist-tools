"""Context-related operations for querying the unique contexts in an aggregates set."""

from __future__ import annotations

import json
import logging
import os
from functools import cached_property
from typing import ClassVar, TYPE_CHECKING, cast

import polars as pl

from ..utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from pathlib import Path

    from .Aggregates import Aggregates

logger = logging.getLogger(__name__)

__all__ = ["ContextOperations"]


class ContextOperations(LazyNamespace):
    """Context-related operations for querying unique contexts.

    Parameters
    ----------
    aggregates : Aggregates
        Aggregates namespace instance that provides contextual explanation data.
    """

    dependencies: ClassVar[list[str]] = ["polars"]
    dependency_group = "explanations"

    def __init__(self, aggregates: Aggregates):
        self.aggregates = aggregates
        self.file_batch_limit = int(os.getenv("PDSTOOLS_FILE_BATCH_LIMIT", "100"))
        super().__init__()

    @cached_property
    def _contexts(self) -> pl.DataFrame:
        """Unique contexts, one row per context, including the raw partition column."""
        partitions = self.aggregates.contextual.select("context_partition").unique().collect().to_series().to_list()
        return pl.from_dicts(
            [{**json.loads(partition)["partition"], "context_partition": partition} for partition in partitions],
        )

    @property
    def unique_contexts_file(self) -> Path:
        """Path of the JSON file holding the context-to-batch mapping."""
        return self.aggregates.data_folderpath / "unique_contexts.json"

    @property
    def context_keys(self) -> list[str]:
        """Context key column names, for example ``["pyChannel", "pyDirection"]``."""
        return list(self._contexts.select(pl.col("^py.*$")).columns)

    def get_df(
        self,
        context_infos: list[dict[str, str]] | None = None,
        with_partition_col: bool = False,
    ) -> pl.DataFrame:
        """Return unique contexts as a DataFrame, optionally filtered.

        Parameters
        ----------
        context_infos : list[dict[str, str]] | None, default None
            Optional context filters. When provided, rows are filtered to the
            matching contexts.
        with_partition_col : bool, default False
            Whether to include the raw ``context_partition`` column in the output.

        Returns
        -------
        pl.DataFrame
            Unique contexts with one row per context.
        """
        df = self._contexts if with_partition_col else self._contexts.select(pl.exclude("context_partition"))

        if not context_infos:
            return df

        masks = [
            pl.all_horizontal(*(pl.col(name).eq(value) for name, value in context_info.items()))
            for context_info in context_infos
        ]
        return df.filter(pl.any_horizontal(*masks))

    def get_list(
        self,
        context_infos: list[dict[str, str]] | None = None,
        with_partition_col: bool = False,
    ) -> list[dict[str, str]]:
        """Return unique contexts as dictionaries, optionally filtered.

        Parameters
        ----------
        context_infos : list[dict[str, str]] | None, default None
            Optional context filters. When provided, rows are filtered to the
            matching contexts.
        with_partition_col : bool, default False
            Whether to include the raw ``context_partition`` field in each dictionary.

        Returns
        -------
        list[dict[str, str]]
            Unique contexts represented as dictionaries.
        """
        return cast(
            "list[dict[str, str]]",
            self.get_df(context_infos, with_partition_col).unique().to_dicts(),
        )

    def create_unique_contexts_file(self) -> dict[str, list[str]]:
        """Create and persist the flat unique-context batch mapping if absent.

        Returns
        -------
        dict[str, list[str]]
            Mapping of batch key to the context partitions in that batch.
        """
        if self.unique_contexts_file.exists():
            return cast("dict[str, list[str]]", json.loads(self.unique_contexts_file.read_text()))

        partitions = self.aggregates.contextual.select("context_partition").unique().collect().to_series().to_list()
        contexts_by_batch = self._create_context_batches(partitions, self.file_batch_limit)

        self.unique_contexts_file.write_text(json.dumps(contexts_by_batch), encoding="utf-8")

        return contexts_by_batch

    def create_batch_parquet_files(self, contexts_by_batch: dict[str, list[str]]) -> None:
        """Write one parquet file per context batch into a ``batches/`` subdirectory.

        Parameters
        ----------
        contexts_by_batch : dict[str, list[str]]
            Mapping of batch key to the context partitions in that batch.
        """
        batch_dir = self.aggregates.data_folderpath / "batches"
        batch_dir.mkdir(exist_ok=True)

        for batch_key, contexts in contexts_by_batch.items():
            batch_df = self.aggregates.contextual.filter(pl.col("context_partition").is_in(contexts)).collect()
            batch_file_path = batch_dir / f"BATCH_{batch_key}.parquet"
            batch_df.write_parquet(batch_file_path)
            logger.info("Created batch file: %s with %d rows", batch_file_path, len(batch_df))

    @staticmethod
    def get_context_info_str(context_info: dict[str, str], sep: str = "-") -> str:
        """Format a context dictionary into a compact string.

        Parameters
        ----------
        context_info : dict[str, str]
            Context dictionary to format.
        sep : str, default "-"
            Separator inserted between values.

        Returns
        -------
        str
            String containing context values joined by ``sep``.
        """
        return sep.join(f"{value}".strip() for value in context_info.values())

    @staticmethod
    def _create_context_batches(all_contexts: list[str] | None, batch_size: int) -> dict[str, list[str]]:
        if not all_contexts:
            return {}
        return {
            str(batch_idx): all_contexts[idx : idx + batch_size]
            for batch_idx, idx in enumerate(range(0, len(all_contexts), batch_size))
        }
