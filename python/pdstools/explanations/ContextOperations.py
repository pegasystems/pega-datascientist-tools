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

    from .Explanations import Explanations

logger = logging.getLogger(__name__)

__all__ = ["ContextOperations"]


class ContextOperations(LazyNamespace):
    """Context-related operations for querying unique contexts.

    Parameters
    ----------
    explanations : Explanations
        Parent instance providing the contextual data and the data folder.
    """

    dependencies: ClassVar[list[str]] = ["polars"]
    dependency_group = "explanations"

    def __init__(self, explanations: Explanations):
        self.explanations = explanations
        self.file_batch_limit = int(os.getenv("PDSTOOLS_FILE_BATCH_LIMIT", "100"))
        super().__init__()

    @cached_property
    def _contexts(self) -> pl.DataFrame:
        """Unique contexts, one row per context, including the raw partition column."""
        partitions = (
            self.explanations.contextual.select("context_partition").unique().sort("context_partition").collect()
        )
        # infer_schema_length=None scans every row: the default of 100 would raise on
        # (or, with the old from_dicts approach, silently drop) a context key that
        # first appears beyond the 100th partition.
        decoded = partitions["context_partition"].str.json_decode(infer_schema_length=None).struct.field("partition")
        return decoded.to_frame().unnest("partition").with_columns(partitions["context_partition"])

    @property
    def unique_contexts_file(self) -> Path:
        """Path of the JSON file holding the context-to-batch mapping."""
        return self.explanations.data_folderpath / "unique_contexts.json"

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
        """Split the unique contexts into batches and persist the mapping.

        The JSON file is an interchange artifact for the report subprocess, not
        a cache: it is rewritten on every call so that a change of dataset or of
        ``PDSTOOLS_FILE_BATCH_LIMIT`` takes effect.

        Returns
        -------
        dict[str, list[str]]
            Mapping of batch key to the context partitions in that batch.
        """
        partitions = self._contexts["context_partition"].to_list()
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
        batch_dir = self.explanations.data_folderpath / "batches"
        batch_dir.mkdir(exist_ok=True)

        batch_of = pl.LazyFrame(
            {
                "context_partition": [ctx for contexts in contexts_by_batch.values() for ctx in contexts],
                "batch_key": [key for key, contexts in contexts_by_batch.items() for _ in contexts],
            }
        )
        # Join the batch assignment on and collect once, rather than re-filtering
        # the whole frame per batch.
        tagged = self.explanations.contextual.join(batch_of, on="context_partition", how="inner").collect()

        for (batch_key,), batch_df in tagged.partition_by("batch_key", as_dict=True).items():
            batch_file_path = batch_dir / f"BATCH_{batch_key}.parquet"
            batch_df.drop("batch_key").write_parquet(batch_file_path)
            logger.info("Created batch file: %s with %d rows", batch_file_path, batch_df.height)

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
