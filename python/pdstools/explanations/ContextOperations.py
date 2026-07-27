"""Context-related operations for querying the unique contexts in an aggregates set."""

from __future__ import annotations

import json
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import polars as pl

from ..utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from .Explanations import Explanations

logger = logging.getLogger(__name__)

__all__ = ["ContextOperations"]


class ContextOperations(LazyNamespace):
    """Context-related operations for querying unique contexts.

    Parameters
    ----------
    explanations : Explanations
        Parent instance providing the contextual data.
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

    def write_batches(self, target_dir: str | Path) -> None:
        """Write the per-batch parquet files and the context-to-batch mapping.

        The report renders one page per batch, so contexts are chunked into
        groups of ``file_batch_limit``. Both artifacts come out of the same
        assignment and cannot disagree: ``unique_contexts.json`` tells the
        report subprocess which contexts belong on which page, and each
        ``batches/BATCH_<n>.parquet`` holds exactly that page's rows.

        Nothing is cached — the files are rewritten on every call, so a change
        of dataset or of ``PDSTOOLS_FILE_BATCH_LIMIT`` takes effect.

        Parameters
        ----------
        target_dir : str | Path
            Directory to write ``unique_contexts.json`` and ``batches/`` into.
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        # _contexts is already unique and sorted, so a row index chunks it
        # directly. It is one row per context, so the mapping is cheap to
        # materialise even when the contextual data itself is large.
        batch_of_context = self._contexts.select(
            "context_partition",
            batch_key=pl.int_range(pl.len()) // self.file_batch_limit,
        )
        contexts_by_batch = {
            str(batch_key): partitions
            for batch_key, partitions in batch_of_context.group_by("batch_key", maintain_order=True)
            .agg("context_partition")
            .iter_rows()
        }
        (target / "unique_contexts.json").write_text(json.dumps(contexts_by_batch), encoding="utf-8")

        # Streamed straight to disk: the contextual frame is never collected.
        # pl.PartitionBy is marked unstable by polars, but emits no warning unless
        # POLARS_WARN_UNSTABLE is set. The fallback if it changes is a single
        # collect() + DataFrame.partition_by(as_dict=True).
        self.explanations.contextual.join(batch_of_context.lazy(), on="context_partition", how="inner").sink_parquet(
            pl.PartitionBy(
                target / "batches",
                key="batch_key",
                include_key=False,
                file_path_provider=lambda args: f"BATCH_{args.partition_keys.item()}.parquet",
            ),
            mkdir=True,
        )

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
