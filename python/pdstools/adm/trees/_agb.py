"""Datamart-level AGB discovery helper."""

from __future__ import annotations

import base64
import json
import logging
import zlib
from typing import TYPE_CHECKING

import polars as pl

from ...utils import cdh_utils
from ...utils.types import QUERY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..ADMDatamart import ADMDatamart
    from ._multi import MultiTrees


class AGB:
    """Datamart helper for discovering and extracting AGB models.

    Reachable as ``ADMDatamart.agb``; not intended to be instantiated
    directly.
    """

    def __init__(self, datamart: ADMDatamart):
        self.datamart = datamart

    def discover_model_types(
        self,
        df: pl.LazyFrame,
        by: str = "Configuration",
    ) -> dict[str, str]:
        """Discover the type of model embedded in the ``Modeldata`` column.

        Groups by ``by`` (typically Configuration, since one model rule
        contains one model type) and decodes the first ``Modeldata`` blob
        per group to extract its ``_serialClass``.

        Parameters
        ----------
        df : pl.LazyFrame
            Datamart slice including ``Modeldata``.  Collected internally.
        by : str
            Grouping column.  ``Configuration`` is recommended.
        """
        if "Modeldata" not in df.collect_schema().names():
            raise ValueError(
                "Modeldata column not in the data. Please make sure to include it by setting 'subset' to False.",
            )

        def _get_type(val: str) -> str:
            return json.loads(zlib.decompress(base64.b64decode(val)))["_serialClass"]

        types = df.filter(pl.col("Modeldata").is_not_null()).group_by(by).agg(pl.col("Modeldata").first()).collect()
        return {row[by]: _get_type(row["Modeldata"]) for row in types.to_dicts()}

    def get_agb_models(
        self,
        last: bool = False,
        n_threads: int = 6,
        query: QUERY | None = None,
    ) -> dict[str, MultiTrees]:
        """Get all AGB models in the datamart, indexed by Configuration.

        Filters down to models whose ``_serialClass`` ends with
        ``GbModel`` and decodes them via :class:`MultiTrees`.

        Parameters
        ----------
        last : bool
            If True, use only the latest snapshot per model.
        n_threads : int
            Worker count for parallel blob decoding.
        query : QUERY | None
            Optional pre-filter applied before discovery.
        """
        from ._multi import MultiTrees

        df = self.datamart.aggregates.last(table="model_data") if last else self.datamart.model_data
        if df is None:
            raise ValueError(
                "Datamart has no model_data; cannot extract AGB models.",
            )
        df = cdh_utils._apply_query(df, query)
        if "Modeldata" not in df.collect_schema().names():
            raise ValueError(
                "Modeldata column not in the data. Please make sure to include it by setting 'subset' to False.",
            )
        types = self.discover_model_types(df, by="Configuration")
        agb_configs = [config for config, t in types.items() if t.endswith("GbModel")]
        logger.info("Found %d AGB configurations: %s", len(agb_configs), agb_configs)
        df_collected = df.filter(pl.col("Configuration").is_in(agb_configs)).collect()
        result: dict[str, MultiTrees] = {}
        for config in agb_configs:
            sub = df_collected.filter(pl.col("Configuration") == config)
            result[config] = MultiTrees.from_datamart(sub, n_threads=n_threads, configuration=config)
        return result
