"""The :class:`MultiTrees` collection — multiple snapshots of one config."""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from ._model import ADMTreesModel

logger = logging.getLogger(__name__)


@dataclass
class MultiTrees:
    """A collection of :class:`ADMTreesModel` snapshots indexed by timestamp.

    Construct via :meth:`from_datamart`.
    """

    trees: dict[str, ADMTreesModel]
    model_name: str | None = None
    context_keys: list | None = None

    def __repr__(self) -> str:
        mod = "" if self.model_name is None else f" for {self.model_name}"
        keys = list(self.trees.keys())
        if not keys:  # pragma: no cover
            return f"MultiTrees object{mod}, empty"
        return f"MultiTrees object{mod}, with {len(self)} trees ranging from {keys[0]} to {keys[-1]}"

    def __getitem__(self, index: int | str) -> ADMTreesModel:
        """Return the :class:`ADMTreesModel` at ``index``.

        Integer indices select by insertion order; string indices select
        by snapshot timestamp.  Use :meth:`items` if you need both keys
        and values together.
        """
        if isinstance(index, int):
            return list(self.trees.values())[index]
        return self.trees[index]

    def __len__(self) -> int:
        return len(self.trees)

    def items(self):
        """Iterate ``(timestamp, model)`` pairs in insertion order."""
        return self.trees.items()

    def values(self):
        """Iterate :class:`ADMTreesModel` instances in insertion order."""
        return self.trees.values()

    def keys(self):
        """Iterate snapshot timestamps in insertion order."""
        return self.trees.keys()

    def __iter__(self):
        return iter(self.trees)

    def __add__(self, other: MultiTrees | ADMTreesModel) -> MultiTrees:
        if isinstance(other, MultiTrees):
            return MultiTrees(
                trees={**self.trees, **other.trees},
                model_name=self.model_name or other.model_name,
                context_keys=self.context_keys or other.context_keys,
            )
        if isinstance(other, ADMTreesModel):
            timestamp = other.metrics.get("factory_update_time")
            if not timestamp:
                raise ValueError(
                    "Cannot add ADMTreesModel to MultiTrees: model has no "
                    "'factory_update_time' to use as a snapshot key. Add it "
                    "to a fresh MultiTrees with an explicit timestamp key.",
                )
            return MultiTrees(
                trees={**self.trees, str(timestamp): other},
                model_name=self.model_name,
                context_keys=self.context_keys,
            )
        return NotImplemented  # pragma: no cover

    @property
    def first(self) -> ADMTreesModel:
        return self[0]

    @property
    def last(self) -> ADMTreesModel:
        return self[-1]

    @classmethod
    def from_datamart(
        cls,
        df: pl.DataFrame,
        n_threads: int = 1,
        configuration: str | None = None,
    ) -> MultiTrees:
        """Decode every Modeldata blob in ``df`` for a single configuration.

        Returns one :class:`MultiTrees` containing one
        :class:`ADMTreesModel` per snapshot.

        Parameters
        ----------
        df : pl.DataFrame
            Datamart slice.  Must contain ``Modeldata``, ``SnapshotTime``
            and ``Configuration`` columns and cover exactly one
            Configuration.  Use :meth:`from_datamart_grouped` if ``df``
            spans multiple configurations.
        n_threads : int
            Worker count for parallel base64+zlib decoding.
        configuration : str | None
            Optional explicit Configuration name; required if ``df``
            doesn't already contain a single Configuration.
        """
        decoded = cls._decode_datamart_frame(df, n_threads=n_threads)
        configs = {cfg for cfg, _, _ in decoded}
        if configuration is not None:
            decoded = [(cfg, ts, mdl) for (cfg, ts, mdl) in decoded if cfg == configuration]
            chosen = configuration
        elif len(configs) == 1:
            chosen = next(iter(configs))
        else:
            raise ValueError(
                f"from_datamart received {len(configs)} configurations "
                f"({sorted(configs)!r}); pass `configuration=` to pick one, "
                "or call from_datamart_grouped to get one MultiTrees per config.",
            )
        trees = {ts: mdl for _, ts, mdl in decoded}
        return cls(trees=trees, model_name=chosen)

    @classmethod
    def from_datamart_grouped(
        cls,
        df: pl.DataFrame,
        n_threads: int = 1,
    ) -> dict[str, MultiTrees]:
        """Decode every Modeldata blob in ``df``, grouped by Configuration.

        Returns a mapping of configuration name to :class:`MultiTrees`.
        Use :meth:`from_datamart` instead when the input has only one
        configuration.
        """
        decoded = cls._decode_datamart_frame(df, n_threads=n_threads)
        per_config: dict[str, dict[str, ADMTreesModel]] = {}
        for cfg, ts, mdl in decoded:
            per_config.setdefault(cfg, {})[ts] = mdl
        return {cfg: cls(trees=trees, model_name=cfg) for cfg, trees in per_config.items()}

    @staticmethod
    def _decode_datamart_frame(
        df: pl.DataFrame,
        n_threads: int = 1,
    ) -> list[tuple[str, str, ADMTreesModel]]:
        """Decode every blob in ``df`` and return ``(config, timestamp, model)`` rows."""
        df = df.filter(pl.col("Modeldata").is_not_null()).select(
            # Format SnapshotTime explicitly — strip_chars_end strips a *set*
            # of characters, not a literal suffix, so it would mangle
            # timestamps ending in 0 (e.g. 12:30:20 → 12:30:2).
            pl.col("SnapshotTime").dt.round("1s").dt.strftime("%Y-%m-%d %H:%M:%S"),
            pl.col("Modeldata").str.decode("base64"),
            pl.col("Configuration").cast(pl.Utf8),
        )
        if len(df) > 50 and n_threads == 1:
            logger.info(
                "Decoding %d models; setting n_threads higher may speed this up.",
                len(df),
            )
        configs = df["Configuration"].to_list()
        timestamps = df["SnapshotTime"].to_list()
        blobs = df["Modeldata"].to_list()
        try:
            from tqdm import tqdm

            iterable: Any = tqdm(blobs)
        except ImportError:  # pragma: no cover
            iterable = blobs
        with multiprocessing.Pool(n_threads) as p:
            decoder = map if n_threads < 2 else p.imap
            models = list(decoder(ADMTreesModel.from_datamart_blob, iterable))
        return list(zip(configs, timestamps, models, strict=True))

    def compute_over_time(
        self,
        predictor_categorization: Callable | None = None,
    ) -> pl.DataFrame:
        """Return per-tree categorisation counts across snapshots, with a
        ``SnapshotTime`` column per row.
        """
        outdf = []
        for timestamp, tree in self.trees.items():
            if predictor_categorization is not None:
                to_plot = tree.compute_categorization_over_time(
                    predictor_categorization,
                )[0]
            else:
                to_plot = tree.splits_per_variable_type[0]
            outdf.append(
                pl.DataFrame(to_plot).with_columns(
                    SnapshotTime=pl.lit(timestamp).str.to_datetime(format="%Y-%m-%d %H:%M:%S").dt.date(),
                ),
            )
        return pl.concat(outdf, how="diagonal")
