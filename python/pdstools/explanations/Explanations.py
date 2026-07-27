from __future__ import annotations

__all__ = ["Explanations"]

import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ..pega_io import is_url, scan_parquet_path
from .Aggregates import Aggregates
from .Plots import Plots
from .Reports import Reports
from .Schema import apply_schema

logger = logging.getLogger(__name__)

_DEFAULT_DATE_RANGE_DAYS = 7


def _join(base_path: str | Path, filename: str | Path) -> str | Path:
    """Resolve ``filename`` against ``base_path``, URL-aware.

    A full path in ``filename`` wins and ``base_path`` is ignored, matching
    :meth:`ADMDatamart.from_ds_export`. ``pathlib`` cannot be used for URLs, so
    those are joined as strings.
    """
    if is_url(filename):
        return filename
    if is_url(base_path):
        return f"{str(base_path).rstrip('/')}/{filename}"
    return Path(base_path).resolve() / filename


def _scan_aggregate(path: str | Path) -> pl.LazyFrame:
    """Scan one aggregated parquet file and normalise it for downstream use."""
    return apply_schema(scan_parquet_path(path)).filter(pl.col("contribution") != 0.0).sort(by="predictor_name")


class Explanations:
    """Process and explore explanation data for Adaptive Gradient Boost models.

    The class is a thin orchestrator over three sub-namespaces (``aggregates``,
    ``plot``, ``report``) that operate on pre-aggregated parquet files.

    The constructor is **pure configuration** — it accepts already-scanned
    :class:`polars.LazyFrame`s and path settings, and performs no I/O. Use
    :meth:`from_aggregates` to read the parquet files from disk.

    Parameters
    ----------
    overall : pl.LazyFrame
        Contributions aggregated across all contexts.
    contextual : pl.LazyFrame
        Contributions aggregated per context.
    model_name : str, optional
        Name of the model rule. Used for report metadata only.
    from_date : datetime, optional
        Start date of the period over which aggregates are computed.
        Defaults to ``to_date - 7 days`` if only ``to_date`` is given,
        or to ``today() - 7 days`` if both are omitted.
    to_date : datetime, optional
        End date of the period over which aggregates are computed.
        Defaults to ``today()`` if only ``from_date`` is given, or to
        ``today()`` if both are omitted.

    See Also
    --------
    Explanations.from_aggregates : Load pre-aggregated parquet files.

    Notes
    -----
    Environment variables that influence the batch parquet file generation:

    ``PDSTOOLS_FILE_BATCH_LIMIT``
        Number of context partitions per batch. Default: ``100``.

    Examples
    --------
    Load pre-aggregated explanation data:

    >>> from pathlib import Path
    >>> exp = Explanations.from_aggregates(
    ...     base_path=Path(".tmp/aggregated_data"),
    ...     model_name="AdaptiveBoostCT",
    ...     from_date=datetime(2025, 3, 28),
    ...     to_date=datetime(2025, 3, 28),
    ... )
    >>> df = exp.overall.collect()  # doctest: +SKIP

    Construct with a custom aggregates path:

    >>> exp = Explanations.from_aggregates(base_path="/path/to/my/aggregates")
    >>> df = exp.overall.collect()  # doctest: +SKIP

    The aggregates may also live behind a URL:

    >>> exp = Explanations.from_aggregates(base_path="https://example.com/aggregates")
    >>> df = exp.overall.collect()  # doctest: +SKIP

    """

    overall: pl.LazyFrame
    """Contributions aggregated across all contexts."""

    contextual: pl.LazyFrame
    """Contributions aggregated per context."""

    def __init__(
        self,
        overall: pl.LazyFrame,
        contextual: pl.LazyFrame,
        *,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ):
        self.overall = overall
        self.contextual = contextual
        self.model_name = model_name
        self.from_date, self.to_date = self._resolve_date_range(from_date, to_date)
        self.aggregates = Aggregates(explanations=self)
        self.plot = Plots(explanations=self)
        self.report = Reports(explanations=self)

    @classmethod
    def from_aggregates(
        cls,
        overall_filename: str | Path = "OVERVIEW.parquet",
        contextual_filename: str | Path = "BY_CONTEXT.parquet",
        base_path: str | Path = ".",
        *,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> Explanations:
        """Construct an ``Explanations`` from pre-aggregated parquet files.

        This is the standard entry point: it points to a folder containing
        pre-aggregated parquet files and returns a ready-to-query instance.

        Parameters
        ----------
        overall_filename : str | Path, default "OVERVIEW.parquet"
            Contributions aggregated across all contexts, relative to
            ``base_path``. A full path is used as-is, ignoring ``base_path``.
        contextual_filename : str | Path, default "BY_CONTEXT.parquet"
            Contributions aggregated per context, relative to ``base_path``. A
            full path is used as-is, ignoring ``base_path``. The report pipeline
            uses this to point a page at a single pre-computed batch (e.g.
            ``"batches/BATCH_3.parquet"``) instead of the full set.
        base_path : str | Path, default "."
            Folder containing the pre-aggregated parquet files. May be an
            ``http(s)://`` URL.
        model_name : str, optional
            Name of the model rule. Used for report metadata only.
        from_date : datetime, optional
            Start date of the period over which aggregates are computed.
            See :class:`Explanations` for default behaviour.
        to_date : datetime, optional
            End date of the period over which aggregates are computed.
            See :class:`Explanations` for default behaviour.

        Returns
        -------
        Explanations
            A fully initialised instance holding LazyFrames over the
            aggregated data.

        Raises
        ------
        FileNotFoundError
            If either aggregate file does not exist.
        """
        return cls(
            overall=_scan_aggregate(_join(base_path, overall_filename)),
            contextual=_scan_aggregate(_join(base_path, contextual_filename)),
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )

    def save_data(self, path: str | Path = ".") -> tuple[Path, Path]:
        """Cache ``overall`` and ``contextual`` to parquet files.

        Mirrors :meth:`ADMDatamart.save_data`. The report pipeline uses this to
        materialise the frames into its working directory, so that the Quarto
        subprocess reads local files regardless of where the data came from -
        a URL, a database, or frames built in memory.

        Parameters
        ----------
        path : str | Path, default "."
            Directory to write into. Created if it does not exist.

        Returns
        -------
        tuple[Path, Path]
            Paths of the written overall and contextual parquet files.
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        overall_file = folder / "OVERVIEW.parquet"
        contextual_file = folder / "BY_CONTEXT.parquet"
        self.overall.sink_parquet(overall_file)
        self.contextual.sink_parquet(contextual_file)
        return overall_file, contextual_file

    @staticmethod
    def _resolve_date_range(
        from_date: datetime | None,
        to_date: datetime | None,
    ) -> tuple[datetime, datetime]:
        """Fill in either missing endpoint of the reporting window.

        ``to_date`` defaults to today, and ``from_date`` to
        ``to_date - _DEFAULT_DATE_RANGE_DAYS``.

        Raises
        ------
        ValueError
            If ``from_date`` is after ``to_date``.
        """
        to_date = to_date or datetime.today()
        from_date = from_date or to_date - timedelta(days=_DEFAULT_DATE_RANGE_DAYS)
        if from_date > to_date:
            raise ValueError(f"from_date ({from_date}) cannot be after to_date ({to_date})")
        return from_date, to_date
