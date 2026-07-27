from __future__ import annotations

__all__ = ["Explanations"]

import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ..pega_io import scan_parquet_path
from .Aggregates import Aggregates
from .Plots import Plots
from .Reports import Reports
from .Schema import apply_schema

logger = logging.getLogger(__name__)


def _scan_aggregate(path: Path) -> pl.LazyFrame:
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
    data_folderpath : str | Path
        Resolved folder used for the context artifacts that the report
        pipeline reads and writes (``unique_contexts.json``, ``batches/``).
    root_dir : str, default ".tmp"
        Scratch directory for generated reports.
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
    ...     data_folder=Path(".tmp/aggregated_data"),
    ...     model_name="AdaptiveBoostCT",
    ...     from_date=datetime(2025, 3, 28),
    ...     to_date=datetime(2025, 3, 28),
    ... )
    >>> df = exp.overall.collect()  # doctest: +SKIP

    Construct with a custom aggregates path:

    >>> exp = Explanations.from_aggregates(data_folder="/path/to/my/aggregates")
    >>> df = exp.overall.collect()  # doctest: +SKIP

    """

    _DEFAULT_ROOT_DIR = ".tmp"
    # Default storage location for aggregated data.
    _DEFAULT_DATA_FOLDER = "aggregated_data"

    overall: pl.LazyFrame
    """Contributions aggregated across all contexts."""

    contextual: pl.LazyFrame
    """Contributions aggregated per context."""

    def __init__(
        self,
        overall: pl.LazyFrame,
        contextual: pl.LazyFrame,
        *,
        data_folderpath: str | Path,
        root_dir: str = _DEFAULT_ROOT_DIR,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ):
        self.overall = overall
        self.contextual = contextual
        self.data_folderpath = Path(data_folderpath)
        self.root_dir = root_dir

        self.model_name = model_name
        self._set_date_range(from_date, to_date)
        self.aggregates = Aggregates(explanations=self)
        self.plot = Plots(explanations=self)
        self.report = Reports(explanations=self)

    @staticmethod
    def _resolve_data_folder(root_dir: str | None, data_folder: str | Path) -> Path:
        """Resolve ``data_folder`` to an absolute path, once.

        ``root_dir`` is the scratch directory for generated reports. It doubles
        as the base for ``data_folder`` only when the caller opted into it by
        passing it explicitly, or when ``data_folder`` is left at its default.
        A relative ``data_folder`` on its own is therefore relative to the
        working directory, which is what a path like ``"../../data/aggregates"``
        is expected to mean.
        """
        use_root = root_dir is not None or str(data_folder) == Explanations._DEFAULT_DATA_FOLDER
        base = Path(root_dir or Explanations._DEFAULT_ROOT_DIR) if use_root else Path()
        return (base / data_folder).resolve()

    @classmethod
    def from_aggregates(
        cls,
        *,
        root_dir: str | None = None,
        data_folder: str | Path = _DEFAULT_DATA_FOLDER,
        contextual_file: str = "BY_CONTEXT.parquet",
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> Explanations:
        """Construct an ``Explanations`` from pre-aggregated parquet files.

        This is the standard entry point: it points to a folder containing
        pre-aggregated parquet files and returns a ready-to-query instance.

        Parameters
        ----------
        root_dir : str, optional
            Scratch directory for generated reports, default ``".tmp"``. When
            given explicitly it is also used as the base for a relative
            ``data_folder``.
        data_folder : str | Path, default "aggregated_data"
            Path to the folder containing pre-aggregated parquet files. Absolute
            paths are used as-is. A relative path is resolved against
            ``root_dir`` when that was passed explicitly, and against the current
            working directory otherwise.
        contextual_file : str, default "BY_CONTEXT.parquet"
            Name of the per-context parquet file inside ``data_folder``. The
            report pipeline uses this to point a page at a single pre-computed
            batch (e.g. ``"batches/BATCH_3.parquet"``) instead of the full set.
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
            If ``data_folder`` does not contain the expected parquet files.
        """
        folder = cls._resolve_data_folder(root_dir, data_folder)
        return cls(
            overall=_scan_aggregate(folder / "OVERVIEW.parquet"),
            contextual=_scan_aggregate(folder / contextual_file),
            data_folderpath=folder,
            root_dir=root_dir or cls._DEFAULT_ROOT_DIR,
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )

    def _set_date_range(
        self,
        from_date: datetime | None,
        to_date: datetime | None,
        days: int = 7,
    ) -> None:
        """Resolve the ``(from_date, to_date)`` window using the default rules.

        Parameters
        ----------
        from_date : datetime or None
            Start of the date range. If ``None`` and ``to_date`` is given,
            defaults to ``to_date - days``.
        to_date : datetime or None
            End of the date range. If ``None`` and ``from_date`` is given,
            defaults to ``datetime.today()``.
        days : int, default 7
            Window length used to fill in the missing endpoint when only one
            of ``from_date`` / ``to_date`` is provided.

        Raises
        ------
        ValueError
            If both endpoints are provided and ``from_date > to_date``.

        """
        if from_date is None and to_date is None:
            to_date = datetime.today()
            from_date = to_date - timedelta(days=days)

        if from_date is None and to_date is not None:
            from_date = to_date - timedelta(days=days)

        if from_date is not None and to_date is None:
            to_date = datetime.today()

        if from_date is not None and to_date is not None:
            if from_date > to_date:
                raise ValueError("from_date cannot be after to_date")

        self.from_date = from_date
        self.to_date = to_date
