from __future__ import annotations

__all__ = ["Explanations"]

import logging
from datetime import datetime, timedelta

from .Aggregate import Aggregate
from .FilterWidget import FilterWidget
from .Plots import Plots
from .Preprocess import Preprocess
from .Reports import Reports

logger = logging.getLogger(__name__)


class Explanations:
    """Process and explore explanation data for Adaptive Gradient Boost models.

    The class is a thin orchestrator over four sub-namespaces (``preprocess``,
    ``aggregate``, ``plot``, ``report``, ``filter``) that operate on the
    parquet files produced by Pega's explanation file repository.

    The constructor is **pure configuration** — it takes no filesystem paths
    and performs no I/O. Use the ``from_local_directory`` classmethod to load
    raw explanation parquet files from disk (or a remote URL) and run the
    DuckDB aggregation step. After that, ``aggregate``, ``plot`` and
    ``report`` can be used freely.

    Parameters
    ----------
    model_name : str, optional
        Name of the model rule. Used to identify and validate raw
        explanation parquet files when loading via
        :meth:`from_local_directory`.
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
    Explanations.from_local_directory : Load raw explanation parquet files
        from a local folder or remote URL and pre-aggregate them.

    Notes
    -----
    Environment variables that influence the (lazy) DuckDB aggregation step:

    ``MODEL_CONTEXT_LIMIT``
        Maximum number of unique contexts processed in a single query.
        Default: ``2500``.
    ``QUERY_BATCH_LIMIT``
        Number of contexts per DuckDB batch query. Default: ``10``.
    ``FILE_BATCH_LIMIT``
        Number of files per DuckDB batch. Default: ``10``.
    ``MEMORY_LIMIT``
        DuckDB buffer memory limit in GB. Default: ``8``.
    ``THREAD_COUNT``
        Number of DuckDB worker threads. Default: ``4``.
    ``PROGRESS_BAR``
        ``"1"`` to enable the DuckDB progress bar. Default: disabled.

    Examples
    --------
    Load and explore a folder of raw explanation parquet files:

    >>> from datetime import datetime
    >>> exp = Explanations.from_local_directory(
    ...     data_folder="explanations_data",
    ...     model_name="AdaptiveBoostCT",
    ...     from_date=datetime(2025, 3, 28),
    ...     to_date=datetime(2025, 3, 28),
    ... )
    >>> df = exp.aggregate.get_df_overall().collect()  # doctest: +SKIP

    Load a single remote parquet file:

    >>> exp = Explanations.from_local_directory(
    ...     data_file="https://example.com/AdaptiveBoostCT_20250328.parquet",
    ...     model_name="AdaptiveBoostCT",
    ... )  # doctest: +SKIP

    Construct without I/O (e.g. inside a Quarto report that already points
    ``aggregate.data_folderpath`` at pre-aggregated parquet files):

    >>> exp = Explanations()
    >>> exp.aggregate.data_folderpath = "/path/to/aggregated_data"  # doctest: +SKIP

    """

    # Default storage locations used when no path overrides are provided.
    # These are *internal* defaults — public path inputs go through
    # ``from_local_directory``.
    _DEFAULT_ROOT_DIR = ".tmp"
    _DEFAULT_DATA_FOLDER = "explanations_data"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ):
        self._init_state(
            root_dir=self._DEFAULT_ROOT_DIR,
            data_folder=self._DEFAULT_DATA_FOLDER,
            data_file=None,
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )

    @classmethod
    def from_local_directory(
        cls,
        root_dir: str = _DEFAULT_ROOT_DIR,
        data_folder: str = _DEFAULT_DATA_FOLDER,
        data_file: str | None = None,
        *,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> "Explanations":
        """Construct an ``Explanations`` from raw parquet files on disk or a URL.

        This is the standard entry point: it wires the path configuration,
        runs the DuckDB pre-aggregation step (writing the per-context and
        per-overall aggregates to ``<root_dir>/aggregated_data/``) and
        returns a ready-to-query instance.

        Parameters
        ----------
        root_dir : str, default ".tmp"
            Working directory under which the pre-aggregated parquet files
            (and report scratch space) are written.
        data_folder : str, default "explanations_data"
            Folder containing the raw model-explanation parquet files
            downloaded from the Pega explanation file repository. Used
            when ``data_file`` is not provided.
        data_file : str, optional
            Direct path or URL to a single explanation parquet file. When
            given, takes precedence over ``data_folder``. ``http://`` and
            ``https://`` URLs are downloaded into ``root_dir`` before
            aggregation.
        model_name : str, optional
            Name of the model rule. Used to filter files in ``data_folder``
            and validate that the correct files are being processed.
        from_date : datetime, optional
            Start date of the period over which aggregates are collected.
            See :class:`Explanations` for default behaviour.
        to_date : datetime, optional
            End date of the period over which aggregates are collected.
            See :class:`Explanations` for default behaviour.

        Returns
        -------
        Explanations
            A fully initialised instance with pre-aggregation completed.

        Raises
        ------
        ValueError
            If ``from_date > to_date``, if no files match ``model_name``
            within the date range, or if a remote ``data_file`` cannot be
            downloaded.

        """
        instance = cls.__new__(cls)
        instance._init_state(
            root_dir=root_dir,
            data_folder=data_folder,
            data_file=data_file,
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )
        instance.preprocess.generate()
        return instance

    def _init_state(
        self,
        *,
        root_dir: str,
        data_folder: str,
        data_file: str | None,
        model_name: str | None,
        from_date: datetime | None,
        to_date: datetime | None,
    ) -> None:
        """Set instance attributes and wire sub-namespaces. Pure (no I/O)."""
        self.root_dir = root_dir
        self.data_folder = data_folder
        self.data_file = data_file

        self.model_name = model_name
        self._set_date_range(from_date, to_date)

        self.preprocess = Preprocess(explanations=self)
        self.aggregate = Aggregate(explanations=self)
        self.plot = Plots(explanations=self)
        self.report = Reports(explanations=self)
        self.filter = FilterWidget(explanations=self)

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
