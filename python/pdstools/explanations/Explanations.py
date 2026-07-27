from __future__ import annotations

__all__ = ["Explanations"]

import logging
from datetime import datetime, timedelta
from pathlib import Path

from .Aggregate import Aggregate
from .Plots import Plots
from .Reports import Reports

logger = logging.getLogger(__name__)


class Explanations:
    """Process and explore explanation data for Adaptive Gradient Boost models.

    The class is a thin orchestrator over three sub-namespaces (``aggregate``,
    ``plot``, ``report``) that operate on pre-aggregated parquet files.

    The constructor is **pure configuration** — it accepts path settings but
    performs no I/O. The parquet files are read lazily on first access.

    Parameters
    ----------
    root_dir : str, optional
        Scratch directory for generated reports, default ``".tmp"``. When given
        explicitly it is also used as the base for a relative ``data_folder``.
    data_folder : str | Path, optional, default "aggregated_data"
        Path to the folder containing pre-aggregated parquet files. Absolute
        paths are used as-is. A relative path is resolved against ``root_dir``
        when that was passed explicitly, and against the current working
        directory otherwise, so ``"../../data/aggregates"`` means what it says.
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
    >>> df = exp.aggregate.overall.collect()  # doctest: +SKIP

    Construct with a custom aggregates path:

    >>> exp = Explanations(data_folder="/path/to/my/aggregates")
    >>> df = exp.aggregate.overall.collect()  # doctest: +SKIP

    """

    _DEFAULT_ROOT_DIR = ".tmp"
    # Default storage location for aggregated data.
    _DEFAULT_DATA_FOLDER = "aggregated_data"

    def __init__(
        self,
        *,
        root_dir: str | None = None,
        data_folder: str | Path = _DEFAULT_DATA_FOLDER,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ):
        # `root_dir` is the scratch directory for generated reports. It doubles as
        # the base for `data_folder` only when the caller opted into it by passing
        # it explicitly, or when `data_folder` is left at its default. A relative
        # `data_folder` on its own is therefore relative to the working directory,
        # which is what a path like "../../data/aggregates" is expected to mean.
        self.root_dir = self._DEFAULT_ROOT_DIR if root_dir is None else root_dir
        self.data_folder = data_folder
        base = Path(self.root_dir) if root_dir is not None or str(data_folder) == self._DEFAULT_DATA_FOLDER else Path()
        self.data_folderpath = (base / data_folder).resolve()

        self.model_name = model_name
        self._set_date_range(from_date, to_date)
        self.aggregate = Aggregate(explanations=self)
        self.plot = Plots(explanations=self)
        self.report = Reports(explanations=self)

    @classmethod
    def from_aggregates(
        cls,
        *,
        root_dir: str | None = None,
        data_folder: str | Path = _DEFAULT_DATA_FOLDER,
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
            A fully initialised instance pointing at the aggregated data.

        Notes
        -----
        The parquet files are read lazily on first access, so a missing or empty
        ``data_folder`` surfaces as a ``FileNotFoundError`` from the first
        operation that touches the data rather than from this constructor.
        """
        return cls(
            root_dir=root_dir,
            data_folder=data_folder,
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
