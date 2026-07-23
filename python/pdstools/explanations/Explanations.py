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

    The constructor is **pure configuration** — it accepts optional filesystem
    path settings (``root_dir`` / ``data_folder``) but performs no I/O.
    Use the ``from_aggregates`` classmethod to point at pre-aggregated data
    (typically ``.tmp/aggregated_data/``) with validation.
    After initialization, ``aggregate``, ``plot`` and ``report`` can be used
    freely.

    Parameters
    ----------
    root_dir : str, optional, default ".tmp"
        Working directory under which the pre-aggregated parquet files
        (and report scratch space) are written. Ignored if a custom ``data_folder`` is provided,
        in which case the parent of ``data_folder`` is used as the root.
    data_folder : str | Path, optional, default "aggregated_data"
        Path to the folder containing pre-aggregated parquet files.
        Can be a relative path (combined with ``root_dir``) or an absolute path.
        Note: validation (FileNotFoundError) only occurs via ``from_aggregates``;
        direct ``__init__`` calls skip validation.
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
    --------
    Environment variable that influence the batch parquet file generation:

    ``FILE_BATCH_LIMIT``
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
    >>> df = exp.aggregate.get_df_overall().collect()  # doctest: +SKIP

    Construct with a custom aggregates path:

    >>> exp = Explanations(data_folder="/path/to/my/aggregates")
    >>> df = exp.aggregate.get_df_overall().collect()  # doctest: +SKIP

    """

    _DEFAULT_ROOT_DIR = ".tmp"
    # Default storage location for aggregated data.
    _DEFAULT_DATA_FOLDER = "aggregated_data"

    def __init__(
        self,
        *,
        root_dir: str = _DEFAULT_ROOT_DIR,
        data_folder: str | Path = _DEFAULT_DATA_FOLDER,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ):
        self._init_state(
            root_dir=root_dir,
            data_folder=data_folder,
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )

    @classmethod
    def from_aggregates(
        cls,
        *,
        root_dir: str = _DEFAULT_ROOT_DIR,
        data_folder: str | Path = _DEFAULT_DATA_FOLDER,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> "Explanations":
        """Construct an ``Explanations`` from pre-aggregated parquet files.

        This is the standard entry point: it points to a folder containing
        pre-aggregated parquet files and returns a ready-to-query instance.

        Parameters
        ----------
        root_dir : str, default ".tmp"
            Working directory under which the pre-aggregated parquet files
            (and report scratch space) are written.
        data_folder : str | Path, default "aggregated_data"
            Path to the folder containing pre-aggregated parquet files.
            Can be a relative path (combined with ``root_dir``) or an absolute path.
            Must exist and be non-empty; raises FileNotFoundError otherwise.
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

        Raises
        ------
        FileNotFoundError
            If ``root_dir`` or ``data_folder`` does not exist or is empty.

        """
        instance = cls.__new__(cls)
        instance._init_state(
            root_dir=root_dir,
            data_folder=data_folder,
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )
        instance.validate_data_folder()
        return instance

    def _init_state(
        self,
        *,
        root_dir: str,
        data_folder: str | Path,
        model_name: str | None,
        from_date: datetime | None,
        to_date: datetime | None,
    ) -> None:
        """Set instance attributes and wire sub-namespaces. Pure (no I/O)."""

        self.root_dir = root_dir
        self.data_folder = str(data_folder) if isinstance(data_folder, Path) else data_folder

        data_folder_path = Path(self.data_folder)
        root_is_default = root_dir == self._DEFAULT_ROOT_DIR
        should_split_data_folder = self.data_folder != self._DEFAULT_DATA_FOLDER and (
            data_folder_path.is_absolute() or (root_is_default and data_folder_path.parent != Path("."))
        )

        # Treat absolute paths and default-root multi-part relative paths as full
        # aggregate paths. When callers pass root_dir explicitly, a relative
        # data_folder remains relative to that root.
        if should_split_data_folder:
            self.data_folder = str(data_folder_path.name)
            self.root_dir = str(data_folder_path.parent)
            logger.info(
                "Using custom data folder: %s, setting root_dir to %s, data_folder to %s",
                data_folder_path,
                self.root_dir,
                self.data_folder,
            )
        self.model_name = model_name
        self._set_date_range(from_date, to_date)
        self.aggregate = Aggregate(explanations=self)
        self.plot = Plots(explanations=self)
        self.report = Reports(explanations=self)

    def validate_data_folder(self) -> None:
        """Validate that data_folder exists and contains parquet files."""
        agg_folder = Path(self.root_dir) / self.data_folder

        if not agg_folder.exists() or not agg_folder.is_dir():
            raise FileNotFoundError(
                f"Aggregated data directory not found: {agg_folder}. "
                "Please ensure that pre-aggregated data is available at the specified path"
            )

        if not any(agg_folder.glob("*.parquet")):
            raise FileNotFoundError(
                f"No parquet files found in {agg_folder}. "
                "Please ensure that pre-aggregated data is available at the specified path"
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
