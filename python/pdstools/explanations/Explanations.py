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

    The constructor is **pure configuration** — it takes no filesystem paths
    and performs no I/O. Use the ``from_aggregates`` classmethod to point at
    pre-aggregated data (typically ``.tmp/aggregated_data/``).
    After initialization, ``aggregate``, ``plot`` and ``report`` can be used
    freely.

    Parameters
    ----------
    aggregated_data_dir : str | Path, default ".tmp/aggregated_data"
        Path to the folder containing pre-aggregated parquet files.
        Must exist and be non-empty; raises FileNotFoundError otherwise.
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

    Examples
    --------
    Load pre-aggregated explanation data:

    >>> from pathlib import Path
    >>> exp = Explanations.from_aggregates(
    ...     aggregated_data_dir=Path(".tmp/aggregated_data"),
    ...     model_name="AdaptiveBoostCT",
    ... )
    >>> df = exp.aggregate.get_df_overall().collect()  # doctest: +SKIP

    Construct with a custom aggregates path:

    >>> exp = Explanations(aggregated_data_dir="/path/to/my/aggregates")
    >>> df = exp.aggregate.get_df_overall().collect()  # doctest: +SKIP

    """

    # Default storage location for aggregated data.
    _DEFAULT_AGGREGATED_DATA_DIR = ".tmp/aggregated_data"

    def __init__(
        self,
        *,
        aggregated_data_dir: str | Path = _DEFAULT_AGGREGATED_DATA_DIR,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ):
        self._init_state(
            aggregated_data_dir=aggregated_data_dir,
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )

    @classmethod
    def from_aggregates(
        cls,
        aggregated_data_dir: str | Path = _DEFAULT_AGGREGATED_DATA_DIR,
        *,
        model_name: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> "Explanations":
        """Construct an ``Explanations`` from pre-aggregated parquet files.

        This is the standard entry point: it points to a folder containing
        pre-aggregated parquet files (typically produced by running
        :class:`Preprocess` separately) and returns a ready-to-query instance.

        Parameters
        ----------
        aggregated_data_dir : str | Path, default ".tmp/aggregated_data"
            Path to the folder containing pre-aggregated parquet files.
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
            If ``aggregated_data_dir`` does not exist or is empty.

        """
        instance = cls(
            aggregated_data_dir=aggregated_data_dir,
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )
        instance._validate_aggregated_data_dir()
        return instance

    def _init_state(
        self,
        *,
        aggregated_data_dir: str | Path,
        model_name: str | None,
        from_date: datetime | None,
        to_date: datetime | None,
    ) -> None:
        """Set instance attributes and wire sub-namespaces. Pure (no I/O)."""
        self.aggregated_data_dir = Path(aggregated_data_dir)
        # For backwards compatibility, expose root_dir. If using the default path,
        # compute it as the parent of aggregated_data's parent; otherwise use the path directly.
        if str(aggregated_data_dir) == self._DEFAULT_AGGREGATED_DATA_DIR:
            self.root_dir = ".tmp"
        else:
            self.root_dir = str(self.aggregated_data_dir.parent.parent)
        self.model_name = model_name
        self._set_date_range(from_date, to_date)
        self.aggregate = Aggregate(explanations=self)
        self.plot = Plots(explanations=self)
        self.report = Reports(explanations=self)

    def _validate_aggregated_data_dir(self) -> None:
        """Validate that aggregated_data_dir exists and contains parquet files.

        This is called lazily when data is first accessed, not during init.
        """
        if not self.aggregated_data_dir.exists():
            raise FileNotFoundError(
                f"Aggregated data directory not found: {self.aggregated_data_dir}. "
                "Please ensure that pre-aggregated data is available at the specified path"
            )

        if not any(self.aggregated_data_dir.glob("*.parquet")):
            raise FileNotFoundError(
                f"No parquet files found in {self.aggregated_data_dir}. "
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
