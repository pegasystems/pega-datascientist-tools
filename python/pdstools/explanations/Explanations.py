__all__ = ["Explanations"]

import logging
from datetime import datetime, timedelta
from typing import Optional

from .Aggregate import Aggregate
from .FilterWidget import FilterWidget
from .Plots import Plots
from .Preprocess import Preprocess
from .Reports import Reports

logger = logging.getLogger(__name__)


class Explanations:
    """
    Process and explore explanation data for Adaptive Gradient Boost models.

    Class is initialied with data location, which should point to the location of the
    model's explanation parquet files downloaded from the explanations file repository.
    These parquet files can then be processed to create aggregates to explain the contribution
    of different predictors on a global level.

    Parameters
    ----------
    data_folder: str
        The path of the folder containing the model explanation parquet files for processing.
    model_name : str, optional
        The name of the model rule. Will be used to identify files in the data folder
        and to validate that the correct files are being processed.
    end_date : datetime, optional, default = datetime.today()
        Defines the end date of the duration over which aggregates will be collected.
    start_date : datetime, optional, default = end_date - timedelta(7)
        Defines the start date of the duration over which aggregaates wille be collected.

    Environment variables
    -------------------
    BATCH_LIMIT: int
        The maximum number of unique contexts to process in a single batch. Default is 10.
    MEMORY_LIMIT: int
        Set the memory limit for the duckdb buffer manager.
        If not set will use 80% of RAM. Default is 2(in GB).
    THREAD_COUNT: int
        Set the amount of threads for duck db parallel query execution. Default is 4.
    PROGRESS_BAR: int
        Show progress bar when running duckdb queries.
        0 = no progress bar, 1 = show progress bar. Default is 0.
    """

    def __init__(
        self,
        root_dir: str = ".tmp",
        data_folder: str = "explanations_data",
        model_name: Optional[str] = "",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ):
        self.root_dir = root_dir
        self.data_folder = data_folder

        self.model_name = model_name
        self.from_date = from_date
        self.to_date = to_date
        self._set_date_range(from_date, to_date)

        self.preprocess = Preprocess(explanations=self)
        self.aggregate = Aggregate(explanations=self)
        self.plot = Plots(explanations=self)
        self.report = Reports(explanations=self)
        self.filter = FilterWidget(explanations=self)

    def _set_date_range(
        self, from_date: Optional[datetime], to_date: Optional[datetime], days: int = 7
    ):
        """Set the date range for processing explanation files.

        Parameters
        ----------
        start_date : datetime, optional
            The start date for the date range. If None, defaults to 7 days before end_date.
        end_date : datetime, optional
            The end date for the date range. If None, defaults to today.
        """
        if from_date is None and to_date is None:
            to_date = datetime.today()
            from_date = to_date - timedelta(days=days)

        # if only `to_date` is provided, set `from_date` to 7 days before `to_date`
        if from_date is None and to_date is not None:
            from_date = to_date - timedelta(days=days)

        # if only `from_date` is provided, set `to_date` to today
        # it can process from any date until today. eg: from_date = 2023-01-01, to_date = today
        if from_date is not None and to_date is None:
            to_date = datetime.today()

        # validate date range if both from_date and to_date are provided
        if from_date is not None and to_date is not None:
            if from_date > to_date:
                raise ValueError("from_date cannot be after to_date")
            # if (to_date - from_date).days > 30:
            #     raise ValueError("Date range cannot be more than 30 days")

        self.from_date = from_date
        self.to_date = to_date
