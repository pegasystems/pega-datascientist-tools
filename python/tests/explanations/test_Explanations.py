import pytest
from datetime import datetime, timedelta
from pdstools.explanations import Explanations

"""
Testing the functionality of the Explanations class
"""


class TestExplanationsInit:
    """Test the initialization of the Explanations class"""

    def test_default_initialization(self):
        """Test initialization with default parameters"""
        explanations = Explanations()

        # Test default values
        assert explanations.root_dir == ".tmp"
        assert explanations.data_folder == "explanations_data"
        assert explanations.aggregates_folder == "aggregated_data"
        assert explanations.report_folder == "reports"
        assert explanations.model_name == ""
        assert explanations.progress_bar is False
        assert explanations.batch_limit == 10
        assert explanations.memory_limit == 2
        assert explanations.thread_count == 4

        # Test that date range is set (should be today and 7 days ago)
        assert explanations.from_date is not None
        assert explanations.to_date is not None
        assert (explanations.to_date - explanations.from_date).days == 7

    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        custom_from_date = datetime(2023, 1, 1)
        custom_to_date = datetime(2023, 1, 8)

        explanations = Explanations(
            root_dir="/custom/root",
            data_folder="custom_data",
            aggregates_folder="custom_aggregates",
            report_folder="custom_reports",
            model_name="test_model",
            from_date=custom_from_date,
            to_date=custom_to_date,
            progress_bar=True,
            batch_limit=20,
            memory_limit=4,
            thread_count=8,
        )

        # Test custom values
        assert explanations.root_dir == "/custom/root"
        assert explanations.data_folder == "custom_data"
        assert explanations.aggregates_folder == "custom_aggregates"
        assert explanations.report_folder == "custom_reports"
        assert explanations.model_name == "test_model"
        assert explanations.from_date == custom_from_date
        assert explanations.to_date == custom_to_date
        assert explanations.progress_bar is True
        assert explanations.batch_limit == 20
        assert explanations.memory_limit == 4
        assert explanations.thread_count == 8

    def test_date_range_only_to_date(self):
        """Test initialization with only to_date provided"""
        to_date = datetime(2023, 1, 8)
        explanations = Explanations(to_date=to_date)

        expected_from_date = to_date - timedelta(days=7)
        assert explanations.from_date == expected_from_date
        assert explanations.to_date == to_date

    def test_date_range_only_from_date(self):
        """Test initialization with only from_date provided"""
        from_date = datetime(2023, 1, 1)
        explanations = Explanations(from_date=from_date)

        expected_to_date = datetime.today().date()
        assert explanations.from_date == from_date
        assert explanations.to_date.date() == expected_to_date

    def test_invalid_date_range(self):
        """Test that invalid date range raises ValueError"""
        from_date = datetime(2023, 1, 8)
        to_date = datetime(2023, 1, 1)  # to_date before from_date

        with pytest.raises(ValueError, match="from_date cannot be after to_date"):
            Explanations(from_date=from_date, to_date=to_date)

    def test_valid_date_range(self):
        """Test that valid date range is accepted"""
        from_date = datetime(2023, 1, 1)
        to_date = datetime(2023, 1, 8)

        explanations = Explanations(from_date=from_date, to_date=to_date)

        assert explanations.from_date == from_date
        assert explanations.to_date == to_date

    def test_same_from_and_to_date(self):
        """Test that same from_date and to_date is valid"""
        date = datetime(2023, 1, 1)

        explanations = Explanations(from_date=date, to_date=date)

        assert explanations.from_date == date
        assert explanations.to_date == date
