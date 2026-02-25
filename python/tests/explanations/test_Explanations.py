"""Testing the functionality of the Explanations class"""

import os
import shutil
from datetime import datetime, timedelta

import pytest
from pdstools.explanations import Explanations


@pytest.fixture(scope="module")
def dummy_explanations_data():
    """Fixture to create a dummy explanations data folder."""
    os.makedirs("explanations_data", exist_ok=True)
    file_path = "explanations_data/dummy_explanations.parquet"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("dummy data")
    yield file_path

    # Cleanup after test
    if os.path.exists("explanations_data"):
        shutil.rmtree("explanations_data")


@pytest.fixture(scope="module")
def dummy_aggregate_data():
    """Fixture to create a dummy aggregate data folder."""
    os.makedirs(".tmp/aggregated_data", exist_ok=True)
    file_path = ".tmp/aggregated_data/dummy_aggregate.parquet"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("dummy data")
    yield file_path

    # Cleanup after test
    if os.path.exists(".tmp"):
        shutil.rmtree(".tmp")


class TestExplanationsDateRange:
    """Test the initialization of the Explanations class"""

    def test_date_range_only_to_date(
        self,
        dummy_explanations_data,
        dummy_aggregate_data,
    ):
        """Test initialization with only to_date provided"""
        to_date = datetime(2023, 1, 8)
        explanations = Explanations(to_date=to_date)

        expected_from_date = to_date - timedelta(days=7)
        assert explanations.from_date == expected_from_date
        assert explanations.to_date == to_date

    def test_date_range_only_from_date(
        self,
        dummy_explanations_data,
        dummy_aggregate_data,
    ):
        """Test initialization with only from_date provided"""
        from_date = datetime(2023, 1, 1)
        explanations = Explanations(from_date=from_date)

        expected_to_date = datetime.today().date()
        assert explanations.from_date == from_date
        assert explanations.to_date.date() == expected_to_date

    def test_invalid_date_range(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that invalid date range raises ValueError"""
        from_date = datetime(2023, 1, 8)
        to_date = datetime(2023, 1, 1)  # to_date before from_date

        with pytest.raises(ValueError, match="from_date cannot be after to_date"):
            Explanations(from_date=from_date, to_date=to_date)

    def test_valid_date_range(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that valid date range is accepted"""
        from_date = datetime(2023, 1, 1)
        to_date = datetime(2023, 1, 8)

        explanations = Explanations(from_date=from_date, to_date=to_date)

        assert explanations.from_date == from_date
        assert explanations.to_date == to_date

    def test_same_from_and_to_date(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that same from_date and to_date is valid"""
        date = datetime(2023, 1, 1)

        explanations = Explanations(from_date=date, to_date=date)

        assert explanations.from_date == date
        assert explanations.to_date == date

    def test_default_date_range(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that default date range is set to a week before today"""
        explanations = Explanations()

        expected_to_date = datetime.today().date()
        expected_from_date = expected_to_date - timedelta(days=7)

        assert explanations.from_date.date() == expected_from_date
        assert explanations.to_date.date() == expected_to_date
