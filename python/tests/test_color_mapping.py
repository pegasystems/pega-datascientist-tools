import polars as pl
import pytest
from pdstools.utils.color_mapping import create_categorical_color_mappings


@pytest.fixture
def sample_colorway():
    """Sample colorway with 3 colors."""
    return ["#FF0000", "#00FF00", "#0000FF"]


@pytest.fixture
def sample_data():
    """Sample data with multiple categorical columns."""
    return pl.LazyFrame(
        {
            "Category": ["A", "B", "C"],
            "Type": ["X", "Y", "X"],
            "Status": ["Active", "Inactive", "Active"],
        }
    )


def test_basic_color_mapping(sample_data, sample_colorway):
    """Test basic color mapping creation."""
    mappings = create_categorical_color_mappings(sample_data, ["Category"], sample_colorway)

    assert "Category" in mappings
    assert isinstance(mappings["Category"], dict)
    assert "A" in mappings["Category"]
    assert "B" in mappings["Category"]
    assert "C" in mappings["Category"]

    # Check colors are from colorway
    for color in mappings["Category"].values():
        assert color in sample_colorway


def test_multiple_columns(sample_data, sample_colorway):
    """Test mapping multiple columns at once."""
    mappings = create_categorical_color_mappings(sample_data, ["Category", "Type", "Status"], sample_colorway)

    assert len(mappings) == 3
    assert "Category" in mappings
    assert "Type" in mappings
    assert "Status" in mappings


def test_alphabetical_sorting(sample_data, sample_colorway):
    """Test that values are sorted alphabetically before color assignment."""
    mappings = create_categorical_color_mappings(sample_data, ["Category"], sample_colorway)

    # With values A, B, C sorted alphabetically, they should get colors in order
    assert mappings["Category"]["A"] == sample_colorway[0]
    assert mappings["Category"]["B"] == sample_colorway[1]
    assert mappings["Category"]["C"] == sample_colorway[2]


def test_modulo_wrapping():
    """Test that color assignment wraps around using modulo."""
    data = pl.LazyFrame({"Category": ["A", "B", "C", "D", "E"]})
    colorway = ["#FF0000", "#00FF00"]  # Only 2 colors

    mappings = create_categorical_color_mappings(data, ["Category"], colorway)

    # Should wrap: A=#FF0000, B=#00FF00, C=#FF0000, D=#00FF00, E=#FF0000
    assert mappings["Category"]["A"] == "#FF0000"
    assert mappings["Category"]["B"] == "#00FF00"
    assert mappings["Category"]["C"] == "#FF0000"
    assert mappings["Category"]["D"] == "#00FF00"
    assert mappings["Category"]["E"] == "#FF0000"


def test_handles_missing_columns(sample_data, sample_colorway):
    """Test that missing columns are skipped gracefully."""
    mappings = create_categorical_color_mappings(sample_data, ["Category", "NonExistent"], sample_colorway)

    assert "Category" in mappings
    assert "NonExistent" not in mappings


def test_column_mapping():
    """Test display name to internal name mapping."""
    data = pl.LazyFrame(
        {
            "pyIssue": ["Sales", "Retention"],
            "pyGroup": ["Cards", "Loans"],
        }
    )
    colorway = ["#FF0000", "#00FF00"]

    mappings = create_categorical_color_mappings(
        data, ["Issue", "Group"], colorway, column_mapping={"Issue": "pyIssue", "Group": "pyGroup"}
    )

    assert "Issue" in mappings
    assert "Group" in mappings
    assert "Sales" in mappings["Issue"]
    assert "Cards" in mappings["Group"]


def test_handles_nulls():
    """Test that null values are dropped."""
    data = pl.LazyFrame({"Category": ["A", None, "B", None, "C"]})
    colorway = ["#FF0000", "#00FF00", "#0000FF"]

    mappings = create_categorical_color_mappings(data, ["Category"], colorway)

    # Should only have A, B, C (nulls dropped)
    assert len(mappings["Category"]) == 3
    assert "A" in mappings["Category"]
    assert "B" in mappings["Category"]
    assert "C" in mappings["Category"]
    assert None not in mappings["Category"]
    assert "None" not in mappings["Category"]


def test_empty_column():
    """Test behavior with empty columns."""
    data = pl.LazyFrame({"Category": pl.Series([], dtype=pl.Utf8)})
    colorway = ["#FF0000"]

    mappings = create_categorical_color_mappings(data, ["Category"], colorway)

    assert "Category" in mappings
    assert len(mappings["Category"]) == 0


def test_many_categories_wrapping():
    """Test that many categories wrap colors correctly."""
    # Create data with 50 categories
    categories = [f"Cat_{i:02d}" for i in range(50)]
    data = pl.LazyFrame({"Category": categories})
    colorway = ["#FF0000", "#00FF00", "#0000FF"]  # Only 3 colors

    mappings = create_categorical_color_mappings(data, ["Category"], colorway)

    # Should have all 50 categories
    assert len(mappings["Category"]) == 50

    # Check modulo wrapping
    assert mappings["Category"]["Cat_00"] == colorway[0]
    assert mappings["Category"]["Cat_01"] == colorway[1]
    assert mappings["Category"]["Cat_02"] == colorway[2]
    assert mappings["Category"]["Cat_03"] == colorway[0]  # Wraps
    assert mappings["Category"]["Cat_04"] == colorway[1]
    assert mappings["Category"]["Cat_49"] == colorway[1]  # 49 % 3 = 1


def test_no_columns_provided():
    """Test behavior when no columns are provided."""
    data = pl.LazyFrame({"Category": ["A", "B", "C"]})
    colorway = ["#FF0000"]

    mappings = create_categorical_color_mappings(data, [], colorway)

    assert len(mappings) == 0
    assert mappings == {}
