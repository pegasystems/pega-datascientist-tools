"""Playwright UI tests for channel filter across Decision Analyzer pages.

Prerequisites:
    1. Install: uv pip install playwright pytest-playwright && playwright install
    2. Start app: uv run pdstools decision_analyzer --data-path <your-data>
    3. Run: uv run pytest python/tests/test_channel_filter_ui.py -v
    4. Skip in CI: uv run pytest python/tests/ -m "not ui"
"""

import pytest

pytestmark = pytest.mark.ui

PAGES_WITH_FILTER = [
    "Action_Distribution",
    "Action_Funnel",
    "Global_Sensitivity",
    "Win_Loss_Analysis",
    "Optionality_Analysis",
    "Offer_Quality_Analysis",
    "Thresholding_Analysis",
    "Arbitration_Component_Distribution",
]


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the Streamlit app.

    Assumes app is running on localhost:8501.
    Start app before running tests: uv run pdstools decision_analyzer
    """
    return "http://localhost:8501"


def test_channel_selector_present_on_all_pages(page, app_url):
    """Verify channel selector appears on all expected pages."""
    for page_name in PAGES_WITH_FILTER:
        page.goto(f"{app_url}/{page_name}")
        page.wait_for_load_state("networkidle")

        selector = page.locator('label:has-text("Channel / Direction")')
        selector.wait_for(state="visible", timeout=15000)


def test_channel_filter_no_errors_on_selection(page, app_url):
    """Test that selecting a channel does not cause errors."""
    page.goto(f"{app_url}/Action_Distribution")
    page.wait_for_load_state("networkidle")

    # Wait for the selector to be present
    page.wait_for_selector('label:has-text("Channel / Direction")', timeout=15000)

    # Find the selectbox and get available options
    selectbox = page.locator('[data-testid="stSidebar"] select').first
    options = selectbox.locator("option").all_text_contents()

    # Select the second option (first non-"Any" channel) if available
    if len(options) > 1:
        selectbox.select_option(index=1)
        page.wait_for_load_state("networkidle", timeout=15000)

    # Verify no exception displayed
    error = page.locator('[data-testid="stException"]')
    assert error.count() == 0, "Exception displayed after channel selection"


def test_any_option_shows_all_data(page, app_url):
    """Test that selecting 'Any' shows all data without errors."""
    page.goto(f"{app_url}/Action_Distribution")
    page.wait_for_load_state("networkidle")

    page.wait_for_selector('label:has-text("Channel / Direction")', timeout=15000)

    selectbox = page.locator('[data-testid="stSidebar"] select').first
    options = selectbox.locator("option").all_text_contents()

    # Select a channel then go back to Any
    if len(options) > 1:
        selectbox.select_option(index=1)
        page.wait_for_load_state("networkidle", timeout=15000)

    selectbox.select_option(label="Any")
    page.wait_for_load_state("networkidle", timeout=15000)

    error = page.locator('[data-testid="stException"]')
    assert error.count() == 0, "Exception displayed after selecting 'Any'"


def test_overview_page_has_no_channel_filter(page, app_url):
    """Verify Overview page does NOT have channel filter."""
    page.goto(f"{app_url}/Overview")
    page.wait_for_load_state("networkidle")

    selector = page.locator('label:has-text("Channel / Direction")')
    assert selector.count() == 0, "Channel filter should not appear on Overview page"
