"""Tests for GlobalExplanations template validation."""

from pathlib import Path


def get_template_path(filename: str) -> Path:
    """Get path to a GlobalExplanations template file."""
    return (
        Path(__file__).parent.parent.parent
        / "pdstools"
        / "reports"
        / "GlobalExplanations"
        / "assets"
        / "templates"
        / filename
    )


class TestTemplateFiles:
    """Test that template files exist and are readable."""

    def test_getting_started_exists(self):
        """Test getting-started.qmd template exists."""
        template_path = get_template_path("getting-started.qmd")
        assert template_path.exists(), f"Template not found: {template_path}"

    def test_overview_exists(self):
        """Test overview.qmd template exists."""
        template_path = get_template_path("overview.qmd")
        assert template_path.exists(), f"Template not found: {template_path}"

    def test_context_exists(self):
        """Test context.qmd template exists."""
        template_path = get_template_path("context.qmd")
        assert template_path.exists(), f"Template not found: {template_path}"

    def test_all_context_header_exists(self):
        """Test all_context_header.qmd template exists."""
        template_path = get_template_path("all_context_header.qmd")
        assert template_path.exists(), f"Template not found: {template_path}"

    def test_all_context_content_exists(self):
        """Test all_context_content.qmd template exists."""
        template_path = get_template_path("all_context_content.qmd")
        assert template_path.exists(), f"Template not found: {template_path}"
