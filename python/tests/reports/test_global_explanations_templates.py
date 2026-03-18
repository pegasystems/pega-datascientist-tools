"""Tests for GlobalExplanations template validation."""

from pathlib import Path
import re

import pytest

TEMPLATES_DIR = (
    Path(__file__).parent.parent.parent / "pdstools" / "reports" / "GlobalExplanations" / "assets" / "templates"
)

TEMPLATE_PLACEHOLDERS: dict[str, set[str]] = {
    "getting-started.qmd": {
        "{DATE_INFO}",
        "{TOP_N}",
        "{TOP_K}",
        "{CONTRIBUTION_TEXT}",
        "{MODEL_CONTEXT_LIMIT}",
    },
    "overview.qmd": {
        "{ROOT_DIR}",
        "{DATA_FOLDER}",
        "{TOP_N}",
        "{TOP_K}",
        "{CONTRIBUTION_TYPE}",
        "{CONTRIBUTION_TEXT}",
    },
    "context.qmd": {
        "{CONTEXT_STR}",
        "{EMBED_PATH_FOR_BATCH}",
        "{CONTEXT_LABEL}",
        "{TOP_N}",
        "{CONTRIBUTION_TEXT}",
    },
    "all_context_header.qmd": {
        "{ROOT_DIR}",
        "{DATA_FOLDER}",
        "{DATA_PATTERN}",
        "{TOP_N}",
        "{CONTRIBUTION_TEXT}",
    },
    "all_context_content.qmd": {
        "{CONTEXT_DICT}",
        "{CONTEXT_LABEL}",
        "{TOP_N}",
        "{TOP_K}",
        "{CONTRIBUTION_TYPE}",
    },
}


TEMPLATES_WITH_PYTHON_BLOCKS: list[str] = [
    "getting-started.qmd",
    "overview.qmd",
    "context.qmd",
    "all_context_header.qmd",
    "all_context_content.qmd",
]

TEMPLATES_WITH_YAML_FRONTMATTER: dict[str, str] = {
    "getting-started.qmd": '"Getting Started"',
    "overview.qmd": '"Model Overview"',
    "context.qmd": '"{CONTEXT_STR}"',
}


def get_template_path(filename: str) -> Path:
    """Get path to a GlobalExplanations template file."""
    return TEMPLATES_DIR / filename


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


class TestGettingStartedTemplate:
    """Test getting-started.qmd meets Quarto standards."""

    def test_has_interactive_plot_disclaimer(self):
        """Test template has interactive plot disclaimer."""
        template_path = get_template_path("getting-started.qmd")
        content = template_path.read_text()

        assert ".callout-tip" in content
        assert "Plotly" in content
        assert "interactive plots" in content


class TestTemplatePlaceholders:
    """Test placeholder correctness across all templates."""

    @pytest.mark.parametrize("template_name", list(TEMPLATE_PLACEHOLDERS.keys()))
    def test_contains_expected_placeholders(self, template_name):
        """Test each template contains all its expected placeholders."""
        content = get_template_path(template_name).read_text()
        expected = TEMPLATE_PLACEHOLDERS[template_name]

        for placeholder in expected:
            assert placeholder in content, f"{template_name} missing placeholder: {placeholder}"

    @pytest.mark.parametrize("template_name", list(TEMPLATE_PLACEHOLDERS.keys()))
    def test_no_unexpected_placeholders(self, template_name):
        """Test each template contains only its expected placeholders."""
        content = get_template_path(template_name).read_text()
        expected = TEMPLATE_PLACEHOLDERS[template_name]
        found = set(re.findall(r"\{[A-Z_]+\}", content))

        unexpected = found - expected
        assert not unexpected, f"{template_name} has unexpected placeholders: {unexpected}"


class TestCommonComponents:
    """Test common components shared across multiple templates."""

    @pytest.mark.parametrize(
        "template_name,expected_title",
        list(TEMPLATES_WITH_YAML_FRONTMATTER.items()),
    )
    def test_has_yaml_frontmatter(self, template_name, expected_title):
        """Test templates have YAML front matter with title, date, and published-title."""
        content = get_template_path(template_name).read_text()

        assert f"title: {expected_title}" in content, f"{template_name} missing expected title: {expected_title}"
        assert "date: today" in content, f"{template_name} missing 'date: today'"
        assert 'published-title: "Report generated on"' in content, f"{template_name} missing published-title"

    @pytest.mark.parametrize(
        "template_name",
        ["getting-started.qmd", "overview.qmd", "context.qmd"],
    )
    def test_has_credits_section(self, template_name):
        """Test templates include a credits section."""
        content = get_template_path(template_name).read_text()

        assert "# Credits" in content, f"{template_name} missing credits header"
        assert "from pdstools.utils import report_utils, show_versions" in content, (
            f"{template_name} missing credits imports"
        )
        assert "report_utils.show_credits" in content, f"{template_name} missing show_credits call"
        assert "show_versions.show_versions" in content, f"{template_name} missing show_versions call"

    @pytest.mark.parametrize("template_name", TEMPLATES_WITH_PYTHON_BLOCKS)
    def test_python_blocks_use_double_braces(self, template_name):
        """Test python code blocks use Quarto template syntax ``{{python}}``."""
        content = get_template_path(template_name).read_text()

        assert "```{{python}}" in content, f"{template_name} has no ```{{{{python}}}}` blocks"
        assert "```{python}" not in content, (
            f"{template_name} uses single-brace ```{{python}}` instead of double-brace ```{{{{{{{{python}}}}}}}}`"
        )


class TestTemplateConsistency:
    """Test consistency across templates."""

    def test_all_code_templates_have_error_handling(self):
        """Test templates with plot code include error handling."""
        code_templates = ["overview.qmd", "all_context_content.qmd"]

        for template_name in code_templates:
            template_path = get_template_path(template_name)
            content = template_path.read_text()

            assert "try:" in content, f"{template_name} missing try block"
            assert "except Exception as e:" in content, f"{template_name} missing except block"
