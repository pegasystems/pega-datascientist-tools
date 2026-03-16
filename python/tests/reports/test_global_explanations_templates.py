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


class TestGettingStartedTemplate:
    """Test getting-started.qmd meets Quarto standards."""

    def test_has_yaml_frontmatter(self):
        """Test template has complete YAML front matter."""
        template_path = get_template_path("getting-started.qmd")
        content = template_path.read_text()

        # Check for required YAML elements
        assert "title-block-banner: true" in content
        assert 'author: "Pega Data Scientist tools"' in content
        assert "date: today" in content
        assert "code-fold: true" in content
        assert "css: assets/pega-report-overrides.css" in content

    def test_has_credits_section(self):
        """Test template has credits section."""
        template_path = get_template_path("getting-started.qmd")
        content = template_path.read_text()

        assert "# Credits" in content
        assert "report_utils.show_credits" in content
        assert "show_versions.show_versions" in content

    def test_has_interactive_plot_disclaimer(self):
        """Test template has interactive plot disclaimer."""
        template_path = get_template_path("getting-started.qmd")
        content = template_path.read_text()

        assert ".callout-tip" in content
        assert "Plotly" in content
        assert "interactive plots" in content

    def test_preserves_placeholders(self):
        """Test template preserves all original placeholders."""
        template_path = get_template_path("getting-started.qmd")
        content = template_path.read_text()

        placeholders = ["{DATE_INFO}", "{TOP_N}", "{TOP_K}", "{CONTRIBUTION_TEXT}", "{MODEL_CONTEXT_LIMIT}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"


class TestOverviewTemplate:
    """Test overview.qmd meets Quarto standards."""

    def test_has_yaml_frontmatter(self):
        """Test template has complete YAML front matter."""
        template_path = get_template_path("overview.qmd")
        content = template_path.read_text()

        assert "title-block-banner: true" in content
        assert 'author: "Pega Data Scientist tools"' in content
        assert "css: assets/pega-report-overrides.css" in content

    def test_has_import_block(self):
        """Test template has proper import block."""
        template_path = get_template_path("overview.qmd")
        content = template_path.read_text()

        assert "from pdstools.utils import report_utils" in content
        assert "echo: false" in content

    def test_has_error_handling(self):
        """Test template wraps plots in try/except."""
        template_path = get_template_path("overview.qmd")
        content = template_path.read_text()

        assert "try:" in content
        assert "except Exception as e:" in content
        assert "report_utils.quarto_plot_exception" in content

    def test_has_pega_template(self):
        """Test template applies Pega styling to plots."""
        template_path = get_template_path("overview.qmd")
        content = template_path.read_text()

        assert 'template="pega"' in content or "template='pega'" in content

    def test_has_credits_section(self):
        """Test template has credits section."""
        template_path = get_template_path("overview.qmd")
        content = template_path.read_text()

        assert "# Credits" in content
        assert "report_utils.show_credits" in content

    def test_preserves_placeholders(self):
        """Test template preserves all original placeholders."""
        template_path = get_template_path("overview.qmd")
        content = template_path.read_text()

        placeholders = [
            "{ROOT_DIR}",
            "{DATA_FOLDER}",
            "{TOP_N}",
            "{TOP_K}",
            "{CONTRIBUTION_TYPE}",
            "{CONTRIBUTION_TEXT}",
        ]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"


class TestContextTemplate:
    """Test context.qmd meets minimal Quarto standards."""

    def test_has_yaml_frontmatter(self):
        """Test template has minimal YAML front matter."""
        template_path = get_template_path("context.qmd")
        content = template_path.read_text()

        assert "format: html" in content

    def test_preserves_embed_syntax(self):
        """Test template preserves Quarto embed syntax."""
        template_path = get_template_path("context.qmd")
        content = template_path.read_text()

        assert "{{{{< embed" in content
        assert ">}}}}" in content

    def test_preserves_placeholders(self):
        """Test template preserves all original placeholders."""
        template_path = get_template_path("context.qmd")
        content = template_path.read_text()

        placeholders = ["{CONTEXT_STR}", "{EMBED_PATH_FOR_BATCH}", "{CONTEXT_LABEL}", "{TOP_N}", "{CONTRIBUTION_TEXT}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"


class TestAllContextHeaderTemplate:
    """Test all_context_header.qmd meets Quarto standards."""

    def test_has_error_handling(self):
        """Test template wraps initialization in try/except."""
        template_path = get_template_path("all_context_header.qmd")
        content = template_path.read_text()

        assert "try:" in content
        assert "except Exception as e:" in content
        assert "logger.error" in content or "report_utils" in content

    def test_has_import_block(self):
        """Test template has proper import block."""
        template_path = get_template_path("all_context_header.qmd")
        content = template_path.read_text()

        assert "from pdstools.explanations import Explanations" in content

    def test_preserves_placeholders(self):
        """Test template preserves all original placeholders."""
        template_path = get_template_path("all_context_header.qmd")
        content = template_path.read_text()

        placeholders = ["{ROOT_DIR}", "{DATA_FOLDER}", "{DATA_PATTERN}", "{TOP_N}", "{CONTRIBUTION_TEXT}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"


class TestAllContextContentTemplate:
    """Test all_context_content.qmd meets Quarto standards."""

    def test_has_error_handling(self):
        """Test template wraps plots in try/except."""
        template_path = get_template_path("all_context_content.qmd")
        content = template_path.read_text()

        assert "try:" in content
        assert "except Exception as e:" in content
        assert "report_utils.quarto_plot_exception" in content

    def test_has_pega_template(self):
        """Test template applies Pega styling to plots."""
        template_path = get_template_path("all_context_content.qmd")
        content = template_path.read_text()

        assert 'template="pega"' in content or "template='pega'" in content

    def test_preserves_placeholders(self):
        """Test template preserves all original placeholders."""
        template_path = get_template_path("all_context_content.qmd")
        content = template_path.read_text()

        placeholders = ["{CONTEXT_DICT}", "{CONTEXT_LABEL}", "{TOP_N}", "{TOP_K}", "{CONTRIBUTION_TYPE}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"
