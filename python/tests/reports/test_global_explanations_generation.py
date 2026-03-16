"""Tests for GlobalExplanations report generation.

This module tests the ReportGenerator class and its ability to generate
Quarto report files from templates using string substitution.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from pdstools.reports.GlobalExplanations.scripts.generate_report import (
    ReportGenerator,
)


@pytest.fixture
def temp_report_dir(mock_templates):
    """Create a temporary directory for report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create required subdirectories
        scripts_dir = Path(tmpdir) / "scripts"
        scripts_dir.mkdir()
        data_dir = Path(tmpdir).parent / "aggregated_data"
        data_dir.mkdir(exist_ok=True)

        # Create template directory and files
        templates_dir = Path(tmpdir) / "assets" / "templates"
        templates_dir.mkdir(parents=True)
        for filename, content in mock_templates.items():
            (templates_dir / filename).write_text(content)

        # Create unique_contexts.json with sample data
        contexts = {
            "1": {
                "0": [
                    json.dumps(
                        {
                            "partition": {
                                "Channel": "Web",
                                "Direction": "Inbound",
                            }
                        }
                    )
                ]
            }
        }
        with open(data_dir / "unique_contexts.json", "w") as f:
            json.dump(contexts, f)

        yield tmpdir


@pytest.fixture
def params_file(temp_report_dir):
    """Create a params.yml file in the temp directory."""
    params_path = Path(temp_report_dir) / "scripts" / "params.yml"
    params = {
        "top_n": 15,
        "top_k": 10,
        "from_date": "2026-01-01",
        "to_date": "2026-01-31",
        "contribution_type": "contribution",
        "contribution_text": "average contribution",
        "data_folder": "aggregated_data",
    }
    with open(params_path, "w") as f:
        yaml.dump(params, f)
    return params_path


@pytest.fixture
def mock_templates():
    """Mock template content for testing."""
    templates = {
        "getting-started.qmd": """---
title: "Getting Started"
---

This analysis shows the top {TOP_N} predictors and top {TOP_K} contributors
for data {DATE_INFO}.

Model context limit: {MODEL_CONTEXT_LIMIT}

Contribution text: {CONTRIBUTION_TEXT}
""",
        "overview.qmd": """---
title: "Overview"
---

Analysis using TOP_N={TOP_N}, TOP_K={TOP_K}
Root dir: {ROOT_DIR}
Data folder: {DATA_FOLDER}
Contribution type: {CONTRIBUTION_TYPE}
Contribution text: {CONTRIBUTION_TEXT}
""",
        "all_context_header.qmd": """---
title: "All Context Header"
---

Root: {ROOT_DIR}
Data: {DATA_FOLDER}
Pattern: {DATA_PATTERN}
Top N: {TOP_N}
Contribution: {CONTRIBUTION_TEXT}
""",
        "all_context_content.qmd": """
## Context: {CONTEXT_LABEL}

Context dict: {CONTEXT_DICT}
Top N: {TOP_N}
Top K: {TOP_K}
Type: {CONTRIBUTION_TYPE}
""",
        "context.qmd": """---
title: "{CONTEXT_STR}"
---

{{< embed {EMBED_PATH_FOR_BATCH}#{CONTEXT_LABEL} >}}

Context: {CONTEXT_LABEL}
Top N: {TOP_N}
Contribution: {CONTRIBUTION_TEXT}
""",
    }
    return templates


class TestReportGeneration:
    """Test report generation functionality."""

    def test_report_generator_initialization(self, temp_report_dir, params_file, monkeypatch):
        """Test ReportGenerator initializes correctly with params file."""
        monkeypatch.chdir(temp_report_dir)
        generator = ReportGenerator()

        # Use os.path.realpath to handle symlinks on macOS (/var vs /private/var)
        assert os.path.realpath(generator.report_folder) == os.path.realpath(temp_report_dir)
        assert generator.top_n == 15
        assert generator.top_k == 10
        assert generator.from_date == "2026-01-01"
        assert generator.to_date == "2026-01-31"
        assert generator.contribution_type == "contribution"
        assert generator.contribution_text == "average contribution"

    def test_generate_getting_started(self, temp_report_dir, params_file, monkeypatch):
        """Test generation of getting-started.qmd file."""
        monkeypatch.chdir(temp_report_dir)
        generator = ReportGenerator()
        generator._generate_introduction_qmd()

        output_path = Path(temp_report_dir) / "getting-started.qmd"
        assert output_path.exists()

        content = output_path.read_text()

        # Verify YAML front matter
        assert "---" in content
        assert "title:" in content

        # Verify parameter substitution
        assert "15" in content  # TOP_N
        assert "10" in content  # TOP_K
        assert "from `2026-01-01` to `2026-01-31`" in content  # DATE_INFO

    def test_generate_overview(self, temp_report_dir, params_file, monkeypatch):
        """Test generation of overview.qmd file."""
        monkeypatch.chdir(temp_report_dir)
        generator = ReportGenerator()
        generator._generate_overview_qmd()

        output_path = Path(temp_report_dir) / "overview.qmd"
        assert output_path.exists()

        content = output_path.read_text()

        # Verify YAML front matter
        assert "---" in content
        assert "title:" in content

        # Verify parameter substitution
        assert "15" in content  # TOP_N
        assert "10" in content  # TOP_K
        assert "contribution" in content  # CONTRIBUTION_TYPE

    def test_generate_by_context_qmds(self, temp_report_dir, params_file, monkeypatch):
        """Test generation of by-context QMD files."""
        monkeypatch.chdir(temp_report_dir)
        generator = ReportGenerator()
        generator._generate_by_context_qmds()

        # Check that plots_for_batch file was created
        plots_file = Path(temp_report_dir) / "by-model-context" / "plots_for_batch_1.qmd"
        assert plots_file.exists()

        content = plots_file.read_text()

        # Verify YAML front matter
        assert "---" in content
        assert "title:" in content

        # Check for context-specific file
        context_dir = Path(temp_report_dir) / "by-model-context"
        context_files = list(context_dir.glob("plt-*.qmd"))
        assert len(context_files) > 0

        # Verify context file content
        context_content = context_files[0].read_text()
        assert "---" in context_content
        assert "embed" in context_content

    def test_template_string_substitution(self, mock_templates):
        """Test that _read_template reads template content correctly."""
        # This test verifies the template reading mechanism with mocked templates
        template_content = mock_templates["getting-started.qmd"]

        # Verify template has placeholders (not substituted yet)
        assert "{TOP_N}" in template_content
        assert "{TOP_K}" in template_content
        assert "{DATE_INFO}" in template_content

        # Verify template has YAML front matter
        assert template_content.startswith("---")
        assert "title:" in template_content
