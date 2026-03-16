# GlobalExplanations Quarto Standards Alignment - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align GlobalExplanations report templates with Quarto standards by adding YAML front matter, error handling, Pega styling, and credits sections.

**Architecture:** Direct enhancement of 6 template files using string substitution patterns. Add standard Quarto elements (YAML, try/except blocks, credits) while preserving all existing placeholder logic. Follow TDD where possible by writing tests first, then updating templates.

**Tech Stack:** Quarto, Python, pdstools report utilities, pytest

---

## Task 1: Setup Test Infrastructure

**Files:**
- Create: `python/tests/reports/test_global_explanations_templates.py`

**Step 1: Create test file with basic structure**

```python
"""Tests for GlobalExplanations template validation."""
import os
from pathlib import Path
import pytest


def get_template_path(filename: str) -> Path:
    """Get path to a GlobalExplanations template file."""
    return Path(__file__).parent.parent.parent / "pdstools" / "reports" / "GlobalExplanations" / "assets" / "templates" / filename


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
```

**Step 2: Run tests to verify baseline**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py -v`
Expected: All tests PASS (templates exist)

**Step 3: Commit baseline tests**

```bash
git add python/tests/reports/test_global_explanations_templates.py
git commit -m "test: add baseline template existence tests for GlobalExplanations"
```

---

## Task 2: Update getting-started.qmd Template

**Files:**
- Modify: `python/pdstools/reports/GlobalExplanations/assets/templates/getting-started.qmd`
- Test: `python/tests/reports/test_global_explanations_templates.py`

**Step 1: Write tests for getting-started.qmd standards**

Add to `test_global_explanations_templates.py`:

```python
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

        placeholders = ["{DATE_INFO}", "{TOP_N}", "{TOP_K}",
                       "{CONTRIBUTION_TEXT}", "{MODEL_CONTEXT_LIMIT}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestGettingStartedTemplate -v`
Expected: FAIL - missing YAML, credits, disclaimer

**Step 3: Update getting-started.qmd template**

Replace content in `python/pdstools/reports/GlobalExplanations/assets/templates/getting-started.qmd`:

```markdown
---
title: "Getting Started"
title-block-banner: true
author: "Pega Data Scientist tools"
date: today
published-title: "Report generated on"
format:
  html:
    page-layout: full
    code-fold: true
    standalone: true
    toc: true
    theme:
      light: flatly
    css: assets/pega-report-overrides.css
---

## Introduction

This report contains a break down of the {CONTRIBUTION_TEXT} for each predictor made to model predictions. The report is generated over the aggregated explanations data {DATE_INFO}.

::: {.callout-tip}
Charts are built with [Plotly](https://plotly.com/python/) and have [user controls for panning, zooming etc](https://plotly.com/chart-studio-help/zoom-pan-hover-controls/).
These interactive plots do not render well in portals like Sharepoint or Box. View from a browser for best experience.
:::

### Model Overview

The model overview page shows `top-{TOP_N}` predictors that, on average, had the highest contribution to the model predictions.
In addition to the {CONTRIBUTION_TEXT} for those `top-{TOP_N}` predictors, the overview page also shows the {CONTRIBUTION_TEXT} for each predictor values. For symbolic predictors, the `top-{TOP_K}` predictor values are shown. For numeric predictors, the predictor values are binned according to the contribution deciles. The model overview will be generated over all model contexts.

### By model context

The **By Model Context** section displays the same information as in the model overview section, but for explanations aggregated within a specific model context. So instead of showing the {CONTRIBUTION_TEXT} of predictors, you can see the {CONTRIBUTION_TEXT} of predictors for a given a pyName, pyOutcome, pyIssue, etc.

*Note*: The model context plots are generated for the `top-{MODEL_CONTEXT_LIMIT}` model contexts sorted by frequency of samples.

# Credits

```{python}
# | echo: false
from pdstools.utils import show_versions
report_utils.show_credits("pega-datascientist-tools/python/pdstools/reports/GlobalExplanations/assets/templates/getting-started.qmd")
show_versions.show_versions(include_dependencies=False)
```

::: {.callout-note collapse="true"}
## Expand for detailed version information
```{python}
# | echo: false
show_versions.show_versions()
```
:::

For more information please see the [Pega Data Scientist Tools](https://github.com/pegasystems/pega-datascientist-tools).
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestGettingStartedTemplate -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/reports/GlobalExplanations/assets/templates/getting-started.qmd python/tests/reports/test_global_explanations_templates.py
git commit -m "feat: add Quarto standards to getting-started.qmd template

- Add complete YAML front matter with Pega branding
- Add credits section with show_credits and show_versions
- Add interactive plot disclaimer callout
- Preserve all existing placeholder substitutions"
```

---

## Task 3: Update overview.qmd Template

**Files:**
- Modify: `python/pdstools/reports/GlobalExplanations/assets/templates/overview.qmd`
- Test: `python/tests/reports/test_global_explanations_templates.py`

**Step 1: Write tests for overview.qmd standards**

Add to `test_global_explanations_templates.py`:

```python
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

        placeholders = ["{ROOT_DIR}", "{DATA_FOLDER}", "{TOP_N}",
                       "{TOP_K}", "{CONTRIBUTION_TYPE}", "{CONTRIBUTION_TEXT}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestOverviewTemplate -v`
Expected: FAIL - missing YAML, imports, error handling, Pega styling, credits

**Step 3: Update overview.qmd template**

Replace content in `python/pdstools/reports/GlobalExplanations/assets/templates/overview.qmd`:

```markdown
---
title: "Model Overview"
title-block-banner: true
author: "Pega Data Scientist tools"
date: today
format:
  html:
    page-layout: full
    code-fold: true
    standalone: true
    toc: true
    theme:
      light: flatly
    css: assets/pega-report-overrides.css
---

```{python}
# | echo: false
# | output: false
from pdstools.explanations import Explanations
from pdstools.utils import report_utils, show_versions
```

## Predictor contributions at model level

The top-{TOP_N} predictors over the entire model, selected by their {CONTRIBUTION_TEXT} to predictions

```{python}
explanations = Explanations(root_dir="{ROOT_DIR}")
explanations.aggregate.data_folderpath = "{DATA_FOLDER}"

try:
    overall_plot, predictor_plots = explanations.plot.plot_contributions_for_overall(
        top_n={TOP_N},
        top_k={TOP_K},
        contribution_calculation="{CONTRIBUTION_TYPE}"
    )
    overall_plot.update_layout(template="pega")
    overall_plot.show()
except Exception as e:
    report_utils.quarto_plot_exception("Overall Predictor Contributions", e)
```

## Predictor value contributions at model level

The predictor value contributions for the top-{TOP_N} predictors. Similar to above, the predictor values are selected by their {CONTRIBUTION_TEXT} to predictions.

```{python}
# | output: asis
try:
    from IPython.display import display, Markdown
    for plot in predictor_plots:
        title = plot.layout['title']['text']
        plot.layout['title']['text'] = ''
        plot.update_layout(template="pega")
        display(Markdown(f'### {title}'), plot)
except Exception as e:
    report_utils.quarto_plot_exception("Predictor Value Contributions", e)
```

# Credits

```{python}
# | echo: false
report_utils.show_credits("pega-datascientist-tools/python/pdstools/reports/GlobalExplanations/assets/templates/overview.qmd")
show_versions.show_versions(include_dependencies=False)
```

::: {.callout-note collapse="true"}
## Expand for detailed version information
```{python}
# | echo: false
show_versions.show_versions()
```
:::

For more information please see the [Pega Data Scientist Tools](https://github.com/pegasystems/pega-datascientist-tools).
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestOverviewTemplate -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/reports/GlobalExplanations/assets/templates/overview.qmd python/tests/reports/test_global_explanations_templates.py
git commit -m "feat: add Quarto standards to overview.qmd template

- Add complete YAML front matter with Pega branding
- Add import block with report_utils
- Wrap plot generation in try/except blocks
- Add template='pega' to all Plotly figures
- Add credits section with show_credits and show_versions
- Preserve all existing placeholder substitutions"
```

---

## Task 4: Update context.qmd Template

**Files:**
- Modify: `python/pdstools/reports/GlobalExplanations/assets/templates/context.qmd`
- Test: `python/tests/reports/test_global_explanations_templates.py`

**Step 1: Write tests for context.qmd standards**

Add to `test_global_explanations_templates.py`:

```python
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

        placeholders = ["{CONTEXT_STR}", "{EMBED_PATH_FOR_BATCH}",
                       "{CONTEXT_LABEL}", "{TOP_N}", "{CONTRIBUTION_TEXT}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestContextTemplate -v`
Expected: FAIL - missing YAML

**Step 3: Update context.qmd template**

Replace content in `python/pdstools/reports/GlobalExplanations/assets/templates/context.qmd`:

```markdown
---
title: "{CONTEXT_STR}"
format: html
---

## Model context info

{{{{< embed {EMBED_PATH_FOR_BATCH}#{CONTEXT_LABEL}-header >}}}}

## Predictor contributions

The top-{TOP_N} predictors for the context, selected by their {CONTRIBUTION_TEXT} to predictions

{{{{< embed {EMBED_PATH_FOR_BATCH}#{CONTEXT_LABEL}-overall >}}}}

## Predictor value contributions

The predictor value contributions for the top-{TOP_N} predictors. Similar to above, the predictor values are selected by their {CONTRIBUTION_TEXT} to predictions.

{{{{< embed {EMBED_PATH_FOR_BATCH}#{CONTEXT_LABEL}-predictors >}}}}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestContextTemplate -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/reports/GlobalExplanations/assets/templates/context.qmd python/tests/reports/test_global_explanations_templates.py
git commit -m "feat: add minimal YAML to context.qmd template

- Add format: html to YAML front matter
- Preserve all embed syntax unchanged
- Preserve all placeholder substitutions"
```

---

## Task 5: Update all_context_header.qmd Template

**Files:**
- Modify: `python/pdstools/reports/GlobalExplanations/assets/templates/all_context_header.qmd`
- Test: `python/tests/reports/test_global_explanations_templates.py`

**Step 1: Write tests for all_context_header.qmd standards**

Add to `test_global_explanations_templates.py`:

```python
class TestAllContextHeaderTemplate:
    """Test all_context_header.qmd meets Quarto standards."""

    def test_has_error_handling(self):
        """Test template wraps plots in try/except."""
        template_path = get_template_path("all_context_header.qmd")
        content = template_path.read_text()

        assert "try:" in content
        assert "except Exception as e:" in content
        assert "report_utils.quarto_plot_exception" in content

    def test_has_pega_template(self):
        """Test template applies Pega styling to plots."""
        template_path = get_template_path("all_context_header.qmd")
        content = template_path.read_text()

        assert 'template="pega"' in content or "template='pega'" in content

    def test_preserves_placeholders(self):
        """Test template preserves all original placeholders."""
        template_path = get_template_path("all_context_header.qmd")
        content = template_path.read_text()

        placeholders = ["{ROOT_DIR}", "{DATA_FOLDER}", "{DATA_PATTERN}",
                       "{TOP_N}", "{CONTRIBUTION_TEXT}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestAllContextHeaderTemplate -v`
Expected: FAIL - missing error handling, Pega styling

**Step 3: Read current all_context_header.qmd to understand structure**

Run: `cat python/pdstools/reports/GlobalExplanations/assets/templates/all_context_header.qmd`

**Step 4: Update all_context_header.qmd template with error handling**

The current template needs to be examined first, but the pattern is to wrap the existing plot code in try/except and add template="pega". Since we don't have the current content, here's the pattern to apply:

```python
# Before (example):
# explanations.plot.some_plot()

# After:
try:
    plot = explanations.plot.some_plot()
    plot.update_layout(template="pega")
    plot.show()
except Exception as e:
    report_utils.quarto_plot_exception("Plot Name", e)
```

Apply this pattern to all plot generation code in the template. Add import at the top:

```python
from pdstools.utils import report_utils
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestAllContextHeaderTemplate -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add python/pdstools/reports/GlobalExplanations/assets/templates/all_context_header.qmd python/tests/reports/test_global_explanations_templates.py
git commit -m "feat: add error handling and Pega styling to all_context_header.qmd

- Wrap plot generation in try/except blocks
- Add template='pega' to all Plotly figures
- Add report_utils import
- Preserve all placeholder substitutions"
```

---

## Task 6: Update all_context_content.qmd Template

**Files:**
- Modify: `python/pdstools/reports/GlobalExplanations/assets/templates/all_context_content.qmd`
- Test: `python/tests/reports/test_global_explanations_templates.py`

**Step 1: Write tests for all_context_content.qmd standards**

Add to `test_global_explanations_templates.py`:

```python
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

        placeholders = ["{CONTEXT_DICT}", "{CONTEXT_LABEL}",
                       "{TOP_N}", "{TOP_K}", "{CONTRIBUTION_TYPE}"]
        for placeholder in placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestAllContextContentTemplate -v`
Expected: FAIL - missing error handling, Pega styling

**Step 3: Read current all_context_content.qmd to understand structure**

Run: `cat python/pdstools/reports/GlobalExplanations/assets/templates/all_context_content.qmd`

**Step 4: Update all_context_content.qmd template with error handling**

Apply the same error handling pattern as in Task 5, wrapping plot generation in try/except blocks and adding template="pega" to figures.

**Step 5: Run tests to verify they pass**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py::TestAllContextContentTemplate -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add python/pdstools/reports/GlobalExplanations/assets/templates/all_context_content.qmd python/tests/reports/test_global_explanations_templates.py
git commit -m "feat: add error handling and Pega styling to all_context_content.qmd

- Wrap plot generation in try/except blocks
- Add template='pega' to all Plotly figures
- Preserve all placeholder substitutions"
```

---

## Task 7: Update generate_report.py Documentation

**Files:**
- Modify: `python/pdstools/reports/GlobalExplanations/scripts/generate_report.py`

**Step 1: Improve module and class docstrings**

Update the module docstring at the top of the file:

```python
"""GlobalExplanations report generation script.

This script generates Quarto report files from templates for model explanations analysis.
Templates follow Quarto standards with YAML front matter, error handling, Pega styling,
and credits sections. The generation process uses string substitution to populate
template placeholders with user-specified parameters.

Templates are located in: assets/templates/
Generated reports are written to the current working directory.
"""
```

Update the `ReportGenerator` class docstring:

```python
class ReportGenerator:
    """Generate GlobalExplanations Quarto report files from templates.

    This class reads template .qmd files, performs string substitution with
    user parameters, and writes the final report files. Templates follow
    established Quarto standards including:
    - Complete YAML front matter with Pega branding
    - Error handling around plot generation
    - Pega template styling for visualizations
    - Credits and version information sections

    The generation process preserves the template architecture while enhancing
    the quality and consistency of generated reports.
    """
```

**Step 2: Improve method docstrings**

Update key method docstrings to clarify their purpose:

```python
def _read_template(template_filename: str) -> str:
    """Read a template file and return its content.

    Templates contain placeholder strings (e.g., {ROOT_DIR}, {TOP_N}) that
    will be substituted with actual values during report generation.

    Parameters
    ----------
    template_filename : str
        Name of the template file in the templates folder

    Returns
    -------
    str
        Template content with placeholders intact
    """
```

**Step 3: Commit documentation improvements**

```bash
git add python/pdstools/reports/GlobalExplanations/scripts/generate_report.py
git commit -m "docs: improve GlobalExplanations generation script documentation

- Add comprehensive module docstring
- Enhance class docstring with standards overview
- Clarify method docstrings
- No functional changes"
```

---

## Task 8: Create Integration Tests

**Files:**
- Create: `python/tests/reports/test_global_explanations_generation.py`

**Step 1: Create integration test file**

```python
"""Integration tests for GlobalExplanations report generation."""
import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def temp_report_dir():
    """Create temporary directory for report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_data_folder(temp_report_dir):
    """Create mock data folder with unique_contexts.json."""
    data_folder = temp_report_dir / "aggregated_data"
    data_folder.mkdir()

    # Create minimal unique_contexts.json
    unique_contexts = {
        "1": {
            "1": ['{"partition": {"pyName": "TestModel", "pyIssue": "Sales"}}']
        }
    }

    import json
    with open(data_folder / "unique_contexts.json", "w") as f:
        json.dump(unique_contexts, f)

    return data_folder


class TestReportGeneration:
    """Test full report generation workflow."""

    def test_report_generator_instantiation(self, temp_report_dir):
        """Test ReportGenerator can be instantiated."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pdstools" / "reports" / "GlobalExplanations"))

        from scripts.generate_report import ReportGenerator

        os.chdir(temp_report_dir)
        generator = ReportGenerator()

        assert generator is not None
        assert generator.report_folder == str(temp_report_dir)

    def test_generates_getting_started(self, temp_report_dir):
        """Test generation creates getting-started.qmd."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pdstools" / "reports" / "GlobalExplanations"))

        from scripts.generate_report import ReportGenerator

        os.chdir(temp_report_dir)
        generator = ReportGenerator()
        generator._generate_introduction_qmd()

        output_file = temp_report_dir / "getting-started.qmd"
        assert output_file.exists()

        content = output_file.read_text()
        assert "title:" in content
        assert "# Credits" in content

    def test_generates_overview(self, temp_report_dir):
        """Test generation creates overview.qmd."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pdstools" / "reports" / "GlobalExplanations"))

        from scripts.generate_report import ReportGenerator

        os.chdir(temp_report_dir)
        generator = ReportGenerator()
        generator._generate_overview_qmd()

        output_file = temp_report_dir / "overview.qmd"
        assert output_file.exists()

        content = output_file.read_text()
        assert "try:" in content
        assert "report_utils.quarto_plot_exception" in content

    def test_generates_context_files(self, temp_report_dir, mock_data_folder):
        """Test generation creates context-specific files."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pdstools" / "reports" / "GlobalExplanations"))

        from scripts.generate_report import ReportGenerator

        os.chdir(temp_report_dir)
        generator = ReportGenerator()
        generator._generate_by_context_qmds()

        # Check batch file was created
        context_folder = temp_report_dir / "by-model-context"
        assert context_folder.exists()

        batch_files = list(context_folder.glob("plots_for_batch_*.qmd"))
        assert len(batch_files) > 0

    def test_string_substitution_works(self, temp_report_dir):
        """Test placeholder substitution produces valid output."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pdstools" / "reports" / "GlobalExplanations"))

        from scripts.generate_report import ReportGenerator

        os.chdir(temp_report_dir)
        generator = ReportGenerator()
        generator.top_n = 25
        generator.contribution_text = "average contribution"
        generator._generate_overview_qmd()

        output_file = temp_report_dir / "overview.qmd"
        content = output_file.read_text()

        # Check substitutions occurred
        assert "top-25" in content or "TOP_N" not in content
        assert "average contribution" in content or "CONTRIBUTION_TEXT" not in content
```

**Step 2: Run integration tests**

Run: `uv run pytest python/tests/reports/test_global_explanations_generation.py -v`
Expected: All tests PASS

**Step 3: Commit integration tests**

```bash
git add python/tests/reports/test_global_explanations_generation.py
git commit -m "test: add integration tests for GlobalExplanations generation

- Test ReportGenerator instantiation
- Test getting-started.qmd generation
- Test overview.qmd generation
- Test context file generation
- Test string substitution works correctly"
```

---

## Task 9: Manual End-to-End Validation

**Files:**
- None (manual testing)

**Step 1: Create test data structure**

Create a test directory with minimal explanations data structure:

```bash
mkdir -p /tmp/test_global_explanations/aggregated_data
cd /tmp/test_global_explanations
```

Create `aggregated_data/unique_contexts.json`:

```json
{
  "1": {
    "1": [
      "{\"partition\": {\"pyName\": \"TestModel\", \"pyIssue\": \"Sales\", \"pyGroup\": \"Cards\"}}"
    ]
  }
}
```

**Step 2: Run report generation**

From the test directory:

```bash
python /Users/perdo/dev/pega-datascientist-tools/python/pdstools/reports/GlobalExplanations/scripts/generate_report.py
```

Expected: No errors, files generated

**Step 3: Verify generated files**

Check that files were created:

```bash
ls -la
ls -la by-model-context/
```

Expected files:
- `getting-started.qmd`
- `overview.qmd`
- `by-model-context/plots_for_batch_1.qmd`
- `by-model-context/plt-testmodel-sales-cards.qmd`

**Step 4: Inspect file contents**

Check getting-started.qmd:

```bash
head -30 getting-started.qmd
```

Expected: YAML front matter, proper formatting

Check overview.qmd:

```bash
grep -A 3 "try:" overview.qmd
```

Expected: Error handling blocks present

**Step 5: Verify Quarto can parse files (optional, if Quarto installed)**

```bash
quarto check getting-started.qmd
quarto check overview.qmd
```

Expected: No syntax errors

**Step 6: Document validation results**

Create validation summary in issue comment or plan document noting:
- All files generated successfully
- YAML front matter present
- Error handling blocks present
- Credits sections present
- Pega styling applied

---

## Task 10: Update Test Coverage

**Files:**
- Modify: `python/tests/reports/test_global_explanations_templates.py`

**Step 1: Add coverage for edge cases**

Add additional tests for edge cases:

```python
class TestTemplatePlaceholders:
    """Test that all templates handle placeholders correctly."""

    def test_no_hardcoded_values(self):
        """Test templates use placeholders, not hardcoded values."""
        templates = [
            "getting-started.qmd",
            "overview.qmd",
            "all_context_header.qmd",
            "all_context_content.qmd"
        ]

        for template_name in templates:
            template_path = get_template_path(template_name)
            content = template_path.read_text()

            # Should not have hardcoded paths
            assert "/absolute/path/" not in content
            assert "C:\\" not in content

            # Should not have hardcoded numbers (except in YAML or formatting)
            # This is tricky, so just check for obvious issues


class TestTemplateConsistency:
    """Test consistency across templates."""

    def test_all_main_templates_have_credits(self):
        """Test main templates include credits sections."""
        main_templates = ["getting-started.qmd", "overview.qmd"]

        for template_name in main_templates:
            template_path = get_template_path(template_name)
            content = template_path.read_text()

            assert "# Credits" in content, f"{template_name} missing credits"
            assert "show_credits" in content, f"{template_name} missing show_credits call"

    def test_all_code_templates_have_error_handling(self):
        """Test templates with plot code include error handling."""
        code_templates = ["overview.qmd", "all_context_header.qmd", "all_context_content.qmd"]

        for template_name in code_templates:
            template_path = get_template_path(template_name)
            content = template_path.read_text()

            assert "try:" in content, f"{template_name} missing try block"
            assert "except Exception as e:" in content, f"{template_name} missing except block"
            assert "quarto_plot_exception" in content, f"{template_name} missing error handler"
```

**Step 2: Run full test suite**

Run: `uv run pytest python/tests/reports/test_global_explanations_templates.py -v`
Expected: All tests PASS

**Step 3: Check coverage**

Run: `uv run pytest python/tests/reports/ --cov=python/pdstools/reports/GlobalExplanations --cov-report=term-missing`
Expected: 80%+ coverage

**Step 4: Commit final tests**

```bash
git add python/tests/reports/test_global_explanations_templates.py
git commit -m "test: add edge case and consistency tests for templates

- Test no hardcoded values in templates
- Test all main templates have credits
- Test all code templates have error handling
- Verify 80%+ test coverage achieved"
```

---

## Task 11: Final Documentation Update

**Files:**
- Modify: `docs/plans/2026-03-16-global-explanations-quarto-standards-implementation.md`

**Step 1: Add completion summary to plan**

Add a completion section at the end of this implementation plan:

```markdown
## Implementation Complete

**Date Completed:** [DATE]

**Summary:**
- ✅ All 6 template files updated with Quarto standards
- ✅ Unit tests created and passing (80%+ coverage)
- ✅ Integration tests created and passing
- ✅ Manual end-to-end validation completed
- ✅ Documentation updated

**Files Modified:**
- python/pdstools/reports/GlobalExplanations/assets/templates/getting-started.qmd
- python/pdstools/reports/GlobalExplanations/assets/templates/overview.qmd
- python/pdstools/reports/GlobalExplanations/assets/templates/context.qmd
- python/pdstools/reports/GlobalExplanations/assets/templates/all_context_header.qmd
- python/pdstools/reports/GlobalExplanations/assets/templates/all_context_content.qmd
- python/pdstools/reports/GlobalExplanations/scripts/generate_report.py

**Tests Created:**
- python/tests/reports/test_global_explanations_templates.py (unit tests)
- python/tests/reports/test_global_explanations_generation.py (integration tests)

**Next Steps:**
- Create PR for review
- Update CLAUDE.md if GlobalExplanations patterns should be documented
- Consider deprecating template generation in favor of direct .qmd files (future work)
```

**Step 2: Commit plan update**

```bash
git add docs/plans/2026-03-16-global-explanations-quarto-standards-implementation.md
git commit -m "docs: mark implementation plan as complete"
```

**Step 3: Run final test suite**

Run: `uv run pytest python/tests/ -v`
Expected: All tests PASS including new GlobalExplanations tests

**Step 4: Final commit for the branch**

```bash
git add -A
git commit -m "feat: complete GlobalExplanations Quarto standards alignment

Closes #564

All templates now follow established Quarto standards:
- Complete YAML front matter with Pega branding
- Error handling around all plot generation
- Pega template styling for visualizations
- Credits and version information sections
- Comprehensive test coverage (80%+)

The template generation architecture is preserved while
enhancing the quality of generated reports."
```

---

## Success Criteria Checklist

- [ ] All 6 template files updated with required standards
- [ ] getting-started.qmd has YAML, credits, disclaimer
- [ ] overview.qmd has YAML, imports, error handling, Pega styling, credits
- [ ] context.qmd has minimal YAML
- [ ] all_context_header.qmd has error handling and Pega styling
- [ ] all_context_content.qmd has error handling and Pega styling
- [ ] generate_report.py documentation improved
- [ ] Unit tests created and passing
- [ ] Integration tests created and passing
- [ ] Manual end-to-end validation completed
- [ ] Test coverage ≥80% for generation logic
- [ ] All tests pass in full suite
- [ ] All commits follow conventional commit format
- [ ] Implementation plan marked complete
- [ ] Ready for PR creation

## Notes

- Focus on minimal, targeted changes - don't refactor beyond requirements
- Preserve all existing placeholder substitution logic
- Test after each template update to catch issues early
- Commit frequently with clear, descriptive messages
- Follow TDD pattern: test → fail → implement → pass → commit
