# Contribution Guide
We love feedback and contributions. Thank you for helping improve Pega Data Scientist Tools!

## Getting Help
- **Bugs / Feature Requests:** Open an [Issue](https://github.com/pegasystems/pega-datascientist-tools/issues).
- **Questions / Discussions:** Use the [Discussions](https://github.com/pegasystems/pega-datascientist-tools/discussions) tab.


## Contributing workflow
1. Fork this repository (creating a fork is only necessary when you are not part of the Pega organization)
2. Clone your fork.
3. Create a new branch for your change (`git checkout -b amazing-feature`).
4. Commit your work with clear messages (`git commit -m 'Added an amazing feature'`).
5. Push to the branch (`git push origin amazing-feature`)
6. Open a Pull Request (PR).


## Local development setup
Run setup commands from the repository root, not from the `python` folder. We use [`uv`](https://docs.astral.sh/uv/) for local development because it mirrors the environment used by CI.

```bash
uv sync --extra tests
```

Install additional dependency groups only when you need them:

```bash
# Documentation build and example notebooks
uv sync --extra docs --extra all

# ADM Health Check reports and their tests
uv sync --extra healthcheck --extra tests

# Formatting, linting, and pre-commit hooks
uv sync --extra dev
uv run pre-commit install
```

For Python commands, prefer `uv run ...` so the command uses the repository virtual environment:

```bash
uv run pytest python/tests
uv run ruff check ./python
```

### Running example notebooks locally
Example notebooks live under `examples/`. Open them from this repository checkout and select the `.venv` Python interpreter as the notebook kernel in VS Code or Jupyter.

Most notebooks use bundled sample data by default. For supported notebooks, set the corresponding environment variable below to use your own input instead of editing the data-loading cell.

Set the variable before starting the notebook kernel. If it is unset, the notebook keeps using its bundled or generated sample data; if it is set to an invalid path, the loading error is raised rather than silently reverting to sample data.

| Notebook(s) | Environment variable | Input |
| --- | --- | --- |
| `articles/AGBExplained.ipynb` | `AGB_REPORT_MODEL` | Exported AGB model JSON file |
| `adm/ADMBinningInsights.ipynb`, `adm/ADM_ActionDiscrimination_Metric.ipynb`, `articles/ADMExplained.ipynb`, and `datamart/Example_ADM_Analysis.ipynb` | `ADM_DATA_PATH` | Folder containing ADM model and predictor exports |
| `datamart/Example_ADM_Analysis.ipynb` individual report example | `ADM_MODEL_ID` | Model ID to report; defaults to the sample model ID |
| `ih/Example_IH_Analysis.ipynb` and `ih/Conversion_Reporting.ipynb` | `IH_DATA_PATH` | Interaction History export file |
| `prediction_studio/Predictions.ipynb` and `datamart/Example_ADM_Analysis.ipynb` | `PREDICTION_DATA_PATH` | `PR_DATA_DM_SNAPSHOTS` export file |
| `valuefinder/vf_analysis.ipynb` | `VALUE_FINDER_DATA_PATH` | `pyValueFinder` export file |
| `decision_analyzer/decision_analyzer.ipynb` | `DECISION_ANALYZER_DATA_PATH` | EEV2 data file supported by `pdstools.pega_io.read_data` |
| `explainability_extract/explainability_extract.ipynb` | `EXPLAINABILITY_EXTRACT_DATA_PATH` | Explainability Extract data file supported by `pdstools.pega_io.read_data` |
| `hds/hds_analysis.ipynb` | `HDS_DATA_PATH` | HDS ZIP archive |
| `impact-analyzer/impact_analyzer.ipynb` | `IMPACT_ANALYZER_DATA_PATH` | VBD Scenario Planner Actuals ZIP archive |
| `articles/explanations/agb_global_explanations.ipynb` | `GLOBAL_EXPLANATIONS_DATA_PATH` | Folder containing `OVERVIEW.parquet` and `BY_CONTEXT.parquet` |

For example:

```bash
export AGB_REPORT_MODEL=/path/to/model.json
export ADM_DATA_PATH=/path/to/adm-exports
```

These variables are used by the corresponding notebooks in `examples/`; each notebook documents its own input format and falls back to its normal sample when its variable is not set.

When exporting a notebook to HTML with executed plots, use the repository virtual environment rather than a system `jupyter` install:

```bash
.venv/bin/python -m nbconvert --to html --execute \
    examples/articles/AGBExplained.ipynb \
    --output AGBExplained.html --output-dir . \
    --ExecutePreprocessor.timeout=300
```

If the notebook kernel cannot import `pdstools`, re-run `uv sync --extra docs --extra all` from the repository root and re-select the `.venv` kernel.


## Testing
Pull requests trigger continuous integration via [GitHub Actions](https://docs.github.com/en/actions). These workflows automatically run tests and check on your changes. You can also enable Actions in your fork to test before submitting. To manually and locally run the tests with pytest, use either your IDE or the command line:

```bash
[uv run] pytest python/tests
```

To test with code coverage analysis run the following (make sure to have installed the `codecov` and `pytest-codecov` packages). Code coverage analysis is part of the GitHub continuous integration too.

```bash
[uv run] pytest python/tests --cov=./python/pdstools --cov-report=xml
```

## Documentation and articles
- The Python documentation uses `Sphinx` to generate the docs, `nbsphinx` to convert the jupyter notebooks to markdown, and `Furo` as the Sphinx template. These dependencies can be installed (from `python/docs`) with `[uv] pip install -r docs-requirements.txt`. `Pandoc` is a requirement too and needs to be installed separately.
- A Makefile is provided to create the documentation. The docs are automatically generated on any push commits in the master branch, but you can build them locally by following the steps below

### Create documentation locally
To generate a new version of the docs, simply navigate to the top-level folder and run the following command:
```sh
cd python/docs && [uv run] make html
```

### Edit or Add new content
To edit or add content, make changes / new files in the examples folder. The articles should be jupyter notebook files, and **should be empty, not pre-run**.

1. Create an article somewhere in the top-level `examples` folder. These articles should be jupyter notebook files, and should be **empty, not pre-run**.
2. Edit the cp command in `python/docs/Makefile` where the articles are copied to a temporary to include your article.
3. Update `python/docs/source/index.rst` and add your entry to the included notebooks. Since they will be moved to the `articles` folder, simply enter the name of the notebook file, without extension.
4. Be aware that the H1 headers of your notebook will be used as the entries in the doc index. If you have multiple H1's, you get multiple entries.

For example, if we added an article called `Example.ipynb` in the `examples/helloworld` folder, we would add `../../examples/helloworld/Example.ipynb` to the cp line in the Makefile (right before `source/articles`, of course), and add `articles/Example` to the `python/docs/source/index.rst` file. To test that it works, we recommend building the docs (`make html`) locally before creating a pull request.

## Style Guide

### Product Naming Conventions

When referring to pdstools applications, use consistent naming:

| Tool | User-facing name | Code/CLI reference |
|------|------------------|-------------------|
| Decision Analyzer | Decision Analysis Tool | `decision_analyzer` |
| ADM Health Check | ADM Health Check | `adm_healthcheck` |

**Rules:**
- **Documentation, UI headers, help text** → Use user-facing name
- **Code, CLI commands, file paths** → Use code reference
- Be consistent within each context

**Examples:**

✅ **Good:**
```markdown
# Documentation
The Decision Analysis Tool provides comprehensive insights...

# Code
from pdstools.decision_analyzer import DecisionAnalyzer
```

❌ **Avoid:**
```markdown
# Don't mix contexts
Run the decision_analyzer tool to analyze decisions.
# Should be: "Run the Decision Analysis Tool..."
```

### Version and Upgrade Text

**Do:**
- Show current version number
- Show "Keep up to date" message
- Let dynamic version checks show specific upgrade commands when needed

**Don't:**
- Include specific install commands in static UI headers
- Assume user's package manager (uv vs pip vs conda)

**Examples:**

✅ **Good:**
```python
st.caption(f"pdstools {version} · Keep up to date")

# Dynamic warning shown when update available:
if newer_version_available:
    st.warning(f"A newer version is available ({latest}). "
               "Run `uv pip install --upgrade pdstools` to update.")
```

❌ **Avoid:**
```python
# Static header with hardcoded command
st.caption(f"pdstools {version} · Keep up to date: `uv pip install --upgrade pdstools`")
```

### Documentation Updates

When updating analysis documentation:
- Keep analysis lists focused on key features
- Match current page structure
- Update descriptions when page names change
- Spell-check all user-facing text
