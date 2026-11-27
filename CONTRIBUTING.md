# Contributing to pdstools

Thank you for helping improve pdstools.

## Getting help

- Report bugs or request features through [Issues](https://github.com/pegasystems/pega-datascientist-tools/issues).
- Ask questions through [Discussions](https://github.com/pegasystems/pega-datascientist-tools/discussions).

## Contribution workflow

Fork the repository if you do not have write access. Open a pull request
describing the problem, the change, compatibility implications, and validation.
The shared [development guide](AGENTS.md) applies to both human and
agent-assisted contributions; it owns coding conventions and publication checks.

## Local development setup

Setup, dependency extras, testing, linting, and build/check entry points are
maintained in [Environment and validation](AGENTS.md#environment-and-validation).
Use those instructions rather than a separate tool-specific setup.

## Running example notebooks locally

Example notebooks live under [examples/](examples/). Use this checkout's
`.venv` interpreter as the notebook kernel, regardless of editor.

Most notebooks use bundled sample data by default. For supported notebooks,
set the corresponding environment variable below to use your own input instead
of editing the data-loading cell.

Set the variable before starting the kernel. When unset, the notebook uses its
bundled or generated sample; an invalid supplied path raises a loading error
rather than silently falling back to sample data.

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

Each notebook documents its own input format. Do not commit private paths,
data, or executed outputs when using these overrides.

To export a notebook to HTML with executed plots:

```bash
uv run python -m nbconvert --to html --execute \
    examples/articles/AGBExplained.ipynb \
    --output AGBExplained.html --output-dir . \
    --ExecutePreprocessor.timeout=300
```

Ensure the execution kernel also uses the project environment. If it cannot
import `pdstools`, check the selected kernel and restore the documented
environment rather than using a system Jupyter installation.

## Documentation and articles

The documentation uses Sphinx, nbsphinx, and Furo. See the shared development
guide for dependencies and the build command; Pandoc is also required for
notebook conversion. The [Makefile](python/docs/Makefile) prepares notebook
articles and generates example reports before building HTML.

To add an article:

1. Add an output-cleared notebook under [examples/](examples/).
2. Ensure the Makefile's copy command includes it, either explicitly or through
   an existing glob, so it reaches `source/articles`.
3. Add its `articles/<name>` entry, without the extension, to
   [the documentation index](python/docs/source/index.rst).
4. Review notebook H1 headings, which become entries in the documentation index,
   and validate the resulting documentation.

## Style and documentation conventions

Use the shared guide's [Python and public API conventions](AGENTS.md#python-and-public-apis)
and [plot, report, and notebook conventions](AGENTS.md#plots-reports-and-notebooks).
Keep analysis documentation aligned with current features and page names.
