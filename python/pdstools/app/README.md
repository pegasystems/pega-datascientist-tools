# pdstools Apps

This directory contains three Streamlit apps:

| App | Command | Description |
|-----|---------|-------------|
| **Adaptive Model Health Check** | `pdstools health_check` | ADM Datamart analysis with report generation |
| **Decision Analysis** | `pdstools decision_analyzer` | NBA decision pipeline diagnostics |
| **Impact Analyzer** | `pdstools impact_analyzer` | A/B test experiment analysis |

## Running

```bash
# Interactive picker
pdstools

# Or directly by name
pdstools decision_analyzer

# For managed deployments (e.g. EC2)
pdstools decision_analyzer --deploy-env ec2
```

The `--deploy-env` flag sets the `PDSTOOLS_DEPLOY_ENV` environment variable, which
controls deployment-specific behavior (e.g. hiding the "File path" data source option,
using a local S3 sample data path). The sample data path can be overridden with the
`PDSTOOLS_SAMPLE_DATA_PATH` environment variable.

## Architecture

All three apps share common infrastructure from `pdstools.utils.streamlit_utils`:

- **`standard_page_config()`** — Consistent page layout and menu items
- **`show_version_header()`** — Version display with upgrade hint and PyPI check
- **`ensure_session_data()`** — Page guard for missing data

App-specific utilities live in each app's `*_streamlit_utils.py` module.
