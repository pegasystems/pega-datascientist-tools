"""Test cases for the Reports class that handles generating reports from aggregated data."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pdstools.explanations import Explanations

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


@pytest.fixture(scope="module")
def reports():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations.from_aggregates(
        base_path=DATA_DIR,
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield explanations.report


@pytest.fixture
def report_folder(tmp_path):
    """Provides a temporary report output directory."""
    return tmp_path / "reports"


def test_copy_report_resources(report_folder):
    """Test the _copy_report_resources static method."""
    from pdstools.explanations.Reports import Reports

    report_folder.mkdir(parents=True, exist_ok=True)
    Reports._copy_report_resources(report_folder)

    assert report_folder.exists(), "Report folder does not exist."
    assert any(report_folder.iterdir()), "Report folder is empty."

    assets_folder = report_folder / "assets"
    assert assets_folder.exists(), "Assets folder not copied."
    assert any(assets_folder.iterdir()), "Assets folder is empty."


def test_copy_report_resources_raises_on_error(report_folder):
    from pdstools.explanations.Reports import Reports

    report_folder.mkdir(parents=True, exist_ok=True)
    with patch(
        "pdstools.explanations.Reports.copy_report_resources",
        side_effect=OSError("fail"),
    ):
        with pytest.raises(OSError):
            Reports._copy_report_resources(report_folder)


def test_set_params(reports, report_folder):
    """Test _set_params writes all parameters including sort_by and display_by."""
    params_file = report_folder / "scripts" / "params.yml"
    data_folder = report_folder / "data"
    reports._set_params(params_file, data_folder, top_n=5, top_k=3, from_date="2026-01-01", to_date="2026-01-31")

    with open(params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert params["top_n"] == 5
    assert params["top_k"] == 3
    assert params["from_date"] == "2026-01-01"
    assert params["to_date"] == "2026-01-31"
    assert params["sort_by"] == "contribution_abs"
    assert params["sort_by_text"] == "absolute average contribution"
    assert params["display_by"] == "contribution"
    assert params["display_by_text"] == "average contribution"
    assert params["data_folder"] == str(data_folder)


def test_set_params_writes_resolved_data_folder(tmp_path):
    nested_aggregate_dir = tmp_path / "nested" / "aggregated_data"
    nested_aggregate_dir.mkdir(parents=True)
    for filename in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
        (nested_aggregate_dir / filename).write_bytes((DATA_DIR / filename).read_bytes())

    explanations = Explanations.from_aggregates(
        base_path=nested_aggregate_dir,
        model_name="AdaptiveBoostCT",
    )
    reports = explanations.report
    params_file = tmp_path / "reports" / "scripts" / "params.yml"
    data_folder = nested_aggregate_dir

    reports._set_params(params_file, data_folder)

    with open(params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # The params file stores whatever path was passed to _set_params.
    assert params["data_folder"] == str(data_folder)


def test_set_params_custom_contribution_types(reports, report_folder):
    """Test _set_params writes custom sort_by and display_by values."""
    sort_by = "contribution_abs"
    display_by = "contribution_abs"
    params_file = report_folder / "scripts" / "params.yml"
    data_folder = report_folder / "data"

    reports._set_params(
        params_file,
        data_folder,
        top_n=10,
        top_k=5,
        from_date="2026-03-01",
        to_date="2026-03-31",
        sort_by=sort_by,
        display_by=display_by,
    )

    with open(params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert params["top_n"] == 10
    assert params["top_k"] == 5
    assert params["sort_by"] == sort_by
    assert params["sort_by_text"] == "absolute average contribution"
    assert params["display_by"] == display_by
    assert params["display_by_text"] == "absolute average contribution"


def test_reports_logging(reports, report_folder, caplog):
    """Test that report operations produce debug logs when logging enabled."""
    from pdstools.explanations.Reports import Reports

    report_folder.mkdir(parents=True, exist_ok=True)
    params_file = report_folder / "scripts" / "params.yml"
    data_folder = report_folder / "data"

    with caplog.at_level(logging.DEBUG):
        Reports._copy_report_resources(report_folder)
        reports._set_params(params_file, data_folder, top_n=5, top_k=3)

    # Should have debug messages from both operations
    debug_messages = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert any("Copying report resources" in m for m in debug_messages), (
        f"Expected a 'Copying report resources' debug message; got: {debug_messages}"
    )
    assert any("Writing report parameters" in m for m in debug_messages), (
        f"Expected a 'Writing report parameters' debug message; got: {debug_messages}"
    )


class TestGenerateFilterKwargs:
    """Tests for filter_kwargs resolution inside generate()."""

    def test_generate_unknown_kwarg(self, reports):
        """Unknown kwargs should be rejected by Python — explicit signature."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            reports.generate(unknown_param=True)

    def test_generate_resolves_defaults(self, reports, report_folder):
        """generate() resolves filter_kwargs and passes enums to _set_params."""
        with (
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params") as mock_set_params,
            patch.object(
                reports.explanations.aggregates.context_operations,
                "write_batches",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
        ):
            reports.generate(output_dir=report_folder)

            mock_set_params.assert_called_once()
            call_kwargs = mock_set_params.call_args
            assert call_kwargs.kwargs["sort_by"] == "contribution_abs"
            assert call_kwargs.kwargs["display_by"] == "contribution"

    def test_generate_resolves_custom_kwargs(self, reports, report_folder):
        """generate() passes custom sort_by/display_by through the resolver."""
        with (
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params") as mock_set_params,
            patch.object(
                reports.explanations.aggregates.context_operations,
                "write_batches",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
        ):
            reports.generate(
                output_dir=report_folder,
                sort_by="contribution",
                display_by="contribution_abs",
            )

            call_kwargs = mock_set_params.call_args
            assert call_kwargs.kwargs["sort_by"] == "contribution"
            assert call_kwargs.kwargs["display_by"] == "contribution_abs"

    def test_generate_calls_write_batches(self, reports, report_folder):
        with (
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
            patch.object(
                reports.explanations.aggregates.context_operations,
                "write_batches",
            ) as mock_write_batches,
        ):
            reports.generate(output_dir=report_folder)

        mock_write_batches.assert_called_once_with(report_folder / "data")

    def test_generate_raises_when_save_data_fails(self, reports, report_folder):
        """Generation surfaces data access errors rather than swallowing them."""
        with (
            patch.object(reports.explanations, "save_data", side_effect=FileNotFoundError("data missing")),
            pytest.raises(FileNotFoundError),
        ):
            reports.generate(output_dir=report_folder)

    def test_generate_raises_when_copy_fails(self, reports, report_folder):
        with (
            patch.object(reports, "_copy_report_resources", side_effect=OSError("copy failed")),
            patch.object(
                reports.explanations.aggregates.context_operations,
                "write_batches",
            ),
        ):
            with pytest.raises(OSError, match="copy failed"):
                reports.generate(output_dir=report_folder)

    def test_generate_raises_when_quarto_process_errors(self, reports, report_folder):
        with (
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch.object(
                reports.explanations.aggregates.context_operations,
                "write_batches",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                side_effect=subprocess.CalledProcessError(1, "quarto"),
            ),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                reports.generate(output_dir=report_folder)

    def test_generate_raises_when_quarto_returns_nonzero(self, reports, report_folder):
        with (
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch.object(
                reports.explanations.aggregates.context_operations,
                "write_batches",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=2,
            ),
        ):
            with pytest.raises(RuntimeError, match="return code 2"):
                reports.generate(output_dir=report_folder)

    def test_generate_with_zip_output_creates_zip(self, reports, report_folder):
        with (
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch.object(
                reports.explanations.aggregates.context_operations,
                "write_batches",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
            patch(
                "pdstools.explanations.Reports.generate_zipped_report",
            ) as mock_zip,
        ):
            reports.generate(report_filename="out.zip", zip_output=True, output_dir=report_folder)

        mock_zip.assert_called_once_with("out.zip", report_folder / "_site")
