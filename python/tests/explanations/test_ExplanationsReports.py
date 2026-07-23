"""Test cases for the Reports class that handles generating reports from aggregated data."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pdstools.explanations import Explanations
from pdstools.explanations.ExplanationsUtils import _CONTRIBUTION_TYPE

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


@pytest.fixture(scope="module")
def reports():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations.from_aggregates(
        data_folder=DATA_DIR,
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield explanations.report


@pytest.fixture
def report_paths(reports, tmp_path):
    reports.report_folderpath = tmp_path / "reports"
    reports.report_output_dir = reports.report_folderpath / "_site"
    reports.params_file = reports.report_folderpath / "scripts" / "params.yml"
    yield reports


def test_validate_report_dir(report_paths):
    """Test the report directory validation."""
    reports = report_paths
    reports._validate_report_dir()

    assert reports.report_folderpath.exists(), "Report folder does not exist."
    assert reports.report_folderpath.is_dir(), "Report folder is not a directory."


def test_copy_report_resources(report_paths):
    """Test the copy_report_resources method."""
    reports = report_paths
    reports._validate_report_dir()
    reports._copy_report_resources()

    assert reports.report_folderpath.exists(), "Report folder does not exist."
    assert any(reports.report_folderpath.iterdir()), "Report folder is empty."

    assets_folder = reports.report_folderpath / "assets"
    assert assets_folder.exists(), "Assets folder not copied."
    assert any(assets_folder.iterdir()), "Assets folder is empty."


def test_copy_report_resources_raises_on_error(report_paths):
    reports = report_paths
    with patch(
        "pdstools.explanations.Reports.copy_report_resources",
        side_effect=OSError("fail"),
    ):
        with pytest.raises(OSError):
            reports._copy_report_resources()


def test_set_params(report_paths):
    """Test _set_params writes all parameters including sort_by and display_by."""
    reports = report_paths
    reports._validate_report_dir()
    reports._copy_report_resources()
    reports._set_params(top_n=5, top_k=3, from_date="2026-01-01", to_date="2026-01-31")

    with open(reports.params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert params["top_n"] == 5
    assert params["top_k"] == 3
    assert params["from_date"] == "2026-01-01"
    assert params["to_date"] == "2026-01-31"
    assert params["sort_by"] == "contribution_abs"
    assert params["sort_by_text"] == "absolute average contribution"
    assert params["display_by"] == "contribution"
    assert params["display_by_text"] == "average contribution"
    assert params["data_folder"] == Path(reports.explanations.data_folder).as_posix()


def test_set_params_preserves_nested_relative_data_folder(tmp_path):
    nested_aggregate_dir = tmp_path / "nested" / "aggregated_data"
    nested_aggregate_dir.mkdir(parents=True)
    for filename in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
        (nested_aggregate_dir / filename).write_bytes((DATA_DIR / filename).read_bytes())

    explanations = Explanations.from_aggregates(
        root_dir=str(tmp_path),
        data_folder="nested/aggregated_data",
        model_name="AdaptiveBoostCT",
    )
    reports = explanations.report
    reports.report_folderpath = tmp_path / "reports"
    reports.report_output_dir = reports.report_folderpath / "_site"
    reports.params_file = reports.report_folderpath / "scripts" / "params.yml"

    reports._validate_report_dir()
    reports._copy_report_resources()
    reports._set_params()

    with open(reports.params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert params["data_folder"] == "nested/aggregated_data"


def test_set_params_custom_contribution_types(report_paths):
    """Test _set_params writes custom sort_by and display_by values."""
    reports = report_paths
    reports._validate_report_dir()
    reports._copy_report_resources()

    sort_by = _CONTRIBUTION_TYPE.CONTRIBUTION_ABS
    display_by = _CONTRIBUTION_TYPE.CONTRIBUTION_ABS

    reports._set_params(
        top_n=10,
        top_k=5,
        from_date="2026-03-01",
        to_date="2026-03-31",
        sort_by=sort_by,
        display_by=display_by,
    )

    with open(reports.params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert params["top_n"] == 10
    assert params["top_k"] == 5
    assert params["sort_by"] == sort_by.value
    assert params["sort_by_text"] == sort_by.text
    assert params["display_by"] == display_by.value
    assert params["display_by_text"] == display_by.text


def test_reports_logging(report_paths, caplog):
    """Test that report operations produce debug logs when logging enabled."""
    reports = report_paths
    reports._validate_report_dir()

    with caplog.at_level(logging.DEBUG):
        reports._copy_report_resources()
        reports._set_params(top_n=5, top_k=3)

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

    def test_generate_resolves_defaults(self, report_paths):
        """generate() resolves filter_kwargs and passes enums to _set_params."""
        reports = report_paths
        with (
            patch.object(reports, "_validate_report_dir"),
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params") as mock_set_params,
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value={100: ["ctx1"]},
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
        ):
            reports.generate()

            mock_set_params.assert_called_once()
            call_kwargs = mock_set_params.call_args
            assert call_kwargs.kwargs["sort_by"] == _CONTRIBUTION_TYPE.CONTRIBUTION_ABS
            assert call_kwargs.kwargs["display_by"] == _CONTRIBUTION_TYPE.CONTRIBUTION

    def test_generate_resolves_custom_kwargs(self, report_paths):
        """generate() passes custom sort_by/display_by through the resolver."""
        reports = report_paths
        with (
            patch.object(reports, "_validate_report_dir"),
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params") as mock_set_params,
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value={100: ["ctx1"]},
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
        ):
            reports.generate(
                sort_by="contribution",
                display_by="contribution_abs",
            )

            call_kwargs = mock_set_params.call_args
            assert call_kwargs.kwargs["sort_by"] == _CONTRIBUTION_TYPE.CONTRIBUTION
            assert call_kwargs.kwargs["display_by"] == _CONTRIBUTION_TYPE.CONTRIBUTION_ABS

    def test_generate_calls_create_unique_contexts_file(self, report_paths):
        reports = report_paths
        with (
            patch.object(reports, "_validate_report_dir"),
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value={100: ["ctx1"]},
            ) as mock_create_unique_contexts_file,
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ),
        ):
            reports.generate()

        mock_create_unique_contexts_file.assert_called_once_with()

    def test_generate_calls_create_batch_parquet_files(self, report_paths):
        reports = report_paths
        contexts = {100: ["ctx1"]}
        with (
            patch.object(reports, "_validate_report_dir"),
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value=contexts,
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ) as mock_create_batch_parquet_files,
        ):
            reports.generate()

        mock_create_batch_parquet_files.assert_called_once_with(contexts)

    def test_generate_raises_when_validation_fails(self, report_paths):
        reports = report_paths
        with patch.object(
            reports.explanations,
            "validate_data_folder",
            side_effect=FileNotFoundError("missing parquet"),
        ):
            with pytest.raises(FileNotFoundError, match="missing parquet"):
                reports.generate()

    def test_generate_raises_when_copy_fails(self, report_paths):
        reports = report_paths
        with (
            patch.object(reports, "_copy_report_resources", side_effect=OSError("copy failed")),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value={"100": ["ctx1"]},
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ),
        ):
            with pytest.raises(OSError, match="copy failed"):
                reports.generate()

    def test_generate_raises_when_quarto_process_errors(self, report_paths):
        reports = report_paths
        with (
            patch.object(reports, "_validate_report_dir"),
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value={"100": ["ctx1"]},
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                side_effect=subprocess.CalledProcessError(1, "quarto"),
            ),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                reports.generate()

    def test_generate_raises_when_quarto_returns_nonzero(self, report_paths):
        reports = report_paths
        with (
            patch.object(reports, "_validate_report_dir"),
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value={"100": ["ctx1"]},
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=2,
            ),
        ):
            with pytest.raises(RuntimeError, match="return code 2"):
                reports.generate()

    def test_generate_with_zip_output_creates_zip(self, report_paths):
        reports = report_paths
        with (
            patch.object(reports, "_validate_report_dir"),
            patch.object(reports, "_copy_report_resources"),
            patch.object(reports, "_set_params"),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_unique_contexts_file",
                return_value={"100": ["ctx1"]},
            ),
            patch.object(
                reports.explanations.aggregate.context_operations,
                "create_batch_parquet_files",
            ),
            patch(
                "pdstools.explanations.Reports.run_quarto",
                return_value=0,
            ),
            patch(
                "pdstools.explanations.Reports.generate_zipped_report",
            ) as mock_zip,
        ):
            reports.generate(report_filename="out.zip", zip_output=True)

        mock_zip.assert_called_once_with("out.zip", reports.report_output_dir)
