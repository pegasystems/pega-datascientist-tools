"""Tests for publishing synthetic report examples in the documentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "docs" / "generate_example_reports.py"
MODULE_SPEC = importlib.util.spec_from_file_location("generate_example_reports", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load docs helper module from {MODULE_PATH}")
example_reports = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(example_reports)


@pytest.fixture
def reports(monkeypatch):
    def render_health_check(**options):
        path = options["output_dir"] / "HealthCheck.html"
        path.write_text("<html>synthetic health check</html>", encoding="utf-8")
        return path

    def render_model_report(model_id, **options):
        path = options["output_dir"] / f"ModelReport_{model_id}.html"
        path.write_text("<html>synthetic model report</html>", encoding="utf-8")
        return path

    generate = SimpleNamespace(
        health_check=Mock(side_effect=render_health_check),
        model_reports=Mock(side_effect=render_model_report),
    )
    sample = SimpleNamespace(datamart=SimpleNamespace(generate=generate), prediction=object())
    demo_data = Mock(return_value=sample)
    monkeypatch.setattr(example_reports, "demo_data", demo_data)
    return sample, demo_data


def test_generates_standalone_reports_from_one_shared_sample(tmp_path, reports):
    sample, demo_data = reports
    output_dir = tmp_path / "static" / "reports"

    example_reports.generate_example_reports(output_dir)

    demo_data.assert_called_once_with(end="canonical")
    generate = sample.datamart.generate
    work_dir = generate.health_check.call_args.kwargs["output_dir"]
    generate.health_check.assert_called_once_with(
        prediction=sample.prediction,
        output_dir=work_dir,
        full_embed=False,
    )
    generate.model_reports.assert_called_once_with(
        "05085857-c52c-5daa-b51b-0af576a038af",
        output_dir=work_dir,
        full_embed=False,
    )
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "health-check.html",
        "naive-bayes-model-report.html",
    ]
    assert (output_dir / "health-check.html").read_text() == "<html>synthetic health check</html>"
    assert (output_dir / "naive-bayes-model-report.html").read_text() == "<html>synthetic model report</html>"
    assert not work_dir.exists()


@pytest.mark.parametrize("report_name", ["health_check", "model_reports"])
def test_rendering_errors_fail_the_build(tmp_path, reports, report_name):
    sample, _ = reports
    report = tmp_path / "broken.html"
    report.write_text("<html>Error rendering plot: synthetic failure</html>", encoding="utf-8")
    method = getattr(sample.datamart.generate, report_name)
    method.side_effect = None
    method.return_value = report

    with pytest.raises(ValueError, match="Rendering errors in broken.html"):
        example_reports.generate_example_reports(tmp_path / "published")


@pytest.mark.parametrize("report_name", ["health_check", "model_reports"])
def test_zip_reports_cannot_be_published_as_html(tmp_path, reports, report_name):
    sample, _ = reports
    method = getattr(sample.datamart.generate, report_name)
    method.side_effect = None
    method.return_value = tmp_path / "report.zip"

    with pytest.raises(ValueError, match="Expected standalone HTML"):
        example_reports.generate_example_reports(tmp_path / "published")


def test_quarto_failure_propagates(tmp_path, reports):
    sample, _ = reports
    sample.datamart.generate.health_check.side_effect = RuntimeError("Quarto rendering failed")

    with pytest.raises(RuntimeError, match="Quarto rendering failed"):
        example_reports.generate_example_reports(tmp_path / "published")
