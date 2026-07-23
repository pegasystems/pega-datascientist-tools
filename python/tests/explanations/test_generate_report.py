"""Tests for flat unique-context iteration in the report generator."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pdstools.reports.GlobalExplanations.scripts.generate_report import ReportGenerator

TEMPLATES_DIR = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "pdstools"
    / "reports"
    / "GlobalExplanations"
    / "assets"
    / "templates"
)


def _write_contexts(data_dir: Path) -> dict[str, list[str]]:
    contexts = {
        "100": [
            '{"partition":{"pyChannel":"Web","pyName":"A1"}}',
            '{"partition":{"pyChannel":"Web","pyName":"A2"}}',
            '{"partition":{"pyChannel":"Web","pyName":"A3"}}',
        ],
        "200": [
            '{"partition":{"pyChannel":"Email","pyName":"B1"}}',
            '{"partition":{"pyChannel":"Email","pyName":"B2"}}',
            '{"partition":{"pyChannel":"Email","pyName":"B3"}}',
        ],
    }
    (data_dir / "unique_contexts.json").write_text(json.dumps(contexts), encoding="utf-8")
    return contexts


def _template_for(filename: str) -> str:
    if filename == "all_context_header.qmd":
        return "---\n{DATA_PATTERN}\n{TOP_N}\n{SORT_BY_TEXT}\n---"
    if filename == "all_context_content.qmd":
        return "{CONTEXT_LABEL}\n{CONTEXT_DICT}\n{TOP_N}\n{TOP_K}\n{SORT_BY}\n{DISPLAY_BY}"
    if filename == "context.qmd":
        return "---\n{EMBED_PATH_FOR_BATCH}\n{CONTEXT_STR}\n{CONTEXT_LABEL}\n{TOP_N}\n{SORT_BY_TEXT}\n---"
    raise AssertionError(f"unexpected template: {filename}")


def test_generate_by_context_qmds_flat_iteration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "aggregated_data"
    data_dir.mkdir()
    _write_contexts(data_dir)

    generator = ReportGenerator()
    generator.data_folder = str(data_dir)

    with patch.object(generator, "_read_template", side_effect=_template_for):
        generator._generate_by_context_qmds()

    plots_qmds = sorted((tmp_path / "by-model-context").glob("plots_for_batch_*.qmd"))
    assert [path.name for path in plots_qmds] == ["plots_for_batch_100.qmd", "plots_for_batch_200.qmd"]


def test_generate_by_context_qmds_context_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "aggregated_data"
    data_dir.mkdir()
    contexts = _write_contexts(data_dir)

    generator = ReportGenerator()
    generator.data_folder = str(data_dir)

    with patch.object(generator, "_read_template", side_effect=_template_for):
        generator._generate_by_context_qmds()

    context_files = sorted((tmp_path / "by-model-context").glob("plt-*.qmd"))
    assert len(context_files) == sum(len(batch) for batch in contexts.values())


def test_generate_by_context_qmds_no_nested_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "aggregated_data"
    data_dir.mkdir()
    contexts = _write_contexts(data_dir)

    generator = ReportGenerator()
    generator.data_folder = str(data_dir)

    with patch.object(generator, "_read_template", side_effect=_template_for):
        generator._generate_by_context_qmds()

    context_files = list((tmp_path / "by-model-context").glob("plt-*.qmd"))
    assert len(context_files) == 6
    assert len(context_files) == sum(len(batch) for batch in contexts.values())


def test_embedded_context_templates_use_display(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    all_context_header_template = (TEMPLATES_DIR / "all_context_header.qmd").read_text(encoding="utf-8")
    all_context_content_template = (TEMPLATES_DIR / "all_context_content.qmd").read_text(encoding="utf-8")
    overview_template = (TEMPLATES_DIR / "overview.qmd").read_text(encoding="utf-8")

    assert "header_tbl.show()" not in all_context_content_template
    assert "overall_fig.show()" not in all_context_content_template
    assert "display(header_tbl)" in all_context_content_template
    assert "display(overall_fig)" in all_context_content_template
    assert 'data_folder="{DATA_FOLDER}"' in all_context_header_template
    assert 'data_folder="{DATA_FOLDER}"' in overview_template
    assert "overall_plot.show()" not in overview_template
    assert "display(overall_plot)" in overview_template


def test_get_unique_contexts_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "aggregated_data"
    data_dir.mkdir()

    generator = ReportGenerator()
    generator.data_folder = str(data_dir)

    with pytest.raises(FileNotFoundError, match="Unique contexts file not found"):
        generator._get_unique_contexts()
