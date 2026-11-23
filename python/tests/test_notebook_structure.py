"""Structural tests for documentation notebooks."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import pytest

BASE_PATH = pathlib.Path(__file__).parent.parent.parent
AGB_EXPLAINED = BASE_PATH / "examples" / "articles" / "AGBExplained.ipynb"


def _source_lines(cell: dict[str, Any]) -> list[str]:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source.splitlines()

    lines: list[str] = []
    for part in source:
        lines.extend(str(part).splitlines())
    return lines


def _markdown_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]


def _level_two_headings(notebook: dict[str, Any]) -> list[str]:
    headings: list[str] = []
    for cell in _markdown_cells(notebook):
        for line in _source_lines(cell):
            if line.startswith("## "):
                headings.append(line.removeprefix("## ").strip())
    return headings


def _intro_toc_entries(notebook: dict[str, Any]) -> list[tuple[str, str]]:
    intro = next(cell for cell in _markdown_cells(notebook) if _source_lines(cell)[0].startswith("# "))
    entry_re = re.compile(r"^- \[(?P<label>[^\]]+)\]\(#(?P<anchor>[^)]+)\)")
    entries: list[tuple[str, str]] = []
    for line in _source_lines(intro):
        match = entry_re.match(line)
        if match is not None:
            entries.append((match.group("label"), match.group("anchor")))
    return entries


def _toc_anchor(title: str) -> str:
    return title.replace(" ", "-")


def test_agb_explained_intro_toc_matches_level_two_headings():
    notebook = json.loads(AGB_EXPLAINED.read_text(encoding="utf-8"))

    headings = _level_two_headings(notebook)
    entries = _intro_toc_entries(notebook)
    labels = [label for label, _ in entries]
    anchors = [anchor for _, anchor in entries]

    assert labels == headings
    assert anchors == [_toc_anchor(label) for label in labels]
    assert not any(re.match(r"\d+\.\s+", heading) for heading in headings)


@pytest.mark.parametrize(("missing_metadata", "predictor_warning"), [(False, False), (True, False), (True, True)])
def test_agb_quick_check_handles_missing_metadata_and_marks_only_actual_flags(
    monkeypatch,
    capsys,
    missing_metadata,
    predictor_warning,
):
    pytest.importorskip("great_tables")
    from pdstools import datasets
    from pdstools.adm.trees import ADMTreesModel

    trees = [
        {
            "score": 0.0,
            "split": f"Customer.A{i % 5} < 2",
            "gain": 1.0,
            "left": {
                "score": 0.0,
                "split": f"Customer.A{i % 5} < 1",
                "gain": 1.0,
                "left": {"score": 0.0},
                "right": {"score": 0.0},
            },
            "right": {"score": 0.0},
        }
        for i in range(10)
    ]
    if predictor_warning:
        trees[0]["split"] = "Customer.Code in { 1, 2, 3 }"
    properties = {} if missing_metadata else {"auc": 0.5, "trainingStats": {"positiveCount": 25, "negativeCount": 10}}
    model = ADMTreesModel(trees={"model": trees}, model=trees, properties=properties)
    monkeypatch.setattr(datasets, "sample_trees", lambda: model)
    notebook = json.loads(AGB_EXPLAINED.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}
    for cell in notebook["cells"]:
        if cell.get("id") in {"32174ba2", "quick-sanity-code"}:
            exec("\n".join(_source_lines(cell)), namespace)

    output = capsys.readouterr().out
    assert "ℹ️ Advisories (not red flags):" in output
    assert "  ℹ️ Cumulative gain:" in output
    assert "Cumulative gain: 50% reached at tree index 4 (5/10 trees)" in output
    leader = "Customer.A1" if predictor_warning else "Customer.A0"
    assert f"{leader} contributes 20.0% of total gain" in output
    if missing_metadata:
        assert "Pooled AUC=unavailable" in output
        assert "unavailable pos / unavailable neg" in output
        assert "No red flags from the available checks" in output
        assert "🚩" not in output
    else:
        assert "Pooled AUC=0.5000" in output
        assert "25 pos / 10 neg" in output
        assert "🚩 3 red flag(s):" in output
        assert output.count("  🚩 ") == 3
    if predictor_warning:
        assert "⚠️ Predictor diagnostics: 1 predictor flag(s) found" in output
        assert "  ⚠️ Customer.Code" in output
    else:
        assert "⚠️" not in output


def test_notebook_examples_do_not_pin_the_canonical_calendar():
    for path in sorted((BASE_PATH / "examples").rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            source = "\n".join(_source_lines(cell))
            assert not re.search(r"""\bend\s*=\s*["']canonical["']""", source), (
                f"{path.relative_to(BASE_PATH)}, cell {index}: omit end to use the most recent completed Sunday"
            )
