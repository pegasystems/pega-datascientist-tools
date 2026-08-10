"""Structural tests for documentation notebooks."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

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
    intro = _markdown_cells(notebook)[0]
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
    notebook = json.loads(AGB_EXPLAINED.read_text())

    headings = _level_two_headings(notebook)
    entries = _intro_toc_entries(notebook)
    labels = [label for label, _ in entries]
    anchors = [anchor for _, anchor in entries]

    assert labels == headings
    assert anchors == [_toc_anchor(label) for label in labels]
    assert not any(re.match(r"\d+\.\s+", heading) for heading in headings)
