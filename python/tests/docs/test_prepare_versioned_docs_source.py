"""Tests for preparing historical docs sources with current scaffolding."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "docs" / "prepare_versioned_docs_source.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "prepare_versioned_docs_source",
    MODULE_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load docs helper module from {MODULE_PATH}")
prepare_docs_source = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(prepare_docs_source)


def test_overlay_current_docs_scaffolding_copies_expected_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    build_root = tmp_path / "build"

    for relative_path in prepare_docs_source.SCAFFOLD_PATHS:
        source = repo_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative_path.as_posix())

    prepare_docs_source.overlay_current_docs_scaffolding(repo_root, build_root)

    for relative_path in prepare_docs_source.SCAFFOLD_PATHS:
        target = build_root / relative_path
        assert target.exists()
        assert target.read_text() == relative_path.as_posix()
