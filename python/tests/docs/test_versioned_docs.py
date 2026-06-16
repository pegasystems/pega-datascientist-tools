"""Tests for the versioned docs publishing helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "docs" / "versioned_docs.py"
MODULE_SPEC = importlib.util.spec_from_file_location("versioned_docs", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
versioned_docs = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(versioned_docs)


def test_normalize_version_slug_strips_tag_prefix() -> None:
    assert versioned_docs.normalize_version_slug("tag", "V5.1.0") == "5.1.0"
    assert versioned_docs.normalize_version_slug("tag", "v5.1.0") == "5.1.0"
    assert versioned_docs.normalize_version_slug("branch", "master") == "dev"


def test_build_manifest_prefers_latest_pypi_release(tmp_path: Path) -> None:
    for version in ["dev", "4.7.0", "4.7.1", "4.8.0rc1"]:
        (tmp_path / version).mkdir()

    manifest, preferred_version = versioned_docs.build_manifest(
        tmp_path,
        "https://example.com/pdstools",
        pypi_version="4.7.1",
    )

    assert preferred_version == "4.7.1"
    assert manifest[0]["version"] == "dev"
    stable_entry = next(entry for entry in manifest if entry["version"] == "4.7.1")
    assert stable_entry["preferred"] is True
    assert stable_entry["url"] == "https://example.com/pdstools/4.7.1/"


def test_build_manifest_falls_back_to_highest_release(tmp_path: Path) -> None:
    for version in ["dev", "4.7.0", "4.7.1"]:
        (tmp_path / version).mkdir()

    manifest, preferred_version = versioned_docs.build_manifest(
        tmp_path,
        "https://example.com/pdstools",
        pypi_version="4.6.9",
    )

    assert preferred_version == "4.7.1"
    assert any(entry["preferred"] for entry in manifest)


def test_write_root_redirect_targets_preferred_release(tmp_path: Path) -> None:
    versioned_docs.write_root_redirect(tmp_path, "4.7.1")

    index_html = (tmp_path / "index.html").read_text()
    assert "url=4.7.1/" in index_html
    assert 'href="4.7.1/"' in index_html


def test_remove_legacy_site_content_prunes_old_top_level_entries(tmp_path: Path) -> None:
    for dirname in versioned_docs.LEGACY_SITE_DIRS:
        (tmp_path / dirname).mkdir()
    for filename in versioned_docs.LEGACY_SITE_FILES:
        (tmp_path / filename).write_text("legacy")
    (tmp_path / "dev").mkdir()
    (tmp_path / "4.7.1").mkdir()

    versioned_docs.remove_legacy_site_content(tmp_path)

    for dirname in versioned_docs.LEGACY_SITE_DIRS:
        assert not (tmp_path / dirname).exists()
    for filename in versioned_docs.LEGACY_SITE_FILES:
        assert not (tmp_path / filename).exists()
    assert (tmp_path / "dev").is_dir()
    assert (tmp_path / "4.7.1").is_dir()
