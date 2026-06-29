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


def test_build_manifest_ignores_non_version_dirs_and_collapses_prereleases(
    tmp_path: Path,
) -> None:
    for version in [
        "dev",
        "4.7.0",
        "4.7.1",
        "5.0.0a1",
        "5.0.0b2",
        "5.0.0rc2",
        "articles",
        "autoapi",
        "latest",
    ]:
        (tmp_path / version).mkdir()

    manifest, preferred_version = versioned_docs.build_manifest(
        tmp_path,
        "https://example.com/pdstools",
        pypi_version="4.7.1",
    )

    assert preferred_version == "4.7.1"
    assert [entry["version"] for entry in manifest] == ["dev", "4.7.1", "4.7.0", "5.0.0rc2"]


def test_build_manifest_hides_prerelease_once_stable_exists(tmp_path: Path) -> None:
    for version in ["dev", "4.7.1", "5.0.0rc2", "5.0.0"]:
        (tmp_path / version).mkdir()

    manifest, preferred_version = versioned_docs.build_manifest(
        tmp_path,
        "https://example.com/pdstools",
        pypi_version="5.0.0rc2",
    )

    assert preferred_version == "5.0.0"
    assert [entry["version"] for entry in manifest] == ["dev", "5.0.0", "4.7.1"]


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


def test_build_manifest_falls_back_to_latest_prerelease_when_no_stable_exists(
    tmp_path: Path,
) -> None:
    for version in ["dev", "5.0.0a1", "5.0.0rc2", "5.1.0b1"]:
        (tmp_path / version).mkdir()

    manifest, preferred_version = versioned_docs.build_manifest(
        tmp_path,
        "https://example.com/pdstools",
        pypi_version="5.0.0rc2",
    )

    assert preferred_version == "5.1.0b1"
    assert [entry["version"] for entry in manifest] == ["dev", "5.1.0b1", "5.0.0rc2"]


def test_write_root_redirect_targets_preferred_release(tmp_path: Path) -> None:
    versioned_docs.write_root_redirect(tmp_path, "4.7.1")

    index_html = (tmp_path / "index.html").read_text()
    assert "url=4.7.1/" in index_html
    assert 'href="4.7.1/"' in index_html


def test_write_latest_aliases_creates_latest_and_legacy_page_redirects(
    tmp_path: Path,
) -> None:
    version_root = tmp_path / "4.7.1"
    (version_root / "articles").mkdir(parents=True)
    (version_root / "autoapi").mkdir(parents=True)
    (version_root / "index.html").write_text("release home")
    (version_root / "articles" / "Example_ADM_Analysis.html").write_text("article")
    (version_root / "autoapi" / "index.html").write_text("api")

    versioned_docs.write_latest_aliases(
        tmp_path,
        "https://example.com/pdstools",
        "4.7.1",
    )

    latest_article = tmp_path / "latest" / "articles" / "Example_ADM_Analysis.html"
    legacy_article = tmp_path / "articles" / "Example_ADM_Analysis.html"
    latest_home = tmp_path / "latest" / "index.html"
    legacy_api = tmp_path / "autoapi" / "index.html"

    assert "url=https://example.com/pdstools/4.7.1/articles/Example_ADM_Analysis.html" in (latest_article.read_text())
    assert "url=https://example.com/pdstools/4.7.1/articles/Example_ADM_Analysis.html" in (legacy_article.read_text())
    assert "url=https://example.com/pdstools/4.7.1/index.html" in latest_home.read_text()
    assert "url=https://example.com/pdstools/4.7.1/autoapi/index.html" in (legacy_api.read_text())
