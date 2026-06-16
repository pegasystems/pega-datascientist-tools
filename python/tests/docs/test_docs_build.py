"""Smoke tests for the docs version switcher assets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
TEMPLATES = DOCS_ROOT / "source" / "_templates"
STATIC = DOCS_ROOT / "source" / "_static"

pytest.importorskip("furo")
pytest.importorskip("sphinx")


CONF_PY = """\
import os

project = "pdstools-test"
author = "tests"
release = "0.0.0"

extensions = []
html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["version-switcher.css"]
html_js_files = ["version-switcher.js"]

docs_version = os.environ.get("PDSTOOLS_DOCS_VERSION", "dev")
html_context = {"docs_version": docs_version}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/version-switcher.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
"""

INDEX_RST = """\
Hello
=====

.. toctree::
   :maxdepth: 1

   nested/page
"""

NESTED_RST = """\
Nested
======

Test page.
"""


def test_version_switcher_renders_on_nested_pages(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    nested = src / "nested"
    src.mkdir()
    nested.mkdir()

    (src / "conf.py").write_text(CONF_PY)
    (src / "index.rst").write_text(INDEX_RST)
    (nested / "page.rst").write_text(NESTED_RST)
    shutil.copytree(TEMPLATES, src / "_templates")
    shutil.copytree(STATIC, src / "_static")

    env = {**os.environ, "PDSTOOLS_DOCS_VERSION": "9.9.9"}
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-q", str(src), str(out)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"sphinx failed:\n{result.stdout}\n{result.stderr}"

    index_html = (out / "index.html").read_text()
    nested_html = (out / "nested" / "page.html").read_text()
    assert 'data-current-version="9.9.9"' in index_html
    assert 'data-current-version="9.9.9"' in nested_html
    assert "version-switcher.js" in nested_html
    assert "version-switcher.css" in nested_html

    js = (out / "_static" / "version-switcher.js").read_text()
    assert "versions.json" in js
    assert "../versions.json" not in js
    assert "pathname.indexOf(marker)" in js
