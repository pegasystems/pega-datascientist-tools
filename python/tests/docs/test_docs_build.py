"""Smoke test for the docs build, focused on the version-switcher template.

We build a tiny standalone Sphinx project (no autoapi, no nbsphinx) that
points at the real ``_templates`` and ``_static`` directories, so the
test stays fast and only exercises what we care about: the switcher
template renders into every page and references the versions.json
fetcher.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DOCS_ROOT = Path(__file__).resolve().parents[3] / "python" / "docs"
TEMPLATES = DOCS_ROOT / "source" / "_templates"
STATIC = DOCS_ROOT / "source" / "_static"

pytest.importorskip("furo")
pytest.importorskip("sphinx")


CONF_PY = """\
import os, sys
sys.path.insert(0, os.path.abspath("."))

project = "pdstools-test"
author = "tests"
release = "0.0.0"

extensions = []
html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["version-switcher.css"]
html_js_files = ["version-switcher.js"]

docs_version = os.environ.get("PDSTOOLS_DOCS_VERSION", "latest")
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

Test page.
"""


def test_version_switcher_renders(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "conf.py").write_text(CONF_PY)
    (src / "index.rst").write_text(INDEX_RST)
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
    assert "version-switcher" in index_html
    assert 'data-current-version="9.9.9"' in index_html
    assert "version-switcher.js" in index_html
    assert "version-switcher.css" in index_html

    js = (out / "_static" / "version-switcher.js").read_text()
    assert "versions.json" in js
