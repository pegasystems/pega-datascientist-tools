from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_doc_links.py"
SPEC = importlib.util.spec_from_file_location("check_doc_links", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load the checker at {SCRIPT_PATH}")
check_doc_links = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_doc_links
SPEC.loader.exec_module(check_doc_links)

LocatedLink = check_doc_links.LocatedLink


def test_content_api_url_maps_bundle_page_and_ignores_query() -> None:
    assert check_doc_links._content_api_url(
        "https://docs.pega.com/bundle/platform/page/platform/decision-management/example.html?language=en-US#details"
    ) == ("https://docs-be.pega.com/api/bundle/platform/page/platform/decision-management/example.html")


def test_content_api_url_rejects_non_article_paths() -> None:
    with pytest.raises(ValueError) as error:
        check_doc_links._content_api_url("https://docs.pega.com/bundle/platform/page")
    assert str(error.value) == "expected a /bundle/{bundle}/page/{article-path} URL"


def test_extract_relative_targets_skips_urls_anchors_and_built_pages() -> None:
    line = (
        "[a](guide.md#setup) [b](https://x.org) [c](#top) [d](mailto:a@b.c) [g](img/chart_(v2).png) "
        '<img src="img/logo.png"> [e](autoapi/pkg/index.html) [f](/abs.md)'
    )
    assert check_doc_links._extract_relative_targets(line) == ["guide.md#setup", "img/chart_(v2).png", "img/logo.png"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "[Python](https://en.wikipedia.org/wiki/Python_(programming_language))",
            ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
        ),
        (
            "See https://en.wikipedia.org/wiki/Python_(programming_language).",
            ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
        ),
        ("(see https://a.org/x).", ["https://a.org/x"]),
        ("(see https://a.org/x_(y)), then", ["https://a.org/x_(y)"]),
        ("[a](https://a.org/one), [b](https://a.org/two).", ["https://a.org/one", "https://a.org/two"]),
        ('"[a](https://a.org/nb)\\n",', ["https://a.org/nb"]),
    ],
)
def test_extract_urls_handles_parentheses_and_punctuation(text, expected) -> None:
    assert check_doc_links._extract_urls(text) == expected


def _notebook(*cells: tuple[str, list[str]]) -> str:
    lines = []
    for cell_type, source in cells:
        lines.append(f'  "cell_type": "{cell_type}",')
        lines.extend(f'    "{line}\\n",' for line in source)
    return "\n".join(lines)


@pytest.mark.parametrize(
    ("suffix", "text", "expected"),
    [
        pytest.param(
            ".md",
            "Read https://a.org/one and `https://host/x`.\n```bash\ncurl https://a.org/code\n# https://a.org/comment\n```\n",
            ["https://a.org/one", "https://a.org/comment"],
            id="md-inline-code-and-fence-comment",
        ),
        pytest.param(
            ".md",
            "````md\n```python\nhttps://a.org/inner\n```\nhttps://a.org/still-code\n````\nhttps://a.org/after\n",
            ["https://a.org/after"],
            id="md-longer-outer-fence",
        ),
        pytest.param(
            ".md",
            "```\n~~~\nhttps://a.org/code\n~~~\n```\nhttps://a.org/after\n",
            ["https://a.org/after"],
            id="md-tilde-inside-backtick",
        ),
        pytest.param(
            ".md",
            "```\n```python\nhttps://a.org/code\n```\nhttps://a.org/after\n",
            ["https://a.org/after"],
            id="md-info-string-does-not-close",
        ),
        pytest.param(".md", "```\nhttps://a.org/unclosed\n", [], id="md-unclosed-fence"),
        pytest.param(
            ".rst",
            "Example::\n\n    https://a.org/literal\n\nSee `docs <https://a.org/three>`_ and ``https://a.org/lit``.\n",
            ["https://a.org/three"],
            id="rst-literal-block-and-inline-literal",
        ),
        pytest.param(
            ".ipynb",
            _notebook(
                ("markdown", ["See https://a.org/doc."]),
                ("code", ['path = \\"https://a.org/prefix\\"', "# See: https://a.org/commented"]),
            ),
            ["https://a.org/doc", "https://a.org/commented"],
            id="ipynb-code-cell-keeps-comments",
        ),
        pytest.param(
            ".ipynb",
            _notebook(
                ("markdown", ["```", "https://a.org/code", "```", "https://a.org/after", "```unclosed"]),
                ("markdown", ["https://a.org/next-cell"]),
            ),
            ["https://a.org/after", "https://a.org/next-cell"],
            id="ipynb-fences-reset-per-cell",
        ),
    ],
)
def test_prose_lines_skips_code(suffix, text, expected) -> None:
    urls = [
        url for _, line in check_doc_links._prose_lines(suffix, text) for url in check_doc_links._extract_urls(line)
    ]
    assert urls == expected


def test_scan_links_reports_locations_and_skips_local_hosts(tmp_path, monkeypatch) -> None:
    (tmp_path / "README.md").write_text(
        "intro\n[Guide](https://docs.pega.com/bundle/p/page/one.html).\nhttp://localhost:8080 [x](CONTRIBUTING.md)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_doc_links, "REPO_ROOT", tmp_path)

    assert check_doc_links._scan_links([Path("README.md")]) == (
        [LocatedLink(url="https://docs.pega.com/bundle/p/page/one.html", path="README.md", line=2)],
        [LocatedLink(url="CONTRIBUTING.md", path="README.md", line=3)],
    )


def test_docs_article_notebooks_follows_makefile_copy(tmp_path, monkeypatch) -> None:
    for notebook in ["articles/a.ipynb", "articles/nested/b.ipynb", "vf/c.ipynb", "other/d.ipynb"]:
        (tmp_path / "examples" / notebook).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "examples" / notebook).write_text("{}", encoding="utf-8")
    (tmp_path / "examples" / "vf" / "image.png").write_bytes(b"")
    (tmp_path / "python" / "docs").mkdir(parents=True)
    (tmp_path / "python" / "docs" / "Makefile").write_text(
        "%: Makefile\n\tmkdir -p source/articles\n"
        "\tcp ../../examples/articles/*.ipynb ../../examples/vf/* ../../examples/articles/nested/b.ipynb source/articles\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_doc_links, "REPO_ROOT", tmp_path)

    assert check_doc_links._docs_article_notebooks() == {
        Path("examples/articles/a.ipynb"),
        Path("examples/articles/nested/b.ipynb"),
        Path("examples/vf/c.ipynb"),
    }


def test_check_relative_links(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    for directory in ["a", "b", "c"]:
        (repo_root / "examples" / directory).mkdir(parents=True)
    (repo_root / "examples" / "a" / "img.png").write_bytes(b"")
    (tmp_path / "outside.txt").write_text("outside repository", encoding="utf-8")
    monkeypatch.setattr(check_doc_links, "REPO_ROOT", repo_root)
    article_notebooks = {Path("examples/a/one.ipynb"), Path("examples/b/two.ipynb")}
    links = [
        LocatedLink(url="img.png#x", path="examples/a/one.ipynb", line=1),
        LocatedLink(url="two.ipynb", path="examples/a/one.ipynb", line=2),
        LocatedLink(url="missing.png", path="examples/a/one.ipynb", line=3),
        LocatedLink(url="two.ipynb", path="examples/c/uncopied.ipynb", line=4),
        LocatedLink(url="three.ipynb", path="examples/a/one.ipynb", line=5),
        LocatedLink(url="../../../outside.txt", path="examples/a/one.ipynb", line=6),
    ]

    assert check_doc_links._check_relative_links(links, article_notebooks) == [
        "examples/a/one.ipynb:3: missing.png (file not found)",
        "examples/c/uncopied.ipynb:4: two.ipynb (file not found)",
        "examples/a/one.ipynb:5: three.ipynb (file not found)",
        "examples/a/one.ipynb:6: ../../../outside.txt (target is outside repository)",
    ]


def test_check_external_links_deduplicates_and_classifies(monkeypatch) -> None:
    stale = "https://docs.pega.com/bundle/alerts/page/old.html"
    statuses = {
        stale: 404,
        "https://github.com/org/repo": 200,
        "https://busy.example.org/": 429,
        "https://redirect.example.org/": 302,
    }
    monkeypatch.setattr(check_doc_links, "_url_status", statuses.__getitem__)

    links = [
        LocatedLink(url=stale, path="a.ipynb", line=1),
        LocatedLink(url="https://github.com/org/repo#readme", path="a.ipynb", line=2),
        LocatedLink(url=stale, path="b.ipynb", line=7),
        LocatedLink(url="https://busy.example.org/", path="b.ipynb", line=8),
        LocatedLink(url="https://docs.pega.com/bundle/x", path="c.md", line=3),
        LocatedLink(url="https://redirect.example.org/", path="d.md", line=9),
    ]

    assert check_doc_links._check_external_links(links, set()) == (
        [
            "c.md:3: https://docs.pega.com/bundle/x (expected a /bundle/{bundle}/page/{article-path} URL)",
            f"a.ipynb:1, b.ipynb:7: {stale} (HTTP 404)",
            "d.md:9: https://redirect.example.org/ (HTTP 302)",
        ],
        ["b.ipynb:8: https://busy.example.org/ (rate limited, HTTP 429)"],
    )


def test_pdstools_docs_source_exists_maps_pages_to_sources(tmp_path, monkeypatch) -> None:
    (tmp_path / "python" / "docs" / "source").mkdir(parents=True)
    (tmp_path / "python" / "docs" / "source" / "GettingStarted.rst").write_text("", encoding="utf-8")
    monkeypatch.setattr(check_doc_links, "REPO_ROOT", tmp_path)
    article_notebooks = {Path("examples/articles/AGBExplained.ipynb")}
    base = "https://pegasystems.github.io/pega-datascientist-tools/latest/"
    exists = check_doc_links._pdstools_docs_source_exists

    assert exists(f"{base}articles/AGBExplained.html", article_notebooks)
    assert exists(f"{base}GettingStarted.html", article_notebooks)
    assert not exists(f"{base}articles/ONNX_PyTorch_Example.html", article_notebooks)
    assert not exists(f"{base}Missing.html", article_notebooks)
    assert not exists(f"{base}autoapi/pdstools/index.html", article_notebooks)
    assert not exists(f"{base}../../../python/docs/source/GettingStarted.html", article_notebooks)
    assert not exists(f"{base}./GettingStarted.html", article_notebooks)
    assert not exists(f"{base}/GettingStarted.html", article_notebooks)
    assert not exists(
        "https://pegasystems.github.io/pega-datascientist-tools/Python/articles/AGBExplained.html", article_notebooks
    )


def test_check_external_links_accepts_undeployed_pdstools_page(monkeypatch) -> None:
    new_page = "https://pegasystems.github.io/pega-datascientist-tools/latest/articles/New.html"
    removed_page = "https://pegasystems.github.io/pega-datascientist-tools/latest/articles/Gone.html"
    monkeypatch.setattr(check_doc_links, "_url_status", lambda url: 404)
    links = [
        LocatedLink(url=new_page, path="README.md", line=1),
        LocatedLink(url=removed_page, path="README.md", line=2),
    ]

    assert check_doc_links._check_external_links(links, {Path("examples/new/New.ipynb")}) == (
        [f"README.md:2: {removed_page} (HTTP 404)"],
        [],
    )


def test_url_status_uses_content_api_and_falls_back_to_get(monkeypatch) -> None:
    calls = []

    def fake_fetch(url, method="GET"):
        calls.append((method, url))
        return 405 if method == "HEAD" else 200

    monkeypatch.setattr(check_doc_links, "_fetch_status", fake_fetch)

    assert check_doc_links._url_status("https://docs.pega.com/bundle/p/page/a.html") == 200
    assert calls == [
        ("HEAD", "https://docs-be.pega.com/api/bundle/p/page/a.html"),
        ("GET", "https://docs-be.pega.com/api/bundle/p/page/a.html"),
    ]


def test_url_status_retries_server_errors_once(monkeypatch) -> None:
    statuses = iter([503, 200])
    monkeypatch.setattr(check_doc_links, "_fetch_status", lambda url, method="GET": next(statuses))
    monkeypatch.setattr(check_doc_links, "RETRY_DELAY_SECONDS", 0)

    assert check_doc_links._url_status("https://github.com/org/repo") == 200


def test_fetch_status_returns_success(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(check_doc_links, "urlopen", lambda *args, **kwargs: Response())

    assert check_doc_links._fetch_status("https://github.com/org/repo") == 200


def test_fetch_status_returns_http_error_status(monkeypatch) -> None:
    def raise_not_found(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=BytesIO(b'{"error_code":404}'))

    monkeypatch.setattr(check_doc_links, "urlopen", raise_not_found)

    assert check_doc_links._fetch_status("https://docs-be.pega.com/api/bundle/alerts/page/example.html", "HEAD") == 404
