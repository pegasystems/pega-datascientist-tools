from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_pega_docs_links.py"
SPEC = importlib.util.spec_from_file_location("check_pega_docs_links", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load the checker at {SCRIPT_PATH}")
check_pega_docs_links = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_pega_docs_links
SPEC.loader.exec_module(check_pega_docs_links)


def test_content_api_url_maps_bundle_page_and_ignores_query() -> None:
    assert check_pega_docs_links._content_api_url(
        "https://docs.pega.com/bundle/platform/page/platform/decision-management/example.html?language=en-US#details"
    ) == ("https://docs-be.pega.com/api/bundle/platform/page/platform/decision-management/example.html")


def test_content_api_url_rejects_non_article_paths() -> None:
    with pytest.raises(ValueError) as error:
        check_pega_docs_links._content_api_url("https://docs.pega.com/bundle/platform/page")
    assert str(error.value) == "expected a /bundle/{bundle}/page/{article-path} URL"


def test_scan_article_links_finds_links_with_locations(tmp_path, monkeypatch) -> None:
    articles = tmp_path / "examples" / "articles" / "nested"
    articles.mkdir(parents=True)
    (articles / "example.ipynb").write_text(
        '"intro"\n'
        '"[Guide](https://docs.pega.com/bundle/platform/page/platform/decision-management/one.html)."\n'
        '"See https://docs.pega.com/bundle/platform/page/platform/decision-management/two.html"\n',
        encoding="utf-8",
    )
    (articles / "image.png").write_bytes(b"https://docs.pega.com/bundle/platform/page/ignored.html")
    monkeypatch.setattr(check_pega_docs_links, "REPO_ROOT", tmp_path)

    assert check_pega_docs_links._scan_article_links() == [
        check_pega_docs_links.LocatedLink(
            url="https://docs.pega.com/bundle/platform/page/platform/decision-management/one.html",
            path="examples/articles/nested/example.ipynb",
            line=2,
        ),
        check_pega_docs_links.LocatedLink(
            url="https://docs.pega.com/bundle/platform/page/platform/decision-management/two.html",
            path="examples/articles/nested/example.ipynb",
            line=3,
        ),
    ]


def test_check_links_reports_each_stale_page_once(monkeypatch) -> None:
    stale = "https://docs.pega.com/bundle/alerts/page/platform/decision-management/old.html"
    live = "https://docs.pega.com/bundle/platform/page/platform/decision-management/live.html"
    statuses = {
        check_pega_docs_links._content_api_url(stale): 404,
        check_pega_docs_links._content_api_url(live): 200,
    }
    monkeypatch.setattr(check_pega_docs_links, "_fetch_status", statuses.__getitem__)

    links = [
        check_pega_docs_links.LocatedLink(url=stale, path="a.ipynb", line=1),
        check_pega_docs_links.LocatedLink(url=live, path="a.ipynb", line=2),
        check_pega_docs_links.LocatedLink(url=stale, path="b.ipynb", line=7),
    ]

    assert check_pega_docs_links._check_links(links) == [
        f"a.ipynb:1, b.ipynb:7: {stale} (content API returned HTTP 404; expected 200)",
    ]


def test_fetch_status_returns_success(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(check_pega_docs_links, "urlopen", lambda *args, **kwargs: Response())

    assert check_pega_docs_links._fetch_status("https://docs-be.pega.com/api/bundle/platform/page/example.html") == 200


def test_fetch_status_returns_not_found_status(monkeypatch) -> None:
    def raise_not_found(request, timeout):
        raise HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=BytesIO(b'{"error_code":404}'),
        )

    monkeypatch.setattr(check_pega_docs_links, "urlopen", raise_not_found)

    assert check_pega_docs_links._fetch_status("https://docs-be.pega.com/api/bundle/alerts/page/example.html") == 404
