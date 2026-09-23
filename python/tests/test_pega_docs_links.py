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


def test_extract_added_links_ignores_removed_links_and_tracks_locations() -> None:
    diff_text = """\
diff --git a/examples/articles/example.ipynb b/examples/articles/example.ipynb
index 1111111..2222222 100644
--- a/examples/articles/example.ipynb
+++ b/examples/articles/example.ipynb
@@ -10 +10,2 @@
- "https://docs.pega.com/bundle/alerts/page/platform/decision-management/old.html"
+ "[Guide](https://docs.pega.com/bundle/platform/page/platform/decision-management/one.html)."
+ "https://docs.pega.com/bundle/platform/page/platform/decision-management/two.html"
"""

    assert check_pega_docs_links._extract_added_links(diff_text) == [
        check_pega_docs_links.LocatedLink(
            url="https://docs.pega.com/bundle/platform/page/platform/decision-management/one.html",
            path="examples/articles/example.ipynb",
            line=10,
        ),
        check_pega_docs_links.LocatedLink(
            url="https://docs.pega.com/bundle/platform/page/platform/decision-management/two.html",
            path="examples/articles/example.ipynb",
            line=11,
        ),
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
