"""Check Pega documentation links in article sources."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLES_DIRECTORY = Path("examples/articles")
ARTICLE_SOURCE_SUFFIXES = {
    ".csv",
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".qmd",
    ".rst",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
DOCS_URL_PATTERN = re.compile(
    r"""https?://docs\.pega\.com[^\s"'<>\\)\]]+""",
    re.IGNORECASE,
)
CONTENT_API_BASE_URL = "https://docs-be.pega.com"
REQUEST_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class LocatedLink:
    url: str
    path: str
    line: int


def _extract_urls(text: str) -> list[str]:
    """Extract documentation URLs from source text."""
    return [match.rstrip(".,;:}") for match in DOCS_URL_PATTERN.findall(text)]


def _content_api_url(url: str) -> str:
    """Map a Pega article URL to its documentation content API endpoint.

    Raises
    ------
    ValueError
        If the URL is not a ``docs.pega.com`` bundle page.
    """
    parsed_url = urlsplit(url)
    if parsed_url.scheme.lower() not in {"http", "https"} or (parsed_url.hostname or "").lower() != "docs.pega.com":
        raise ValueError("expected an HTTP(S) URL on docs.pega.com")

    path = parsed_url.path.strip("/")
    path_parts = path.split("/")
    if (
        len(path_parts) < 4
        or path_parts[0] != "bundle"
        or path_parts[2] != "page"
        or not path_parts[1]
        or not any(path_parts[3:])
    ):
        raise ValueError("expected a /bundle/{bundle}/page/{article-path} URL")

    return f"{CONTENT_API_BASE_URL}/api/{path}"


def _scan_article_links() -> list[LocatedLink]:
    """Find Pega links in supported text files under examples/articles."""
    article_root = REPO_ROOT / ARTICLES_DIRECTORY
    if not article_root.is_dir():
        raise FileNotFoundError(f"Article directory not found: {article_root}")

    links: list[LocatedLink] = []
    for path in sorted(article_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ARTICLE_SOURCE_SUFFIXES:
            continue

        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            links.extend(LocatedLink(url=url, path=relative_path, line=line_number) for url in _extract_urls(line))

    return links


def _fetch_status(api_url: str) -> int:
    """Return the HTTP status from the Pega documentation content API."""
    request = Request(
        api_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "pdstools-docs-link-check",
        },
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status
    except HTTPError as error:
        error.close()
        return error.code


def _check_links(links: list[LocatedLink]) -> list[str]:
    """Return errors for article links that do not resolve through the API."""
    locations_by_api_url: dict[str, list[LocatedLink]] = {}
    errors: list[str] = []

    for link in links:
        try:
            api_url = _content_api_url(link.url)
        except ValueError as error:
            errors.append(f"{link.path}:{link.line}: {link.url} ({error})")
            continue
        locations_by_api_url.setdefault(api_url, []).append(link)

    for api_url, occurrences in locations_by_api_url.items():
        locations = ", ".join(f"{link.path}:{link.line}" for link in occurrences)
        try:
            status = _fetch_status(api_url)
        except (OSError, URLError) as error:
            errors.append(f"{locations}: unable to query {api_url}: {error}")
            continue

        if status != 200:
            urls = ", ".join(dict.fromkeys(link.url for link in occurrences))
            errors.append(f"{locations}: {urls} (content API returned HTTP {status}; expected 200)")

    return errors


def main() -> int:
    """Check every Pega documentation link under examples/articles."""
    try:
        links = _scan_article_links()
    except FileNotFoundError as error:
        print(f"Pega documentation link check could not run: {error}", file=sys.stderr)
        return 2

    if not links:
        print("No Pega documentation links to check.")
        return 0

    errors = _check_links(links)
    if errors:
        print("Pega documentation link check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    checked_pages = len({_content_api_url(link.url) for link in links})
    print(f"Checked {checked_pages} unique Pega documentation page(s); all resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
