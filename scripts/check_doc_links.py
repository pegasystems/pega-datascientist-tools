"""Check external and relative links in documentation sources.

``docs.pega.com`` is a single-page app that answers HTTP 200 for any path,
so those links are verified through the Pega documentation content API.
Other external links are fetched directly, and relative links are resolved
against the repository checkout.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHSPECS = (
    ":(glob)*.md",
    ":(glob)docs/*.md",
    ":(glob)examples/**/*.ipynb",
    ":(glob)examples/**/*.md",
    ":(glob)examples/**/*.qmd",
    ":(glob)python/docs/**/*.md",
    ":(glob)python/docs/**/*.rst",
    ":(glob)python/pdstools/reports/**/*.qmd",
)
EXCLUDED_FILES = {"AGENTS.md", "CLAUDE.md"}
SKIPPED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "example.com", "agilestudio.pega.com"}

URL_PATTERN = re.compile(r"""https?://[^\s"'<>\\)\]`|]+""", re.IGNORECASE)
MARKDOWN_TARGET_PATTERN = re.compile(r"""\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)""")
HTML_TARGET_PATTERN = re.compile(r"""\b(?:href|src)\s*=\s*\\?["']([^"'\\]+)\\?["']""", re.IGNORECASE)
RST_TARGET_PATTERN = re.compile(r"""^\s*\.\.\s+(?:image|figure|include|literalinclude)::\s*(\S+)""")
SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
NOTEBOOK_CELL_TYPE_PATTERN = re.compile(r'^\s*"cell_type":\s*"(\w+)"')
CODE_COMMENT_PATTERN = re.compile(r"""(?:^|[\s"])#\s.*$""")
INLINE_CODE_PATTERN = re.compile(r"`[^`]*`")
RST_INLINE_LITERAL_PATTERN = re.compile(r"``[^`]*``")
RST_LITERAL_START_PATTERN = re.compile(r"^\s*\.\.\s+(?:code-block|code|sourcecode)::|^\s*(?!\.\.)\S.*::\s*$")

CONTENT_API_BASE_URL = "https://docs-be.pega.com"
PDSTOOLS_DOCS_PREFIX = "https://pegasystems.github.io/pega-datascientist-tools/latest/"
DOCS_MAKEFILE = Path("python/docs/Makefile")
REQUEST_TIMEOUT_SECONDS = 20
MAX_WORKERS = 8
RETRY_DELAY_SECONDS = 5
USER_AGENT = "Mozilla/5.0 (compatible; pdstools-doc-link-check)"


@dataclass(frozen=True)
class LocatedLink:
    url: str
    path: str
    line: int


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:}*'")


def _extract_urls(text: str) -> list[str]:
    """Extract absolute HTTP(S) URLs from source text."""
    return [_clean_url(match) for match in URL_PATTERN.findall(text)]


def _extract_relative_targets(text: str) -> list[str]:
    """Extract relative link and image targets from Markdown, HTML and RST."""
    targets = [
        *MARKDOWN_TARGET_PATTERN.findall(text),
        *HTML_TARGET_PATTERN.findall(text),
        *RST_TARGET_PATTERN.findall(text),
    ]
    return [target for target in targets if _is_checkable_relative_target(target)]


def _is_checkable_relative_target(target: str) -> bool:
    if not target or target.startswith(("#", "/")) or SCHEME_PATTERN.match(target):
        return False
    if any(character in target for character in "{}<>$"):
        return False
    # Links into the built Sphinx site (for example ``autoapi/.../index.html``)
    # have no counterpart in the checkout.
    return not target.split("#", 1)[0].endswith(".html")


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


def _doc_files() -> list[Path]:
    """Return tracked documentation sources, relative to the repository root."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *DOC_PATHSPECS],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return sorted(Path(name) for name in result.stdout.split("\0") if name and name not in EXCLUDED_FILES)


def _code_comment(line: str) -> str:
    """Return the trailing ``# comment`` of a code line, or an empty string."""
    match = CODE_COMMENT_PATTERN.search(line)
    return match.group(0) if match else ""


def _prose_lines(suffix: str, text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, line)`` pairs for prose, skipping code.

    Code (notebook code cells, fenced or literal blocks, inline code) often
    holds URL prefixes or placeholders rather than links meant for readers,
    so only ``# comments`` are kept from notebook cells and fenced blocks.
    """
    lines: list[tuple[int, str]] = []
    cell_type = None
    in_fence = False
    literal_indent: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if suffix == ".ipynb":
            match = NOTEBOOK_CELL_TYPE_PATTERN.match(line)
            if match:
                cell_type = match.group(1)
            if cell_type != "markdown":
                comment = _code_comment(line)
                if comment:
                    lines.append((line_number, comment))
                continue
            stripped = stripped.strip('"').replace("\\n", "").strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            comment = _code_comment(line)
            if comment:
                lines.append((line_number, comment))
            continue
        if suffix == ".rst":
            indent = len(line) - len(line.lstrip())
            if literal_indent is not None:
                if not stripped or indent > literal_indent:
                    continue
                literal_indent = None
            if RST_LITERAL_START_PATTERN.search(line):
                literal_indent = indent
            lines.append((line_number, RST_INLINE_LITERAL_PATTERN.sub("", line)))
        else:
            lines.append((line_number, INLINE_CODE_PATTERN.sub("", line)))
    return lines


def _scan_links(files: list[Path]) -> tuple[list[LocatedLink], list[LocatedLink]]:
    """Return ``(external_links, relative_links)`` found in ``files``."""
    external: list[LocatedLink] = []
    relative: list[LocatedLink] = []
    for file in files:
        path = file.as_posix()
        text = (REPO_ROOT / file).read_text(encoding="utf-8")
        for line_number, line in _prose_lines(file.suffix.lower(), text):
            for url in _extract_urls(line):
                if (urlsplit(url).hostname or "").lower() not in SKIPPED_HOSTS:
                    external.append(LocatedLink(url=url, path=path, line=line_number))
            relative.extend(
                LocatedLink(url=target, path=path, line=line_number) for target in _extract_relative_targets(line)
            )
    return external, relative


def _docs_article_notebooks() -> set[Path]:
    """Return the notebooks that ``python/docs/Makefile`` copies into ``articles``.

    The Sphinx build flattens a curated set of example notebooks into
    ``python/docs/source/articles``; only those become ``articles/<name>.html``.
    Returns an empty set if the copy command cannot be found, which makes the
    notebook fallbacks strict rather than lenient.
    """
    makefile = REPO_ROOT / DOCS_MAKEFILE
    if not makefile.is_file():
        return set()
    notebooks: set[Path] = set()
    for line in makefile.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "cp" or tokens[-1] != "source/articles":
            continue
        for pattern in tokens[1:-1]:
            repo_pattern = Path(os.path.normpath(DOCS_MAKEFILE.parent / pattern))
            if repo_pattern.is_absolute() or repo_pattern.parts[:1] == ("..",):
                continue
            notebooks.update(
                source.relative_to(REPO_ROOT)
                for source in REPO_ROOT.glob(repo_pattern.as_posix())
                if source.suffix == ".ipynb" and source.is_file()
            )
    return notebooks


def _check_relative_links(links: list[LocatedLink], article_notebooks: set[Path]) -> list[str]:
    """Return errors for relative links that do not exist in the checkout.

    The Sphinx build copies ``article_notebooks`` into one ``articles``
    folder, so one of them may link to another by bare file name.
    """
    article_names = {notebook.name for notebook in article_notebooks}
    errors = []
    for link in links:
        target = unquote(link.url.split("#", 1)[0].split("?", 1)[0])
        if (REPO_ROOT / link.path).parent.joinpath(target).exists():
            continue
        if "/" not in target and Path(link.path) in article_notebooks and target in article_names:
            continue
        errors.append(f"{link.path}:{link.line}: {link.url} (file not found)")
    return errors


def _fetch_status(url: str, method: str = "GET") -> int:
    """Return the HTTP status for ``url`` after following redirects."""
    request = Request(
        url,
        method=method,
        headers={"Accept": "*/*", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status
    except HTTPError as error:
        error.close()
        return error.code


def _url_status(url: str) -> int:
    """Return the status used to judge ``url``, retrying transient failures once."""
    target = _content_api_url(url) if (urlsplit(url).hostname or "").lower() == "docs.pega.com" else url
    for attempt in range(2):
        try:
            status = _fetch_status(target, method="HEAD")
            if status in {403, 404, 405, 501}:
                # Some servers reject HEAD but answer GET normally.
                status = _fetch_status(target)
        except (OSError, URLError):
            if attempt:
                raise
            status = 0
        if status not in {0, 429} and status < 500:
            return status
        if not attempt:
            time.sleep(RETRY_DELAY_SECONDS)
    return status


def _pdstools_docs_source_exists(url: str, article_notebooks: set[Path]) -> bool:
    """Return whether a pdstools docs page is built from a source in the checkout.

    Pages added in the same change are not deployed yet, so a 404 on the live
    site is accepted when the source that generates the page exists.
    ``articles/<name>.html`` comes from one of ``article_notebooks``; other
    pages come from ``python/docs/source/<page>.rst``.
    """
    if not url.startswith(PDSTOOLS_DOCS_PREFIX):
        return False
    page = urlsplit(url[len(PDSTOOLS_DOCS_PREFIX) :]).path
    if not page.endswith(".html"):
        return False
    page = page.removesuffix(".html")
    if page.startswith("articles/") and "/" not in page.removeprefix("articles/"):
        stem = page.removeprefix("articles/")
        return any(notebook.stem == stem for notebook in article_notebooks)
    return (REPO_ROOT / "python/docs/source" / f"{page}.rst").is_file()


def _check_external_links(links: list[LocatedLink], article_notebooks: set[Path]) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for external links.

    Rate-limited responses (HTTP 429) are warnings so that a busy remote host
    does not fail the build.
    """
    locations_by_url: dict[str, list[LocatedLink]] = {}
    errors: list[str] = []
    for link in links:
        if (urlsplit(link.url).hostname or "").lower() == "docs.pega.com":
            try:
                _content_api_url(link.url)
            except ValueError as error:
                errors.append(f"{link.path}:{link.line}: {link.url} ({error})")
                continue
        locations_by_url.setdefault(link.url.split("#", 1)[0], []).append(link)

    def check(url: str) -> tuple[str, int | str]:
        try:
            return url, _url_status(url)
        except (OSError, URLError, ValueError) as error:
            return url, str(error)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(check, locations_by_url))

    warnings: list[str] = []
    for url, status in results:
        locations = ", ".join(f"{link.path}:{link.line}" for link in locations_by_url[url])
        if status == 429:
            warnings.append(f"{locations}: {url} (rate limited, HTTP 429)")
        elif isinstance(status, str):
            errors.append(f"{locations}: {url} (request failed: {status})")
        elif status == 404 and _pdstools_docs_source_exists(url, article_notebooks):
            continue
        elif status >= 400:
            errors.append(f"{locations}: {url} (HTTP {status})")
    return errors, warnings


def main() -> int:
    """Check every external and relative link in the documentation sources."""
    files = _doc_files()
    external, relative = _scan_links(files)
    article_notebooks = _docs_article_notebooks()

    errors = _check_relative_links(relative, article_notebooks)
    external_errors, warnings = _check_external_links(external, article_notebooks)
    errors.extend(external_errors)

    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        print("Documentation link check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    unique_urls = len({link.url.split("#", 1)[0] for link in external})
    print(
        f"Checked {unique_urls} unique external URL(s) and {len(relative)} relative link(s) "
        f"in {len(files)} file(s); all resolved."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
