"""HTML post-processing: local asset inlining, zip bundling, error scanning."""

from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import zipfile
from pathlib import Path

from ._common import logger

_LINK_STYLESHEET_RE = re.compile(
    r"<link\b[^>]*\brel=[\"']stylesheet[\"'][^>]*>",
    re.IGNORECASE,
)
_HREF_ATTR_RE = re.compile(r"\bhref=[\"']([^\"']+)[\"']", re.IGNORECASE)
_SCRIPT_SRC_RE = re.compile(
    r"<script\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>\s*</script>",
    re.IGNORECASE,
)
_CSS_URL_RE = re.compile(r"url\(\s*([\"']?)([^\"')]+)\1\s*\)", re.IGNORECASE)

# References we never try to localise: they are already remote or self-contained.
_REMOTE_PREFIXES = ("http://", "https://", "//", "data:", "#", "about:")


def _is_remote(reference: str) -> bool:
    """Whether a URL reference points somewhere other than the local filesystem."""
    return reference.strip().startswith(_REMOTE_PREFIXES)


def _as_data_uri(path: Path) -> str:
    """Encode a binary asset (font, image) as a base64 ``data:`` URI."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _inline_css_urls(css: str, css_dir: Path) -> str:
    """Rewrite relative ``url(...)`` references in a stylesheet to data URIs.

    Stylesheets pull in fonts and images by relative path. Once the CSS text is
    lifted into a ``<style>`` block those paths break, so the referenced assets
    are embedded directly. Remote and already-inline references are untouched,
    as are references that do not resolve to a file on disk.

    Parameters
    ----------
    css : str
        Stylesheet text.
    css_dir : Path
        Directory of the stylesheet, used to resolve relative references.

    Returns
    -------
    str
        The stylesheet with resolvable local references replaced by data URIs.
    """

    def _replace(match: re.Match) -> str:
        reference = match.group(2).strip()
        if _is_remote(reference):
            return match.group(0)
        # Strip cache-busting query strings and fragments: "font.woff?v=2#iefix".
        asset_path = (css_dir / reference.split("?")[0].split("#")[0]).resolve()
        if not asset_path.is_file():
            return match.group(0)
        return f'url("{_as_data_uri(asset_path)}")'

    return _CSS_URL_RE.sub(_replace, css)


def _inline_css(html_path: Path, base_dir: Path) -> int:
    """Inline relative CSS ``<link>`` tags in an HTML file.

    Replaces each ``<link rel="stylesheet" href="...">`` whose ``href`` is a
    relative path with an inline ``<style>`` block containing the CSS text.
    Fonts and images referenced from within that CSS are embedded as data URIs.
    Absolute URLs (``http://``, ``https://``, ``//``) are left untouched.
    Missing files are logged as warnings and left alone.

    Parameters
    ----------
    html_path : Path
        HTML file to patch in-place.
    base_dir : Path
        Directory used to resolve relative ``href`` values.

    Returns
    -------
    int
        Number of CSS files successfully inlined.
    """
    content = html_path.read_text(encoding="utf-8")
    inlined = 0

    def _replace(match: re.Match) -> str:
        nonlocal inlined
        tag = match.group(0)
        href_match = _HREF_ATTR_RE.search(tag)
        if not href_match:
            return tag
        href = str(href_match.group(1))
        if _is_remote(href):
            return tag
        css_path = (base_dir / href).resolve()
        if not css_path.is_file():
            logger.warning("CSS file not found, leaving <link> tag intact: %s", css_path)
            return tag
        css_content = _inline_css_urls(css_path.read_text(encoding="utf-8"), css_path.parent)
        inlined += 1
        return f"<style>\n{css_content}\n</style>"

    patched = _LINK_STYLESHEET_RE.sub(_replace, content)
    if inlined:
        html_path.write_text(patched, encoding="utf-8")
        logger.debug("Inlined %d CSS file(s) into %s", inlined, html_path.name)
    return inlined


def _inline_js(html_path: Path, base_dir: Path) -> int:
    """Inline relative ``<script src="...">`` tags in an HTML file.

    Replaces each script tag whose ``src`` is a relative path with an inline
    ``<script>`` block containing the JavaScript source. Absolute URLs are left
    untouched, so CDN-hosted libraries keep loading from the CDN. Missing files
    are logged as warnings and left alone.

    Parameters
    ----------
    html_path : Path
        HTML file to patch in-place.
    base_dir : Path
        Directory used to resolve relative ``src`` values.

    Returns
    -------
    int
        Number of JavaScript files successfully inlined.
    """
    content = html_path.read_text(encoding="utf-8")
    inlined = 0

    def _replace(match: re.Match) -> str:
        nonlocal inlined
        src = str(match.group(1))
        if _is_remote(src):
            return match.group(0)
        js_path = (base_dir / src).resolve()
        if not js_path.is_file():
            logger.warning("JS file not found, leaving <script> tag intact: %s", js_path)
            return match.group(0)
        js_content = js_path.read_text(encoding="utf-8")
        # A literal "</script>" inside the source would close the tag early.
        js_content = js_content.replace("</script>", "<\\/script>")
        inlined += 1
        return f"<script>\n{js_content}\n</script>"

    patched = _SCRIPT_SRC_RE.sub(_replace, content)
    if inlined:
        html_path.write_text(patched, encoding="utf-8")
        logger.debug("Inlined %d JS file(s) into %s", inlined, html_path.name)
    return inlined


def inline_local_assets(html_path: Path, base_dir: Path) -> tuple[int, int]:
    """Inline every locally-hosted CSS and JS asset an HTML report depends on.

    This is the CDN-mode counterpart to Quarto's ``embed-resources``: it makes
    the HTML self-contained without invoking esbuild, which is unavailable in
    hardened environments. Remote (CDN) references are deliberately preserved.

    Parameters
    ----------
    html_path : Path
        HTML file to patch in-place.
    base_dir : Path
        Directory used to resolve relative references.

    Returns
    -------
    tuple[int, int]
        Number of CSS and JS files inlined, respectively.
    """
    return _inline_css(html_path, base_dir), _inline_js(html_path, base_dir)


def drop_inlined_resources(html_path: Path) -> bool:
    """Delete a Quarto ``<stem>_files`` folder once nothing references it.

    Called after :func:`inline_local_assets`. If the HTML still mentions the
    resources folder — an asset type we do not inline, or a file that failed to
    resolve — the folder is kept so the report is not broken.

    Parameters
    ----------
    html_path : Path
        The rendered HTML file whose companion resources folder to consider.

    Returns
    -------
    bool
        True if the resources folder was removed.
    """
    html_path = Path(html_path)
    resources_dir = html_path.with_name(f"{html_path.stem}_files")
    if not (html_path.is_file() and resources_dir.is_dir()):
        return False

    if resources_dir.name in html_path.read_text(encoding="utf-8"):
        logger.info(
            "%s still references %s; keeping the resources folder.",
            html_path.name,
            resources_dir.name,
        )
        return False

    shutil.rmtree(resources_dir, ignore_errors=True)
    logger.debug("Removed fully inlined resources folder %s", resources_dir.name)
    return True


def generate_zipped_report(output_filename: str, folder_to_zip: Path):
    """Generate a zipped archive of a directory.

    This is a general-purpose utility function that can compress any directory
    into a zip archive. While named for report generation, it works with any
    directory structure.

    Parameters
    ----------
    output_filename : str
        Name of the output file (extension will be replaced with .zip)
    folder_to_zip : Path
        Path to the directory to be compressed

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the folder to zip does not exist or is not a directory

    Examples
    --------
    >>> generate_zipped_report("my_archive.zip", Path("/path/to/directory"))
    >>> generate_zipped_report("report_2023", Path("/tmp/report_output"))

    """
    if not folder_to_zip.exists():
        logger.warning(
            f"The {folder_to_zip} directory does not exist. Skipping zip creation.",
        )
        return

    if not folder_to_zip.is_dir():
        logger.error(f"The output path {folder_to_zip} is not a directory.")
        return

    base_filename = Path(output_filename).with_suffix("")
    zippy = shutil.make_archive(str(base_filename), "zip", str(folder_to_zip))
    logger.info(f"created zip file...{zippy}")


def bundle_quarto_resources(output_path: Path) -> Path:
    """Bundle a Quarto-rendered file with its resources folder into a zip.

    When Quarto renders an HTML report without ``embed-resources``, it emits
    the HTML alongside a ``<basename>_files/`` directory containing the
    JavaScript and CSS assets the report needs. This helper detects that
    pattern and wraps both into a single ``<basename>.zip`` archive so the
    report can be distributed and unpacked as one unit.

    If no companion resources folder exists next to ``output_path`` (e.g. the
    report was fully embedded, or the format doesn't produce resources), the
    function is a no-op and returns ``output_path`` unchanged.

    Parameters
    ----------
    output_path : Path
        Path to the rendered report file (typically an HTML file). The
        companion resources folder is expected at ``<output_path stem>_files``
        in the same directory.

    Returns
    -------
    Path
        Path to the zip archive when bundling occurred, otherwise the
        original ``output_path``.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return output_path

    resources_dir = output_path.with_name(f"{output_path.stem}_files")
    if not resources_dir.is_dir():
        return output_path

    zip_path = output_path.with_suffix(".zip")
    logger.info(
        f"Bundling {output_path.name} with resources folder {resources_dir.name} into {zip_path.name}",
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_path, output_path.name)
        for file in resources_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(output_path.parent))

    shutil.rmtree(resources_dir, ignore_errors=True)
    try:
        output_path.unlink()
    except OSError:  # pragma: no cover
        pass
    return zip_path


def check_report_for_errors(html_path: str | Path) -> list[str]:
    """Check generated report HTML for error indicators.

    Scans the HTML file for error patterns that indicate plot rendering failures
    or exceptions during report generation. These errors are typically hidden in
    collapsed callout sections but should be caught in testing.

    Parameters
    ----------
    html_path : str or Path
        Path to the HTML file to check

    Returns
    -------
    list[str]
        List of error descriptions found (empty if no errors)

    Raises
    ------
    FileNotFoundError
        If the HTML file does not exist

    Examples
    --------
    >>> from pdstools.utils.report_utils import check_report_for_errors
    >>> errors = check_report_for_errors("HealthCheck.html")
    >>> if errors:
    ...     print(f"Found {len(errors)} error(s):")
    ...     for error in errors:
    ...         print(f"  - {error}")
    """
    html_path = Path(html_path)

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    try:
        if html_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(html_path) as zf:
                html_members = [n for n in zf.namelist() if n.endswith(".html")]
                if not html_members:
                    raise OSError(f"No HTML file found inside zip: {html_path}")
                content = zf.read(html_members[0]).decode("utf-8")
        else:
            content = html_path.read_text(encoding="utf-8")
    except Exception as e:
        raise OSError(f"Failed to read HTML file: {e}") from e

    errors = []

    # Common error patterns in HTML output from quarto_plot_exception
    error_patterns = [
        ("Error rendering", "Plot rendering error"),
        ("Traceback (most recent call last)", "Python traceback"),
        ("ValueError:", "ValueError exception"),
        ("TypeError:", "TypeError exception"),
        ("KeyError:", "KeyError exception"),
        ("AttributeError:", "AttributeError exception"),
        ("NameError:", "NameError exception"),
        ("Exception:", "Generic exception"),
        ("The given query resulted in an empty dataframe", "Empty dataframe error"),
    ]

    for pattern, description in error_patterns:
        if pattern in content:
            count = content.count(pattern)
            if count > 1:
                errors.append(f"{description} (found {count} times)")
            else:
                errors.append(description)

    return errors
