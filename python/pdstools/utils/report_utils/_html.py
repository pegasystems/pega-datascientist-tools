"""HTML post-processing: CSS inlining, zip bundling, error scanning."""

import os
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


def _inline_css(html_path: Path, base_dir: Path) -> int:
    """Inline relative CSS ``<link>`` tags in an HTML file.

    Replaces each ``<link rel="stylesheet" href="...">`` whose ``href`` is a
    relative path with an inline ``<style>`` block containing the CSS text.
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
        href = href_match.group(1)
        if href.startswith(("http://", "https://", "//")):
            return tag
        css_path = (base_dir / href).resolve()
        if not css_path.is_file():
            logger.warning("CSS file not found, leaving <link> tag intact: %s", css_path)
            return tag
        css_content = css_path.read_text(encoding="utf-8")
        inlined += 1
        return f"<style>\n{css_content}\n</style>"

    patched = _LINK_STYLESHEET_RE.sub(_replace, content)
    if inlined:
        html_path.write_text(patched, encoding="utf-8")
        logger.debug("Inlined %d CSS file(s) into %s", inlined, html_path.name)
    return inlined


def generate_zipped_report(output_filename: str, folder_to_zip: str):
    """Generate a zipped archive of a directory.

    This is a general-purpose utility function that can compress any directory
    into a zip archive. While named for report generation, it works with any
    directory structure.

    Parameters
    ----------
    output_filename : str
        Name of the output file (extension will be replaced with .zip)
    folder_to_zip : str
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
    >>> generate_zipped_report("my_archive.zip", "/path/to/directory")
    >>> generate_zipped_report("report_2023", "/tmp/report_output")

    """
    if not os.path.isdir(folder_to_zip):
        logger.error(f"The output path {folder_to_zip} is not a directory.")
        return

    if not os.path.exists(folder_to_zip):
        logger.warning(
            f"The {folder_to_zip} directory does not exist. Skipping zip creation.",
        )
        return

    base_filename = os.path.splitext(output_filename)[0]
    zippy = shutil.make_archive(base_filename, "zip", folder_to_zip)
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
                    raise IOError(f"No HTML file found inside zip: {html_path}")
                content = zf.read(html_members[0]).decode("utf-8")
        else:
            content = html_path.read_text(encoding="utf-8")
    except Exception as e:
        raise IOError(f"Failed to read HTML file: {e}")

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
