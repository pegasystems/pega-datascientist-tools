"""Helpers for publishing versioned documentation to GitHub Pages."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def normalize_version_slug(ref_type: str, ref_name: str) -> str:
    """Return the published docs slug for a Git ref."""

    if ref_type != "tag":
        return "dev"
    return ref_name.removeprefix("V").removeprefix("v")


def _stable_release_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def fetch_pypi_version(package_name: str, timeout: int = 10) -> str | None:
    """Return the latest version reported by PyPI for a package."""

    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError):
        return None
    return payload.get("info", {}).get("version")


def build_manifest(
    site_root: Path,
    base_url: str,
    *,
    pypi_version: str | None = None,
) -> tuple[list[dict[str, object]], str | None]:
    """Build the versions manifest and preferred version for the site."""

    version_dirs = sorted(
        directory.name
        for directory in site_root.iterdir()
        if directory.is_dir() and not directory.name.startswith(".") and directory.name != "dev"
    )
    stable_releases = sorted(
        [version for version in version_dirs if SEMVER_RE.match(version)],
        key=_stable_release_key,
        reverse=True,
    )
    other_versions = sorted(version for version in version_dirs if not SEMVER_RE.match(version))

    ordered_versions = []
    if (site_root / "dev").is_dir():
        ordered_versions.append("dev")
    ordered_versions.extend(stable_releases)
    ordered_versions.extend(other_versions)

    if pypi_version in ordered_versions:
        preferred_version = pypi_version
    elif stable_releases:
        preferred_version = stable_releases[0]
    elif ordered_versions:
        preferred_version = ordered_versions[0]
    else:
        preferred_version = None

    manifest = [
        {
            "version": version,
            "url": f"{base_url.rstrip('/')}/{version}/",
            "preferred": version == preferred_version,
        }
        for version in ordered_versions
    ]
    return manifest, preferred_version


def write_manifest(site_root: Path, manifest: list[dict[str, object]]) -> None:
    """Write versions.json into the site root."""

    output = site_root / "versions.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")


def write_root_redirect(site_root: Path, preferred_version: str | None) -> None:
    """Write the root index.html redirect for the preferred docs version."""

    target = f"{preferred_version}/" if preferred_version else "dev/"
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={target}">
    <title>Redirecting…</title>
  </head>
  <body>
    <p>Redirecting to <a href="{target}">the documentation</a>.</p>
  </body>
</html>
"""
    (site_root / "index.html").write_text(html)


def iter_html_relative_paths(version_root: Path) -> list[Path]:
    """Return all published HTML paths for a version, relative to its root."""

    return sorted(path.relative_to(version_root) for path in version_root.rglob("*.html"))


def write_html_redirect(destination: Path, target_url: str) -> None:
    """Write a small HTML redirect page."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={target_url}">
    <title>Redirecting…</title>
  </head>
  <body>
    <p>Redirecting to <a href="{target_url}">the documentation</a>.</p>
  </body>
</html>
"""
    destination.write_text(html)


def write_latest_aliases(
    site_root: Path,
    base_url: str,
    preferred_version: str | None,
) -> None:
    """Write stable alias redirects for the preferred docs version."""

    latest_root = site_root / "latest"
    if latest_root.exists():
        shutil.rmtree(latest_root)

    if preferred_version is None:
        return

    version_root = site_root / preferred_version
    if not version_root.is_dir():
        return

    base_url = base_url.rstrip("/")
    for relative_path in iter_html_relative_paths(version_root):
        target_url = f"{base_url}/{preferred_version}/{relative_path.as_posix()}"
        write_html_redirect(latest_root / relative_path, target_url)
        if relative_path != Path("index.html"):
            write_html_redirect(site_root / relative_path, target_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-root", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--package-name", default="pdstools")
    parser.add_argument("--pypi-version")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    site_root = Path(args.site_root)
    pypi_version = args.pypi_version or fetch_pypi_version(args.package_name)
    manifest, preferred_version = build_manifest(
        site_root,
        args.base_url,
        pypi_version=pypi_version,
    )
    write_manifest(site_root, manifest)
    write_root_redirect(site_root, preferred_version)
    write_latest_aliases(site_root, args.base_url, preferred_version)
    print(json.dumps({"preferred_version": preferred_version, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
