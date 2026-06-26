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
PRERELEASE_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<phase>a|b|rc)(?P<serial>\d+)$")
PRERELEASE_PHASE_ORDER = {"a": 0, "b": 1, "rc": 2}


def normalize_version_slug(ref_type: str, ref_name: str) -> str:
    """Return the published docs slug for a Git ref."""

    if ref_type != "tag":
        return "dev"
    return ref_name.removeprefix("V").removeprefix("v")


def _stable_release_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def _parse_prerelease(version: str) -> tuple[int, int, int, str, int] | None:
    match = PRERELEASE_RE.match(version)
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        match.group("phase"),
        int(match.group("serial")),
    )


def _prerelease_key(version: str) -> tuple[int, int, int, int, int]:
    major, minor, patch, phase, serial = _parse_prerelease(version) or (0, 0, 0, "a", 0)
    return (major, minor, patch, PRERELEASE_PHASE_ORDER[phase], serial)


def _collect_ordered_versions(site_root: Path) -> tuple[list[str], list[str], list[str]]:
    stable_releases: list[str] = []
    stable_bases: set[tuple[int, int, int]] = set()
    prereleases_by_base: dict[tuple[int, int, int], str] = {}

    for directory in site_root.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue

        version = directory.name
        if version == "dev":
            continue

        if SEMVER_RE.match(version):
            base = _stable_release_key(version)
            stable_releases.append(version)
            stable_bases.add(base)
            continue

        prerelease = _parse_prerelease(version)
        if prerelease is None:
            continue

        base = prerelease[:3]
        current = prereleases_by_base.get(base)
        if current is None or _prerelease_key(version) > _prerelease_key(current):
            prereleases_by_base[base] = version

    stable_releases.sort(key=_stable_release_key, reverse=True)
    prereleases = sorted(
        [version for base, version in prereleases_by_base.items() if base not in stable_bases],
        key=_prerelease_key,
        reverse=True,
    )

    ordered_versions = []
    if (site_root / "dev").is_dir():
        ordered_versions.append("dev")
    ordered_versions.extend(stable_releases)
    ordered_versions.extend(prereleases)
    return ordered_versions, stable_releases, prereleases


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

    ordered_versions, stable_releases, prereleases = _collect_ordered_versions(site_root)

    if pypi_version in stable_releases:
        preferred_version = pypi_version
    elif stable_releases:
        preferred_version = stable_releases[0]
    elif prereleases:
        preferred_version = prereleases[0]
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
