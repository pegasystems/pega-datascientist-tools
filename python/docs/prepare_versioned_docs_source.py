"""Prepare historical docs sources with the current publishing scaffolding."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

SCAFFOLD_PATHS = [
    Path("python/docs/source/conf.py"),
    Path("python/docs/source/_static/version-switcher.css"),
    Path("python/docs/source/_static/version-switcher.js"),
    Path("python/docs/source/_templates/sidebar/version-switcher.html"),
]


def export_git_ref(repo_root: Path, source_ref: str, output_dir: Path) -> None:
    """Export a git ref into an output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", source_ref],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["tar", "-xf", "-", "-C", str(output_dir)],
        cwd=repo_root,
        input=archive.stdout,
        check=True,
    )


def overlay_current_docs_scaffolding(repo_root: Path, build_root: Path) -> None:
    """Copy the current docs switcher/config scaffolding into a build tree."""

    for relative_path in SCAFFOLD_PATHS:
        source = repo_root / relative_path
        target = build_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    export_git_ref(repo_root, args.source_ref, output_dir)
    overlay_current_docs_scaffolding(repo_root, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
