from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ALLOW_MARKER = "quality-guard: allow-weak-assert"

WEAK_ASSERT_PATTERNS = {
    "assert ... is not None": re.compile(r"assert\b.*\bis not None\b"),
    "assert isinstance(..., pl.DataFrame/LazyFrame)": re.compile(r"isinstance\([^)]*pl\.(?:DataFrame|LazyFrame)\)"),
    "assert len(...) > 0": re.compile(r"assert\b.*len\([^)]*\)\s*>\s*0\b"),
}


def _tracked_changed_files() -> list[Path]:
    base_ref = _git_output(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], allow_fail=True)
    if base_ref:
        merge_base = _git_output(["merge-base", "HEAD", base_ref])
        diff_range = f"{merge_base}..HEAD"
    else:
        head_parent = _git_output(["rev-parse", "HEAD~1"], allow_fail=True)
        diff_range = "HEAD~1..HEAD" if head_parent else "HEAD"

    files = _git_output(["diff", "--name-only", "--diff-filter=ACMR", diff_range])
    return [Path(line) for line in files.splitlines() if line.strip()]


def _git_output(args: list[str], allow_fail: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if allow_fail:
            return ""
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _iter_target_files(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(arg) for arg in argv if arg.endswith(".py")]
    return _tracked_changed_files()


def _check_weak_asserts(path: Path) -> list[str]:
    violations: list[str] = []
    if "python/tests/" not in path.as_posix():
        return violations

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if "assert" not in stripped or ALLOW_MARKER in stripped:
            continue
        for label, pattern in WEAK_ASSERT_PATTERNS.items():
            if pattern.search(stripped):
                violations.append(f"{path}:{line_no}: weak assertion ({label})")
                break
    return violations


def _check_type_ignore_reasons(path: Path) -> list[str]:
    violations: list[str] = []
    if "python/pdstools/" not in path.as_posix():
        return violations

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "# type: ignore" not in line:
            continue
        if not re.search(r"# type: ignore(?:\[[^]]+\])?\s+#\s+\S", line):
            violations.append(f"{path}:{line_no}: # type: ignore requires an explanatory trailing comment")
    return violations


def main(argv: list[str]) -> int:
    files = [path for path in _iter_target_files(argv) if path.exists()]
    violations: list[str] = []
    for path in files:
        violations.extend(_check_weak_asserts(path))
        violations.extend(_check_type_ignore_reasons(path))

    if violations:
        print("Quality guard violations found:")
        for violation in violations:
            print(f"- {violation}")
        print(
            "\nUse exact-value/behavior assertions instead of structural placeholders, "
            "or add an explanatory allow marker when the invariant is genuinely intentional."
        )
        return 1

    print(f"Quality guards passed for {len(files)} changed Python file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
