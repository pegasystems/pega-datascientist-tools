"""Measure cold-cache import cost of the launcher entry script.

Spawns a fresh Python subprocess (so ``sys.modules`` is empty) and
times how long the launcher's top-level imports take. Useful when
auditing lazy-loading boundaries — the launcher is re-executed on
every Streamlit rerun, so import cost is paid repeatedly per session.
"""

from __future__ import annotations

import subprocess
import sys
import time

ITERATIONS = 5


def _measure_once() -> float:
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", "import pdstools.app.launcher.Home"],
        check=True,
    )
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    samples = [_measure_once() for _ in range(ITERATIONS)]
    samples.sort()
    median = samples[len(samples) // 2]
    print(f"samples (ms): {[f'{s:.0f}' for s in samples]}")
    print(f"median: {median:.0f}ms  min: {min(samples):.0f}ms  max: {max(samples):.0f}ms")


if __name__ == "__main__":
    main()
