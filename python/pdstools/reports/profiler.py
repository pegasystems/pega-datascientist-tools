"""
Cell-by-cell profiling utility for Quarto reports.

Usage in your .qmd file:
```python
from pdstools.reports.profiler import CellProfiler, show_profiling_summary

profiler = CellProfiler()

# Then wrap your cells:
with profiler.time_cell("Cell Name"):
    # your code here
    pass

# At the end of the document:
show_profiling_summary(profiler)
```
"""

import time
from contextlib import contextmanager
from typing import List, Dict
import polars as pl
from great_tables import GT, style, loc
from IPython.display import display


class CellProfiler:
    """Profile execution time of code cells in Quarto documents."""

    def __init__(self):
        self.timings: List[Dict[str, float]] = []
        self._start_time = time.perf_counter()

    @contextmanager
    def time_cell(self, cell_name: str, verbose: bool = True):
        """
        Context manager to time cell execution.

        Args:
            cell_name: Name/description of the cell being timed
            verbose: If True, print timing information as cells execute
        """
        start = time.perf_counter()
        if verbose:
            print(f"⏱️  Starting: {cell_name}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings.append(
                {
                    "Cell": cell_name,
                    "Duration (s)": round(elapsed, 2),
                    "Duration (min)": round(elapsed / 60, 2),
                }
            )
            if verbose:
                print(
                    f"✓ Completed: {cell_name} in {elapsed:.2f}s ({elapsed/60:.2f} min)"
                )

    def get_summary(self) -> pl.DataFrame:
        """Get profiling summary as a Polars DataFrame."""
        if not self.timings:
            return pl.DataFrame(
                {"Cell": [], "Duration (s)": [], "Duration (min)": [], "% of Total": []}
            )

        df = pl.DataFrame(self.timings)
        total_time = df["Duration (s)"].sum()

        df = df.with_columns(
            (pl.col("Duration (s)") / total_time * 100).round(1).alias("% of Total")
        ).sort("Duration (s)", descending=True)

        # Add a total row
        total_row = pl.DataFrame(
            {
                "Cell": ["TOTAL"],
                "Duration (s)": [total_time],
                "Duration (min)": [round(total_time / 60, 2)],
                "% of Total": [100.0],
            }
        )

        return pl.concat([df, total_row])

    def print_summary(self):
        """Print a simple text summary."""
        df = self.get_summary()
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)
        print(df)
        print("=" * 80 + "\n")


def show_profiling_summary(
    profiler: CellProfiler, title: str = "Cell Execution Time Profile"
):
    """
    Display a formatted profiling summary table using Great Tables.

    Args:
        profiler: CellProfiler instance with timing data
        title: Title for the summary table
    """
    df = profiler.get_summary()

    if df.height == 0:
        print("No profiling data collected.")
        return

    # Create Great Table
    gt = (
        GT(df)
        .tab_header(title=title)
        .fmt_number(decimals=2, columns=["Duration (s)", "Duration (min)"])
        .fmt_number(decimals=1, columns=["% of Total"])
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(rows=pl.col("Cell") == "TOTAL"),
        )
        .tab_style(
            style=style.fill(color="#FFF3CD"),
            locations=loc.body(
                rows=(pl.col("% of Total") > 10) & (pl.col("Cell") != "TOTAL")
            ),
        )
        .tab_style(
            style=style.fill(color="#F8D7DA"),
            locations=loc.body(
                rows=(pl.col("% of Total") > 20) & (pl.col("Cell") != "TOTAL")
            ),
        )
    )

    display(gt)

    # Also print key insights
    df_without_total = df.filter(pl.col("Cell") != "TOTAL")
    if df_without_total.height > 0:
        slowest = df_without_total.row(0, named=True)
        print("\n📊 Insights:")
        print(
            f"   • Slowest cell: '{slowest['Cell']}' ({slowest['Duration (s)']}s, {slowest['% of Total']:.1f}% of total)"
        )

        slow_cells = df_without_total.filter(pl.col("% of Total") > 10)
        if slow_cells.height > 0:
            print(f"   • {slow_cells.height} cell(s) take >10% of total time")

        very_slow_cells = df_without_total.filter(pl.col("% of Total") > 20)
        if very_slow_cells.height > 0:
            print(
                f"   • {very_slow_cells.height} cell(s) take >20% of total time (potential bottlenecks)"
            )
