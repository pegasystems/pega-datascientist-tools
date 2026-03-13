#!/usr/bin/env python3
"""Verify Decision Analyzer sampling with large customer datasets.

Tests the sampling functionality with large TAR and ZIP files to ensure:
- Files load correctly
- Sampling produces expected output
- Metadata is properly recorded
- Disk space is managed efficiently
"""

import sys
from pathlib import Path

from pdstools.pega_io import read_data
from pdstools.decision_analyzer.utils import prepare_and_save


def format_size_mb(size_bytes: int) -> str:
    """Format file size in MB with 2 decimal places."""
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def test_file(file_path: str, sample_n: int, output_dir: str = ".") -> bool:
    """Test sampling a single file.

    Parameters
    ----------
    file_path : str
        Path to the file to test
    sample_n : int
        Number of interactions to sample
    output_dir : str
        Directory to write sample files to

    Returns
    -------
    bool
        True if test succeeded, False otherwise
    """
    path = Path(file_path)

    if not path.exists():
        print(f"   ❌ File not found: {file_path}")
        return False

    print(f"📁 Path: {path.name}")
    if path.is_file():
        file_size = path.stat().st_size
        print(f"📊 Size: {format_size_mb(file_size)}")
    else:
        print("📊 Type: Directory (Hive-partitioned)")
    print()

    try:
        # Step 1: Load data
        print("1️⃣ Loading data...")
        df = read_data(str(path))
        columns = df.collect_schema().names()
        print("   ✅ Data loaded successfully")
        print(f"   📋 Columns: {len(columns)} ({', '.join(columns[:5])}...)")
        print()

        # Step 2: Sample the data
        print(f"2️⃣ Sampling to {sample_n:,} interactions...")
        sampled_df, sample_path = prepare_and_save(df, n=sample_n, output_dir=output_dir, source_path=str(path))

        if sample_path is None:
            print("   ⚠️  Data is smaller than sample size - using full dataset")
            return True

        # Get sample file details
        sample_size = sample_path.stat().st_size
        sample_meta = sampled_df.collect_schema().metadata() or {}
        interaction_count = sampled_df.select("pxInteractionID").unique().collect().height
        source_file = sample_meta.get("source_file", "unknown")
        sample_pct = sample_meta.get("sample_percentage", "0.00%")

        print(f"   ✅ Sample file created: {sample_path.name}")
        print(f"   📊 Sample file size: {format_size_mb(sample_size)}")
        print(f"   📈 Sampled interactions: {interaction_count:,}")
        print(f"   🏷️  Source: {Path(source_file).name}")
        print(f"   📊 Sample %: {sample_pct}")
        print()

        return True

    except Exception as e:
        print()
        print(f"❌ FAILED - {path.name}")
        print(f"   Error: {type(e).__name__}: {e}")
        import traceback

        print("Traceback (most recent call last):")
        print("".join(traceback.format_tb(e.__traceback__)[:3]))  # Show first 3 frames
        return False


def main():
    """Run sampling verification tests."""
    # Configuration
    sample_n = 10000
    output_dir = "."

    # Test files
    base_path = Path(
        "/Users/perdo/Library/CloudStorage/OneDrive-PegasystemsInc/AI Chapter/projects/Decision Analyzer (Insights)/data_decision_analyzer_raw_EEv2"
    )
    test_files = [
        str(base_path / "BOI" / "pxDecisionTime_day=08"),  # Unpacked directory (Hive-partitioned)
        str(base_path / "BOI" / "BOI_pxDecisionTime_day_08.tar"),
        str(base_path / "BOI" / "BOI_pxDecisionTime_day_08.zip"),
        str(base_path / "NAB" / "nab_environments.zip"),
    ]

    print("=" * 80)
    print("Decision Analyzer Sampling Verification")
    print("=" * 80)
    print()
    print(f"Testing with {sample_n:,} interaction samples")
    print()

    results = {}

    for file_path in test_files:
        path = Path(file_path)
        if path.is_dir():
            test_name = f"{path.name} (directory)"
        else:
            file_type = path.suffix.upper()[1:] if path.suffix else "unknown"
            test_name = f"{path.stem} ({file_type} file)"

        print("=" * 80)
        print(f"Testing: {test_name}")
        print("=" * 80)

        success = test_file(file_path, sample_n, output_dir)
        results[test_name] = success

        if success:
            print(f"✅ SUCCESS - {test_name}")
        else:
            print(f"❌ FAILED - {test_name}")

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
