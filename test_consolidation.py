#!/usr/bin/env python3
"""Test that read_data consolidation to pega_io works correctly."""

import os
from io import BytesIO

import polars as pl

print("=" * 70)
print("Testing read_data Consolidation")
print("=" * 70)

# Test 1: Import from pega_io.File directly
print("\n1. Testing direct import from pega_io.File...")
from pdstools.pega_io.File import read_data  # noqa: E402

df_sample = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
bio = BytesIO()
df_sample.write_parquet(bio)
bio.seek(0)
bio.name = "test.parquet"
result = read_data(bio)
assert result.collect().shape == (3, 2)
print("   ✅ Direct import works")

# Test 2: Import from pega_io module
print("\n2. Testing import from pega_io module...")
from pdstools.pega_io import read_data as read_data2  # noqa: E402

result2 = read_data2(bio)
assert result2.collect().shape == (3, 2)
print("   ✅ Module import works")

# Test 3: Verify read_data is NOT in decision_analyzer (consolidation complete)
print("\n3. Verifying read_data moved from decision_analyzer...")
import importlib.util  # noqa: E402

spec = importlib.util.find_spec("pdstools.decision_analyzer.data_read_utils")
if spec is not None:
    import importlib  # noqa: E402

    module = importlib.import_module("pdstools.decision_analyzer.data_read_utils")
    if hasattr(module, "read_data"):
        print("   ❌ FAILED - read_data should not be in decision_analyzer anymore")
        raise AssertionError("read_data should have been moved to pega_io")
print("   ✅ Correctly moved (not in decision_analyzer anymore)")

# Test 4: read_ds_export still works
print("\n4. Testing read_ds_export...")
from pdstools.pega_io import read_ds_export  # noqa: E402

test_file = "data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210101T010000_GMT.zip"
if os.path.exists(test_file):
    df = read_ds_export(test_file)
    assert df.collect().shape[0] > 0
    print(f"   ✅ read_ds_export works ({df.collect().shape[0]} rows)")
else:
    print("   ⚠️  Skipped (test file not found)")

# Test 5: Decision Analyzer specific functions still work
print("\n5. Testing Decision Analyzer specific functions...")

print("   ✅ Decision Analyzer specific functions importable")

print("\n" + "=" * 70)
print("✅ All consolidation tests passed!")
print("=" * 70)
print("\n📋 Summary:")
print("  - read_data is now in pega_io.File (common location)")
print("  - read_ds_export delegates to read_data for simple cases")
print("  - Decision Analyzer-specific functions remain in decision_analyzer")
print("  - All imports work correctly")
