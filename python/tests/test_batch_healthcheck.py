"""Integration test for the batch report generator (scripts/batch_healthcheck.py).

Runs the actual script as a subprocess with sample data, verifying it
produces valid HTML healthcheck reports, model reports, and Excel exports.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "batch_healthcheck.py"


@pytest.fixture
def hc_layout(tmp_path):
    """Create a realistic HC/ directory from the repo's sample CSVs."""
    model_csv = DATA_DIR / "pr_data_dm_admmart_mdl_fact.csv"
    pred_csv = DATA_DIR / "pr_data_dm_admmart_pred.csv"
    if not model_csv.exists():
        pytest.skip("Sample CSV data not available")

    hc_dir = tmp_path / "SampleCustomer" / "HC"
    hc_dir.mkdir(parents=True)

    pl.read_csv(model_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")
    if pred_csv.exists():
        pl.read_csv(pred_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_PRED.parquet")

    return tmp_path


def _run_batch(hc_layout, output_dir, extra_args=None):
    """Run batch_healthcheck.py and return the subprocess result."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(hc_layout),
        "--output",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_batch_healthcheck_cdn(hc_layout, tmp_path):
    """Verify the CDN healthcheck report is produced."""
    output_dir = tmp_path / "reports"
    result = _run_batch(hc_layout, output_dir)

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Verify CDN HTML report was created
    cdn_files = list(output_dir.glob("*_cdn.html"))
    assert len(cdn_files) >= 1, f"No CDN report found, files: {[f.name for f in output_dir.glob('*.html')]}"

    for html_file in cdn_files:
        size_kb = html_file.stat().st_size / 1024
        assert size_kb > 100, f"{html_file.name} is suspiciously small: {size_kb:.1f} KB"

    # Verify summary CSV
    summary = output_dir / "summary.csv"
    assert summary.exists(), "summary.csv was not created"
    df = pl.read_csv(summary)
    assert len(df) == 1
    assert df["HC_CDN_Status"][0] == "Success"
    assert df["HC_CDN_MB"][0] > 0

    # Verify Excel export was created
    xlsx_files = list(output_dir.glob("*.xlsx"))
    assert len(xlsx_files) >= 1, f"No Excel export found in {output_dir}"
    assert df["Excel_Status"][0] == "Success"
    assert df["Excel_MB"][0] > 0


@pytest.mark.slow
def test_batch_healthcheck_full_embed(hc_layout, tmp_path):
    """Verify both CDN and full-embed reports are produced and compare sizes."""
    output_dir = tmp_path / "reports"
    result = _run_batch(hc_layout, output_dir)

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    cdn_files = list(output_dir.glob("*_cdn.html"))
    full_files = list(output_dir.glob("*_full.html"))
    assert len(cdn_files) >= 1, f"No CDN report, files: {[f.name for f in output_dir.glob('*.html')]}"
    assert len(full_files) >= 1, f"No full-embed report, files: {[f.name for f in output_dir.glob('*.html')]}"

    # HealthCheck: full-embed should be larger than CDN
    hc_cdn = [f for f in cdn_files if "models" not in f.name]
    hc_full = [f for f in full_files if "models" not in f.name]
    if hc_cdn and hc_full:
        cdn_size = hc_cdn[0].stat().st_size
        full_size = hc_full[0].stat().st_size
        print(f"HC CDN:        {cdn_size / (1024 * 1024):.1f} MB")
        print(f"HC full-embed: {full_size / (1024 * 1024):.1f} MB")
        print(f"Ratio:         {full_size / cdn_size:.1f}x")
        assert full_size > cdn_size, "Full-embed HC should be larger than CDN"

    # Verify summary has both HC modes
    df = pl.read_csv(output_dir / "summary.csv")
    assert df["HC_Embed_Status"][0] == "Success"
    assert df["HC_Embed_MB"][0] > 0

    # Verify model reports if the sample data had qualifying models
    n_models = df["ModelReport_Models"][0]
    print(f"Model reports generated: {n_models}")
    if n_models > 0:
        # Multi-model reports are zipped; single model reports are HTML
        all_outputs = list(output_dir.glob("*models*"))
        print(f"Model report files: {[f.name for f in all_outputs]}")
        assert len(all_outputs) >= 2, (
            f"Expected CDN + full-embed model report outputs, found: {[f.name for f in all_outputs]}"
        )
        assert df["ModelReport_CDN_MB"][0] > 0
        assert df["ModelReport_Embed_MB"][0] > 0
