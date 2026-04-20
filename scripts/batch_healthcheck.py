#!/usr/bin/env python
"""Batch ADM Report Generator.

Generate ADM HealthCheck reports, Model Reports, and Excel exports for
multiple datasets.

This script discovers ADM model and predictor data files, generates reports
in both CDN and full-embed modes, and creates a summary of results with
error detection.

Usage Examples
--------------
Process all datasets in a directory:
    python batch_healthcheck.py /path/to/data

Process a single dataset:
    python batch_healthcheck.py /path/to/data/CustomerA

Specify output directory:
    python batch_healthcheck.py /path/to/data --output ./reports

Process specific datasets by name:
    python batch_healthcheck.py /path/to/data --datasets CustomerA CustomerB

Generate model reports for up to 5 interesting models:
    python batch_healthcheck.py /path/to/data --max-models 5

Directory Structure
-------------------
The script automatically discovers data in these patterns:
- /path/to/data/Dataset1/HC/*.parquet
- /path/to/data/Dataset2/HC/*.parquet
- /path/to/data/HC/*.parquet (if single dataset)
- /path/to/data/*.parquet (if files at root)

Required files:
- Model file: PR_DATA_DM_ADMMART_MDL_FACT.parquet (or *MDL_FACT.parquet)
- Predictor file: PR_DATA_DM_ADMMART_PRED.parquet (optional, or *PRED.parquet)
"""

import argparse
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl

from pdstools import ADMDatamart
from pdstools.utils.report_utils import check_report_for_errors


# Default file name patterns
MODEL_FILE_PATTERNS = ["PR_DATA_DM_ADMMART_MDL_FACT.parquet", "*MDL_FACT.parquet"]
PREDICTOR_FILE_PATTERNS = ["PR_DATA_DM_ADMMART_PRED.parquet", "*PRED.parquet"]


def find_data_directories(root_path: Path) -> list[dict]:
    """Discover directories containing ADM data files.

    Parameters
    ----------
    root_path : Path
        Root directory to search for data

    Returns
    -------
    list[dict]
        List of dictionaries with keys: name, data_dir, model_file, predictor_file
    """
    datasets = []

    # Check if root_path itself contains data files
    for pattern in MODEL_FILE_PATTERNS:
        model_files = list(root_path.glob(pattern))
        if model_files:
            # Found data at root level
            model_file = model_files[0]
            predictor_file = None
            for pred_pattern in PREDICTOR_FILE_PATTERNS:
                pred_files = list(root_path.glob(pred_pattern))
                if pred_files:
                    predictor_file = pred_files[0]
                    break

            datasets.append(
                {
                    "name": root_path.name,
                    "data_dir": root_path,
                    "model_file": model_file,
                    "predictor_file": predictor_file,
                }
            )
            return datasets  # If we found data at root, don't search subdirs

    # Check for HC subdirectory at root level
    hc_dir = root_path / "HC"
    if hc_dir.exists() and hc_dir.is_dir():
        for pattern in MODEL_FILE_PATTERNS:
            model_files = list(hc_dir.glob(pattern))
            if model_files:
                model_file = model_files[0]
                predictor_file = None
                for pred_pattern in PREDICTOR_FILE_PATTERNS:
                    pred_files = list(hc_dir.glob(pred_pattern))
                    if pred_files:
                        predictor_file = pred_files[0]
                        break

                datasets.append(
                    {
                        "name": root_path.name,
                        "data_dir": hc_dir,
                        "model_file": model_file,
                        "predictor_file": predictor_file,
                    }
                )
                return datasets

    # Search subdirectories for HC folders or direct data
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Check subdir/HC pattern
        hc_dir = subdir / "HC"
        if hc_dir.exists() and hc_dir.is_dir():
            for pattern in MODEL_FILE_PATTERNS:
                model_files = list(hc_dir.glob(pattern))
                if model_files:
                    model_file = model_files[0]
                    predictor_file = None
                    for pred_pattern in PREDICTOR_FILE_PATTERNS:
                        pred_files = list(hc_dir.glob(pred_pattern))
                        if pred_files:
                            predictor_file = pred_files[0]
                            break

                    datasets.append(
                        {
                            "name": subdir.name,
                            "data_dir": hc_dir,
                            "model_file": model_file,
                            "predictor_file": predictor_file,
                        }
                    )
                    break

        # Check subdir directly for data files
        if not any(d["name"] == subdir.name for d in datasets):
            for pattern in MODEL_FILE_PATTERNS:
                model_files = list(subdir.glob(pattern))
                if model_files:
                    model_file = model_files[0]
                    predictor_file = None
                    for pred_pattern in PREDICTOR_FILE_PATTERNS:
                        pred_files = list(subdir.glob(pred_pattern))
                        if pred_files:
                            predictor_file = pred_files[0]
                            break

                    datasets.append(
                        {
                            "name": subdir.name,
                            "data_dir": subdir,
                            "model_file": model_file,
                            "predictor_file": predictor_file,
                        }
                    )
                    break

    return datasets


def get_file_size_mb(file_path: Path | None) -> float:
    """Get file size in MB."""
    if file_path and file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def select_interesting_models(datamart: ADMDatamart, max_n: int = 3) -> list[str]:
    """Select a diverse set of interesting models for model reports.

    Picks top-performing Naive Bayes models (excluding AGB/Classifier) with
    sufficient volume, selecting the best performer per Channel/Direction/Issue
    combination for diversity.

    Parameters
    ----------
    datamart : ADMDatamart
        The loaded datamart
    max_n : int
        Maximum number of models to select

    Returns
    -------
    list[str]
        List of ModelID strings
    """
    group_keys = [c for c in ["Channel", "Direction", "Issue"] if c in datamart.combined_data.collect_schema().names()]

    if datamart.predictor_data is None:
        print("  ℹ No predictor data — skipping model selection")
        return []

    # Exclude AGB models: only keep models that have real predictor bins
    nb_model_ids = set(
        datamart.predictor_data.filter((pl.col("BinType") != "NONE") & (pl.col("EntryType") != "Classifier"))
        .select(pl.col("ModelID").unique())
        .collect()["ModelID"]
        .to_list()
    )

    if not nb_model_ids:
        print("  ℹ No Naive Bayes models found")
        return []

    # Also require Classifier data (needed for score distribution in reports)
    classifier_model_ids = set(
        datamart.predictor_data.filter(pl.col("EntryType") == "Classifier")
        .select(pl.col("ModelID").unique())
        .collect()["ModelID"]
        .to_list()
    )
    nb_model_ids = list(nb_model_ids & classifier_model_ids)

    if not nb_model_ids:
        print("  ℹ No models with both predictor bins and Classifier data found")
        return []

    mdls = (
        datamart.combined_data.filter(pl.col("ModelID").is_in(nb_model_ids))
        .filter((pl.col("Positives") >= 200) & (pl.col("ResponseCount") >= 1000))
        .group_by(group_keys)
        .agg(
            pl.col("ModelID").top_k_by("Performance", k=1).first(),
            pl.col("Performance").max(),
        )
        .sort(group_keys)
        .collect()
    )

    selected = mdls["ModelID"].head(max_n).to_list()
    print(f"  ✓ Selected {len(selected)} interesting model(s) for reports")
    return selected


def _check_output_for_errors(output_file: Path) -> list[str]:
    """Check report output for HTML rendering errors.

    Handles both plain HTML files and zip archives (multi-model reports).
    For zips, extracts HTML files to a temp directory and checks each one.
    """
    if output_file.suffix == ".zip":
        all_errors = []
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(output_file) as zf:
                zf.extractall(tmp)
            for html_file in Path(tmp).glob("*.html"):
                errors = check_report_for_errors(html_file)
                if errors:
                    all_errors.extend(f"{html_file.name}: {e}" for e in errors)
        return all_errors

    return check_report_for_errors(output_file)


def _generate_quarto_report(
    generate_fn,
    label: str,
    output_dir: Path,
    *,
    full_embed: bool,
    **kwargs,
) -> tuple[float, str, str | None]:
    """Generate a Quarto report and return (size_mb, status, errors).

    Parameters
    ----------
    generate_fn : callable
        Bound method like datamart.generate.health_check or .model_reports
    label : str
        Human-readable label for logging (e.g. "HealthCheck CDN")
    output_dir : Path
        Output directory
    full_embed : bool
        Whether to embed all resources
    **kwargs
        Additional keyword arguments passed to generate_fn
    """
    mode = "full-embed" if full_embed else "CDN"
    print(f"  → Generating {label} ({mode})...")

    try:
        output_path = generate_fn(
            output_dir=str(output_dir),
            full_embed=full_embed,
            **kwargs,
        )

        output_file = Path(output_path)
        size_mb = get_file_size_mb(output_file)
        print(f"  ✓ {label} ({mode}): {size_mb:.1f} MB")

        # Check HTML files for rendering errors
        html_errors = _check_output_for_errors(output_file)
        if html_errors:
            errors_str = "; ".join(html_errors)
            print(f"  ⚠ HTML errors in {label} ({mode}):")
            for error in html_errors:
                print(f"    - {error}")
            return size_mb, "Success (with errors)", errors_str

        print(f"  ✓ No errors in {label} ({mode})")
        return size_mb, "Success", None

    except Exception as e:
        print(f"  ✗ Error in {label} ({mode}): {e}")
        traceback.print_exc()
        return 0.0, "Error", str(e)


def process_dataset(
    dataset: dict,
    output_dir: Path,
    *,
    max_models: int = 3,
) -> dict:
    """Process a single dataset and generate all reports.

    Generates HealthCheck reports (CDN + full-embed), Model Reports for
    selected interesting models, and an Excel export.

    Parameters
    ----------
    dataset : dict
        Dataset information (name, data_dir, model_file, predictor_file)
    output_dir : Path
        Directory for output reports
    max_models : int
        Maximum number of model reports to generate

    Returns
    -------
    dict
        Processing results with status and metrics
    """
    name = dataset["name"]
    print(f"\n{'=' * 60}")
    print(f"Processing: {name}")
    print(f"{'=' * 60}")
    print(f"  Data directory: {dataset['data_dir']}")

    result = {
        "Dataset": name,
        "Model_File_MB": 0.0,
        "Predictor_File_MB": 0.0,
        "HC_CDN_MB": 0.0,
        "HC_CDN_Status": "Not Found",
        "HC_CDN_Errors": None,
        "HC_Embed_MB": 0.0,
        "HC_Embed_Status": "Not Found",
        "HC_Embed_Errors": None,
        "ModelReport_Models": 0,
        "ModelReport_CDN_MB": 0.0,
        "ModelReport_CDN_Status": "Skipped",
        "ModelReport_Embed_MB": 0.0,
        "ModelReport_Embed_Status": "Skipped",
        "Excel_MB": 0.0,
        "Excel_Status": "Skipped",
    }

    model_file = dataset["model_file"]
    predictor_file = dataset["predictor_file"]

    result["Model_File_MB"] = get_file_size_mb(model_file)
    result["Predictor_File_MB"] = get_file_size_mb(predictor_file)

    print(f"  ✓ Model file: {result['Model_File_MB']:.1f} MB")
    if predictor_file:
        print(f"  ✓ Predictor file: {result['Predictor_File_MB']:.1f} MB")
    else:
        print("  ℹ No predictor file found")

    try:
        print("  → Loading datamart...")
        datamart = ADMDatamart.from_ds_export(
            model_filename=str(model_file),
            predictor_filename=str(predictor_file) if predictor_file else None,
        )
        n_models = len(datamart.model_data.collect())
        print(f"  ✓ Datamart loaded: {n_models} models")

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ── HealthCheck reports (CDN + full-embed) ──────────────────
        for full_embed, key_prefix in [(False, "HC_CDN"), (True, "HC_Embed")]:
            suffix = "_full" if full_embed else "_cdn"
            mb, status, errors = _generate_quarto_report(
                datamart.generate.health_check,
                "HealthCheck",
                output_dir,
                full_embed=full_embed,
                name=safe_name + suffix,
                title=f"ADM Health Check - {name}",
                subtitle=f"Generated on {timestamp}",
            )
            result[f"{key_prefix}_MB"] = mb
            result[f"{key_prefix}_Status"] = status
            result[f"{key_prefix}_Errors"] = errors

        if result["HC_CDN_MB"] > 0 and result["HC_Embed_MB"] > 0:
            ratio = result["HC_Embed_MB"] / result["HC_CDN_MB"]
            print(
                f"  ℹ HC size: CDN {result['HC_CDN_MB']:.1f} MB vs embed {result['HC_Embed_MB']:.1f} MB ({ratio:.1f}x)"
            )

        # ── Model reports for interesting models ────────────────────
        selected_models = select_interesting_models(datamart, max_n=max_models)
        result["ModelReport_Models"] = len(selected_models)

        if selected_models:
            for full_embed, key_prefix in [(False, "ModelReport_CDN"), (True, "ModelReport_Embed")]:
                suffix = "_full" if full_embed else "_cdn"
                mb, status, errors = _generate_quarto_report(
                    datamart.generate.model_reports,
                    f"ModelReport ({len(selected_models)} models)",
                    output_dir,
                    full_embed=full_embed,
                    model_ids=selected_models,
                    name=f"{safe_name}_models{suffix}",
                    title=f"Model Reports - {name}",
                    subtitle=f"Generated on {timestamp}",
                )
                result[f"{key_prefix}_MB"] = mb
                result[f"{key_prefix}_Status"] = status

            if result["ModelReport_CDN_MB"] > 0 and result["ModelReport_Embed_MB"] > 0:
                ratio = result["ModelReport_Embed_MB"] / result["ModelReport_CDN_MB"]
                print(
                    f"  ℹ Model report size: CDN {result['ModelReport_CDN_MB']:.1f} MB"
                    f" vs embed {result['ModelReport_Embed_MB']:.1f} MB ({ratio:.1f}x)"
                )

        # ── Excel export ────────────────────────────────────────────
        print("  → Generating Excel export...")
        try:
            excel_path = output_dir / f"{safe_name}.xlsx"
            path, warnings = datamart.generate.excel_report(
                name=excel_path,
                predictor_binning=True,
            )
            if path:
                result["Excel_MB"] = get_file_size_mb(Path(path))
                result["Excel_Status"] = "Success"
                print(f"  ✓ Excel export: {result['Excel_MB']:.1f} MB")
                if warnings:
                    for w in warnings:
                        print(f"    ⚠ {w}")
            else:
                result["Excel_Status"] = "No data"
                print("  ℹ Excel export: no data available")
        except Exception as e:
            result["Excel_Status"] = "Error"
            print(f"  ✗ Excel export error: {e}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        for key in result:
            if key.endswith("_Status") and result[key] in ("Not Found", "Skipped"):
                result[key] = "Error"
        traceback.print_exc()

    return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Batch generate ADM reports (HealthCheck, Model Reports, Excel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/customers
  %(prog)s /path/to/customers --output ./reports
  %(prog)s /path/to/customers --datasets CustomerA CustomerB
  %(prog)s /path/to/single_customer/HC
  %(prog)s /path/to/data --max-models 5

For more information, see:
  https://github.com/pegasystems/pega-datascientist-tools
        """,
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to directory containing datasets (with HC folders) or a single dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./healthcheck_reports"),
        help="Output directory for generated reports (default: ./healthcheck_reports)",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        help="Specific dataset names to process (default: process all found)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=3,
        help="Maximum number of model reports to generate per dataset (default: 3)",
    )

    args = parser.parse_args()

    # Validate input path
    if not args.data_path.exists():
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)

    if not args.data_path.is_dir():
        print(f"Error: Data path is not a directory: {args.data_path}")
        sys.exit(1)

    # Discover datasets
    print(f"\n{'=' * 60}")
    print("Discovering datasets...")
    print(f"{'=' * 60}")
    print(f"Searching in: {args.data_path.absolute()}")

    all_datasets = find_data_directories(args.data_path)

    if not all_datasets:
        print("\nNo datasets found!")
        print("\nExpected file patterns:")
        print(f"  Model: {', '.join(MODEL_FILE_PATTERNS)}")
        print(f"  Predictor: {', '.join(PREDICTOR_FILE_PATTERNS)}")
        print("\nExpected directory structures:")
        print("  - /path/to/data/Dataset1/HC/*.parquet")
        print("  - /path/to/data/Dataset1/*.parquet")
        print("  - /path/to/data/HC/*.parquet")
        print("  - /path/to/data/*.parquet")
        sys.exit(1)

    # Filter datasets if specific ones requested
    if args.datasets:
        requested = set(args.datasets)
        datasets_to_process = [d for d in all_datasets if d["name"] in requested]

        if not datasets_to_process:
            print("\nError: None of the requested datasets found")
            print(f"Requested: {', '.join(args.datasets)}")
            print(f"Available: {', '.join(d['name'] for d in all_datasets)}")
            sys.exit(1)

        found_names = {d["name"] for d in datasets_to_process}
        for name in requested - found_names:
            print(f"Warning: Dataset '{name}' not found, skipping")
    else:
        datasets_to_process = all_datasets

    # Display summary
    print(f"\nFound {len(all_datasets)} dataset(s):")
    for ds in all_datasets:
        marker = "→" if ds in datasets_to_process else " "
        print(f"  {marker} {ds['name']}")

    print(f"\n{'=' * 60}")
    print("Batch ADM Report Generator")
    print(f"{'=' * 60}")
    print(f"Output directory: {args.output.absolute()}")
    print(f"Datasets to process: {len(datasets_to_process)}")
    print(f"Max model reports per dataset: {args.max_models}")

    # Process all datasets
    results = []
    summary_file = args.output / "summary.csv"
    for i, dataset in enumerate(datasets_to_process, 1):
        print(f"\n[{i}/{len(datasets_to_process)}]")
        result = process_dataset(dataset, args.output, max_models=args.max_models)
        results.append(result)

        df_incremental = pl.DataFrame(results)
        df_incremental.write_csv(summary_file)
        print(f"  ✓ Summary updated: {summary_file}")

    # Create summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    df = pl.DataFrame(results)

    summary_table = df.select(
        [
            pl.col("Dataset"),
            pl.col("Model_File_MB").round(1).alias("Input (MB)"),
            pl.col("HC_CDN_MB").round(1).alias("HC CDN"),
            pl.col("HC_Embed_MB").round(1).alias("HC Embed"),
            pl.col("HC_CDN_Status").alias("HC Status"),
            pl.col("ModelReport_Models").alias("# Models"),
            pl.col("ModelReport_CDN_MB").round(1).alias("MR CDN"),
            pl.col("ModelReport_Embed_MB").round(1).alias("MR Embed"),
            pl.col("Excel_MB").round(1).alias("Excel"),
            pl.col("Excel_Status").alias("XLS Status"),
        ]
    )

    print(summary_table)

    # Show HC HTML errors if any
    for mode, col in [("HC CDN", "HC_CDN_Errors"), ("HC Embed", "HC_Embed_Errors")]:
        errors_df = df.filter(pl.col(col).is_not_null())
        if len(errors_df) > 0:
            print(f"\n{'=' * 60}")
            print(f"HTML Errors Detected ({mode})")
            print(f"{'=' * 60}")
            for row in errors_df.iter_rows(named=True):
                print(f"\n{row['Dataset']}:")
                for error in row[col].split("; "):
                    print(f"  - {error}")

    print(f"\n✓ Final summary: {summary_file}")

    # Print statistics
    print(f"\n{'=' * 60}")
    print("Results:")
    for report, prefix in [("HealthCheck", "HC_CDN"), ("HC full-embed", "HC_Embed")]:
        s = (df[f"{prefix}_Status"] == "Success").sum()
        e = (df[f"{prefix}_Status"] == "Success (with errors)").sum()
        f = len(df) - s - e
        print(f"  {report}: {s} success, {e} with errors, {f} failed")
    print(f"  Model reports generated: {df['ModelReport_Models'].sum()} total")
    print(f"  Excel exports: {(df['Excel_Status'] == 'Success').sum()} success")
    print(f"\nTotal HC CDN:    {df['HC_CDN_MB'].sum():.1f} MB")
    print(f"Total HC embed:  {df['HC_Embed_MB'].sum():.1f} MB")
    print(f"Total MR CDN:    {df['ModelReport_CDN_MB'].sum():.1f} MB")
    print(f"Total MR embed:  {df['ModelReport_Embed_MB'].sum():.1f} MB")
    print(f"Total Excel:     {df['Excel_MB'].sum():.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
