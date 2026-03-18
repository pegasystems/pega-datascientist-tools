#!/usr/bin/env python
"""Batch HealthCheck Report Generator.

Generate ADM HealthCheck reports for multiple datasets.

This script discovers ADM model and predictor data files, generates HealthCheck
reports, and creates a summary of results with error detection.

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
from pathlib import Path
from datetime import datetime
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


def process_dataset(
    dataset: dict,
    output_dir: Path,
) -> dict:
    """Process a single dataset and generate HealthCheck report.

    Parameters
    ----------
    dataset : dict
        Dataset information (name, data_dir, model_file, predictor_file)
    output_dir : Path
        Directory for output reports

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
        "HTML_File_MB": 0.0,
        "Status": "Not Found",
        "Error": None,
        "HTML_Errors": None,
    }

    model_file = dataset["model_file"]
    predictor_file = dataset["predictor_file"]

    # Get input file sizes
    result["Model_File_MB"] = get_file_size_mb(model_file)
    result["Predictor_File_MB"] = get_file_size_mb(predictor_file)

    print(f"  ✓ Model file: {result['Model_File_MB']:.1f} MB")
    if predictor_file:
        print(f"  ✓ Predictor file: {result['Predictor_File_MB']:.1f} MB")
    else:
        print("  ℹ No predictor file found")

    try:
        # Create ADMDatamart
        print("  → Loading datamart...")
        datamart = ADMDatamart.from_ds_export(
            model_filename=str(model_file),
            predictor_filename=str(predictor_file) if predictor_file else None,
        )

        print(f"  ✓ Datamart loaded: {len(datamart.model_data.collect())} models")

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate HealthCheck report
        print("  → Generating HealthCheck report...")
        output_path = datamart.generate.health_check(
            name=name.lower().replace(" ", "_").replace(".", "_"),
            title=f"ADM Health Check - {name}",
            subtitle=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            output_dir=str(output_dir),
            size_reduction_method="cdn",
        )

        print(f"  ✓ Report generated: {output_path}")

        # Get output file size
        html_path = Path(output_path)
        result["HTML_File_MB"] = get_file_size_mb(html_path)
        print(f"  ✓ Report size: {result['HTML_File_MB']:.1f} MB")

        # Scan HTML for errors
        print("  → Scanning HTML for errors...")
        html_errors = check_report_for_errors(html_path)
        if html_errors:
            result["HTML_Errors"] = "; ".join(html_errors)
            result["Status"] = "Success (with errors)"
            print("  ⚠ HTML contains errors:")
            for error in html_errors:
                print(f"    - {error}")
        else:
            result["Status"] = "Success"
            print("  ✓ No errors found in HTML")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        result["Status"] = "Error"
        result["Error"] = str(e)
        import traceback

        traceback.print_exc()

    return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Batch generate ADM HealthCheck reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/customers
  %(prog)s /path/to/customers --output ./reports
  %(prog)s /path/to/customers --datasets CustomerA CustomerB
  %(prog)s /path/to/single_customer/HC

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

        # Warn about datasets not found
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
    print("Batch HealthCheck Report Generator")
    print(f"{'=' * 60}")
    print(f"Output directory: {args.output.absolute()}")
    print(f"Datasets to process: {len(datasets_to_process)}")

    # Process all datasets
    results = []
    for i, dataset in enumerate(datasets_to_process, 1):
        print(f"\n[{i}/{len(datasets_to_process)}]")
        result = process_dataset(dataset, args.output)
        results.append(result)

    # Create summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    df = pl.DataFrame(results)

    # Format the table for display
    summary_table = df.select(
        [
            pl.col("Dataset"),
            pl.col("Model_File_MB").round(1).alias("Model (MB)"),
            pl.col("Predictor_File_MB").round(1).alias("Predictor (MB)"),
            pl.col("HTML_File_MB").round(1).alias("HTML (MB)"),
            pl.col("Status"),
        ]
    )

    print(summary_table)

    # Show HTML errors if any
    errors_df = df.filter(pl.col("HTML_Errors").is_not_null())
    if len(errors_df) > 0:
        print(f"\n{'=' * 60}")
        print("HTML Errors Detected")
        print(f"{'=' * 60}")
        for row in errors_df.iter_rows(named=True):
            print(f"\n{row['Dataset']}:")
            for error in row["HTML_Errors"].split("; "):
                print(f"  - {error}")

    # Save full summary to CSV
    summary_file = args.output / "summary.csv"
    df.write_csv(summary_file)
    print(f"\n✓ Summary saved to: {summary_file}")

    # Print statistics
    success_count = (df["Status"] == "Success").sum()
    success_with_errors_count = (df["Status"] == "Success (with errors)").sum()
    failed_count = len(df) - success_count - success_with_errors_count
    total_model_mb = df["Model_File_MB"].sum()
    total_html_mb = df["HTML_File_MB"].sum()

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  ✓ Clean success: {success_count}")
    print(f"  ⚠ Success with HTML errors: {success_with_errors_count}")
    print(f"  ✗ Generation failed: {failed_count}")
    print(f"Total input size: {total_model_mb:.1f} MB")
    print(f"Total output size: {total_html_mb:.1f} MB")
    if total_model_mb > 0:
        print(f"Compression ratio: {(total_html_mb / total_model_mb * 100):.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
