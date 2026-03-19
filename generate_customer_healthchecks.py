#!/usr/bin/env python
"""Batch script to generate HealthCheck reports for customer data.

This script:
1. Finds customer data in the OneDrive folder
2. Creates ADMDatamart from parquet files
3. Generates HealthCheck HTML reports
4. Creates a summary table with file sizes

Usage:
    python generate_customer_healthchecks.py                    # Process all customers
    python generate_customer_healthchecks.py Achmea CBA Citi    # Process specific customers
"""

import sys
from pathlib import Path
import polars as pl
from datetime import datetime
from pdstools import ADMDatamart

# Configuration
CUSTOMER_DATA_DIR = Path(
    "/Users/perdo/Library/CloudStorage/OneDrive-SharedLibraries-PegasystemsInc/AI Chapter Data Sets - Documents/Customers"
)
OUTPUT_DIR = Path("./temp_healthcheck_output")
MODEL_FILE = "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
PREDICTOR_FILE = "PR_DATA_DM_ADMMART_PRED.parquet"


def find_available_customers():
    """Find all customers with HC folders and model files.

    Returns
    -------
    list[str]
        List of customer names that have the required data files
    """
    customers = []
    if not CUSTOMER_DATA_DIR.exists():
        print(f"Error: Customer data directory not found: {CUSTOMER_DATA_DIR}")
        return customers

    for customer_dir in sorted(CUSTOMER_DATA_DIR.iterdir()):
        if not customer_dir.is_dir():
            continue

        hc_dir = customer_dir / "HC"
        model_file = hc_dir / MODEL_FILE

        if hc_dir.exists() and model_file.exists():
            customers.append(customer_dir.name)

    return customers


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def scan_html_for_errors(html_path: Path) -> list[str]:
    """Scan generated HTML file for error indicators.

    Parameters
    ----------
    html_path : Path
        Path to the HTML file to scan

    Returns
    -------
    list[str]
        List of error messages found (empty if no errors)
    """
    if not html_path.exists():
        return ["HTML file not found"]

    try:
        content = html_path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Failed to read HTML: {e}"]

    errors = []

    # Common error patterns in HTML output
    error_patterns = [
        ("Error rendering", "Plot rendering error"),
        ("Traceback (most recent call last)", "Python traceback"),
        ("ValueError:", "ValueError exception"),
        ("TypeError:", "TypeError exception"),
        ("KeyError:", "KeyError exception"),
        ("AttributeError:", "AttributeError exception"),
        ("NameError:", "NameError exception"),
        ("Exception:", "Generic exception"),
        ("The given query resulted in an empty dataframe", "Empty dataframe error"),
    ]

    for pattern, description in error_patterns:
        if pattern in content:
            # Count occurrences
            count = content.count(pattern)
            if count > 1:
                errors.append(f"{description} (found {count} times)")
            else:
                errors.append(description)

    return errors


def process_customer(customer_name: str) -> dict:
    """Process a single customer and generate HealthCheck report.

    Parameters
    ----------
    customer_name : str
        Name of the customer folder

    Returns
    -------
    dict
        Dictionary with customer name, file sizes, and status
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {customer_name}")
    print(f"{'=' * 60}")

    # Paths
    customer_dir = CUSTOMER_DATA_DIR / customer_name / "HC"
    model_file = customer_dir / MODEL_FILE
    predictor_file = customer_dir / PREDICTOR_FILE

    result = {
        "Customer": customer_name,
        "Model_File_MB": 0.0,
        "Predictor_File_MB": 0.0,
        "HTML_File_MB": 0.0,
        "Status": "Not Found",
        "Error": None,
        "HTML_Errors": None,
    }

    # Check if files exist
    if not customer_dir.exists():
        print(f"  ✗ HC folder not found: {customer_dir}")
        result["Status"] = "No HC Folder"
        return result

    if not model_file.exists():
        print(f"  ✗ Model file not found: {model_file}")
        result["Status"] = "No Model File"
        return result

    # Get input file sizes
    result["Model_File_MB"] = get_file_size_mb(model_file)
    result["Predictor_File_MB"] = get_file_size_mb(predictor_file) if predictor_file.exists() else 0.0

    print(f"  ✓ Model file: {result['Model_File_MB']:.1f} MB")
    if predictor_file.exists():
        print(f"  ✓ Predictor file: {result['Predictor_File_MB']:.1f} MB")
    else:
        print("  ℹ No predictor file found")

    try:
        # Create ADMDatamart
        print("  → Loading datamart...")
        datamart = ADMDatamart.from_ds_export(
            model_filename=str(model_file),
            predictor_filename=str(predictor_file) if predictor_file.exists() else None,
        )

        print(f"  ✓ Datamart loaded: {len(datamart.model_data.collect())} models")

        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Generate HealthCheck report
        print("  → Generating HealthCheck report...")
        output_path = datamart.generate.health_check(
            name=customer_name.lower().replace(" ", "_"),
            title=f"ADM Health Check - {customer_name}",
            subtitle=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            output_dir=str(OUTPUT_DIR),
            size_reduction_method="cdn",  # Use CDN to reduce file size
        )

        print(f"  ✓ Report generated: {output_path}")

        # Get output file size
        html_path = Path(output_path)
        result["HTML_File_MB"] = get_file_size_mb(html_path)
        print(f"  ✓ Report size: {result['HTML_File_MB']:.1f} MB")

        # Scan HTML for errors
        print("  → Scanning HTML for errors...")
        html_errors = scan_html_for_errors(html_path)
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
    # Determine which customers to process
    available_customers = find_available_customers()

    if len(sys.argv) > 1:
        # Process specific customers from command line
        requested_customers = sys.argv[1:]
        customers_to_process = []

        for customer in requested_customers:
            if customer in available_customers:
                customers_to_process.append(customer)
            else:
                print(f"Warning: Customer '{customer}' not found or has no data. Skipping.")

        if not customers_to_process:
            print("Error: No valid customers specified.")
            print(f"Available customers: {', '.join(available_customers[:10])}...")
            return
    else:
        # Process all available customers
        customers_to_process = available_customers

    print(f"\n{'=' * 60}")
    print("Customer HealthCheck Report Generator")
    print(f"{'=' * 60}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Available customers: {len(available_customers)}")
    print(f"Customers to process: {len(customers_to_process)}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Process all customers
    results = []
    for i, customer in enumerate(customers_to_process, 1):
        print(f"\n[{i}/{len(customers_to_process)}]")
        result = process_customer(customer)
        results.append(result)

    # Create summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    df = pl.DataFrame(results)

    # Format the table for display
    summary_table = df.select(
        [
            pl.col("Customer"),
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
            print(f"\n{row['Customer']}:")
            for error in row["HTML_Errors"].split("; "):
                print(f"  - {error}")

    # Save full summary to CSV
    summary_file = OUTPUT_DIR / "summary.csv"
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
    print(f"Compression ratio: {(total_html_mb / total_model_mb * 100) if total_model_mb > 0 else 0:.1f}%")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    main()
