#!/usr/bin/env python
"""Batch ADM Report Generator.

Generate ADM HealthCheck reports, Model Reports, and Excel exports for
multiple datasets.

This script discovers ADM model, predictor, and prediction data files, generates reports
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
- Prediction file: PR_DATA_DM_SNAPSHOTS.parquet (optional, or *SNAPSHOTS.parquet)
"""

import argparse
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from pdstools import ADMDatamart, Prediction
from pdstools.utils.report_utils import check_report_for_errors, is_esbuild_available

# Default file name patterns
MODEL_FILE_PATTERNS = ["PR_DATA_DM_ADMMART_MDL_FACT.parquet", "*MDL_FACT.parquet"]
PREDICTOR_FILE_PATTERNS = ["PR_DATA_DM_ADMMART_PRED.parquet", "*PRED.parquet"]
PREDICTION_FILE_PATTERNS = ["PR_DATA_DM_SNAPSHOTS.parquet", "*SNAPSHOTS.parquet"]


def _first_matching_file(directory: Path, patterns: list[str]) -> Path | None:
    """Return the first file matching the configured pattern priority."""
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _dataset_in_directory(name: str, directory: Path) -> dict | None:
    """Build one dataset entry when a directory contains model data."""
    model_file = _first_matching_file(directory, MODEL_FILE_PATTERNS)
    if model_file is None:
        return None
    return {
        "name": name,
        "data_dir": directory,
        "model_file": model_file,
        "predictor_file": _first_matching_file(directory, PREDICTOR_FILE_PATTERNS),
        "prediction_file": _first_matching_file(directory, PREDICTION_FILE_PATTERNS),
    }


def find_data_directories(root_path: Path) -> list[dict]:
    """Discover directories containing ADM data files.

    Parameters
    ----------
    root_path : Path
        Root directory to search for data

    Returns
    -------
    list[dict]
        List of dictionaries with keys: name, data_dir, model_file,
        predictor_file, and prediction_file.
    """
    datasets = []

    root_dataset = _dataset_in_directory(root_path.name, root_path)
    if root_dataset is not None:
        return [root_dataset]

    # Check for HC subdirectory at root level
    hc_dir = root_path / "HC"
    if hc_dir.exists() and hc_dir.is_dir():
        root_hc_dataset = _dataset_in_directory(root_path.name, hc_dir)
        if root_hc_dataset is not None:
            return [root_hc_dataset]

    # Search subdirectories for HC folders or direct data
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Check subdir/HC pattern
        hc_dir = subdir / "HC"
        if hc_dir.exists() and hc_dir.is_dir():
            dataset = _dataset_in_directory(subdir.name, hc_dir)
            if dataset is not None:
                datasets.append(dataset)

        # Check subdir directly for data files
        if not any(d["name"] == subdir.name for d in datasets):
            dataset = _dataset_in_directory(subdir.name, subdir)
            if dataset is not None:
                datasets.append(dataset)

    return datasets


def get_file_size_mb(file_path: Path | None) -> float:
    """Get file size in MB."""
    if file_path and file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def _path_or_none(path: Path | None) -> str | None:
    """Return a string path for CSV output, preserving missing optional files."""
    return str(path) if path else None


def _print_report_size_comparison(label: str, cdn_mb: float, embed_mb: float) -> None:
    """Print CDN vs full-embed report sizes and flag inverted size ordering."""
    if cdn_mb <= 0 or embed_mb <= 0:
        return

    ratio = embed_mb / cdn_mb
    print(f"  ℹ {label} size: CDN {cdn_mb:.1f} MB vs embed {embed_mb:.1f} MB ({ratio:.1f}x)")
    if embed_mb < cdn_mb:
        print(
            f"  ⚠ {label} full-embed output is smaller than CDN output; "
            "file sizes depend on Quarto/esbuild rendering and report content."
        )


def _print_dataset_paths(row: dict) -> None:
    """Print input paths for an errored dataset summary row."""
    path_fields = [
        ("Data directory", "Data_Dir"),
        ("Model file", "Model_File"),
        ("Predictor file", "Predictor_File"),
        ("Prediction file", "Prediction_File"),
    ]
    for label, field in path_fields:
        path = row.get(field)
        if path:
            print(f"  {label}: {path}")


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
    if datamart.predictor_data is None:
        print("  ℹ No predictor data — skipping model selection")
        return []

    if datamart.combined_data is None:
        print("  ℹ No combined data — skipping model selection")
        return []

    group_keys = [c for c in ["Channel", "Direction", "Issue"] if c in datamart.combined_data.collect_schema().names()]

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

    # Model reports use the reachable classifier range for score distribution
    # and AUC. Skip models whose computed range contains no classifier bins.
    active_ranges = (
        datamart.active_ranges(nb_model_ids)
        .select(
            "ModelID",
            "AUC_ActiveRange",
            "idx_min",
            "idx_max",
        )
        .collect()
    )
    nb_model_ids = active_ranges.filter(
        pl.col("AUC_ActiveRange").is_not_null() & (pl.col("idx_min") < pl.col("idx_max"))
    )["ModelID"].to_list()

    if not nb_model_ids:
        print("  ℹ No models with a non-empty active classifier range found")
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


def _active_nb_model_ids(datamart: ADMDatamart) -> set[str]:
    """Return model IDs that have NB predictor bins and classifier rows."""
    if datamart.predictor_data is None:
        return set()

    predictor_data = datamart.predictor_data
    nb_model_ids = set(
        predictor_data.filter((pl.col("BinType") != "NONE") & (pl.col("EntryType") != "Classifier"))
        .select(pl.col("ModelID").unique())
        .collect()["ModelID"]
        .to_list()
    )
    classifier_model_ids = set(
        predictor_data.filter(pl.col("EntryType") == "Classifier")
        .select(pl.col("ModelID").unique())
        .collect()["ModelID"]
        .to_list()
    )
    return nb_model_ids & classifier_model_ids


def _compute_ci_maturity_analysis(
    datamart: ADMDatamart,
    *,
    active_window_days: int,
    positives_maturity_threshold: int,
) -> tuple[dict[str, float | int | str | None], pl.DataFrame]:
    """Compute optional maturity-versus-CI analysis for active NB models."""
    import numpy as np

    def _float_or_none(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))

    if datamart.model_data is None:
        metrics = {
            "Active_NB_Models": 0,
            "Active_NB_Models_With_CI": 0,
            "Maturity_Pct_Above_Threshold": None,
            "CI_Width_Mean": None,
            "CI_Width_Median": None,
            "CI_Width_P90": None,
            "CI_Width_Mean_AboveThreshold": None,
            "CI_Width_Mean_AtOrBelowThreshold": None,
            "CI_Width_Ratio_AtOrBelow_over_Above": None,
            "Positives_vs_CI_Width_Spearman": None,
        }
        return metrics, pl.DataFrame()

    nb_model_ids = _active_nb_model_ids(datamart)
    if not nb_model_ids:
        metrics = {
            "Active_NB_Models": 0,
            "Active_NB_Models_With_CI": 0,
            "Maturity_Pct_Above_Threshold": None,
            "CI_Width_Mean": None,
            "CI_Width_Median": None,
            "CI_Width_P90": None,
            "CI_Width_Mean_AboveThreshold": None,
            "CI_Width_Mean_AtOrBelowThreshold": None,
            "CI_Width_Ratio_AtOrBelow_over_Above": None,
            "Positives_vs_CI_Width_Spearman": None,
        }
        return metrics, pl.DataFrame()

    model_columns = datamart.model_data.collect_schema().names()
    select_exprs = [pl.col("ModelID")]
    if "Positives" in model_columns:
        select_exprs.append(pl.col("Positives").cast(pl.Float64))
    else:
        select_exprs.append(pl.lit(0.0).alias("Positives"))
    if "ResponseCount" in model_columns:
        select_exprs.append(pl.col("ResponseCount").cast(pl.Float64))
    else:
        select_exprs.append(pl.lit(0.0).alias("ResponseCount"))
    if "SnapshotTime" in model_columns:
        snapshot_expr = pl.col("SnapshotTime")
        if datamart.model_data.collect_schema().get("SnapshotTime") == pl.Utf8:
            snapshot_expr = snapshot_expr.str.strptime(pl.Datetime, strict=False)
        select_exprs.append(snapshot_expr.alias("SnapshotTime"))
    else:
        select_exprs.append(pl.lit(None).alias("SnapshotTime"))

    model_rows = datamart.model_data.filter(pl.col("ModelID").is_in(list(nb_model_ids))).select(select_exprs).collect()

    if model_rows.height == 0:
        metrics = {
            "Active_NB_Models": 0,
            "Active_NB_Models_With_CI": 0,
            "Maturity_Pct_Above_Threshold": None,
            "CI_Width_Mean": None,
            "CI_Width_Median": None,
            "CI_Width_P90": None,
            "CI_Width_Mean_AboveThreshold": None,
            "CI_Width_Mean_AtOrBelowThreshold": None,
            "CI_Width_Ratio_AtOrBelow_over_Above": None,
            "Positives_vs_CI_Width_Spearman": None,
        }
        return metrics, pl.DataFrame()

    reference_timestamp = model_rows.get_column("SnapshotTime").drop_nulls().max()
    if isinstance(reference_timestamp, datetime):
        cutoff = reference_timestamp - timedelta(days=active_window_days)
        active_candidates = model_rows.filter(
            (pl.col("SnapshotTime") >= cutoff)
            & (pl.col("SnapshotTime") <= reference_timestamp)
            & (pl.col("ResponseCount") > 0)
        )
    else:
        active_candidates = model_rows.filter(pl.col("ResponseCount") > 0)

    active_agg = active_candidates.group_by("ModelID").agg(
        Positives=pl.col("Positives").sum(),
        ResponseCount=pl.col("ResponseCount").sum(),
    )

    if active_agg.height == 0:
        metrics = {
            "Active_NB_Models": 0,
            "Active_NB_Models_With_CI": 0,
            "Maturity_Pct_Above_Threshold": 0.0,
            "CI_Width_Mean": None,
            "CI_Width_Median": None,
            "CI_Width_P90": None,
            "CI_Width_Mean_AboveThreshold": None,
            "CI_Width_Mean_AtOrBelowThreshold": None,
            "CI_Width_Ratio_AtOrBelow_over_Above": None,
            "Positives_vs_CI_Width_Spearman": None,
        }
        return metrics, pl.DataFrame()

    analysis_rows: list[dict] = []
    for row in active_agg.iter_rows(named=True):
        model_id = row["ModelID"]
        ci_data = {
            "AUC_ActiveRange": None,
            "AUC_ActiveRange_CI_Lower": None,
            "AUC_ActiveRange_CI_Upper": None,
            "AUC_ActiveRange_CI_Available": False,
            "AUC_ActiveRange_CI_Reason": "analysis_error",
        }
        try:
            ar = (
                datamart.active_ranges(model_id)
                .collect()
                .select(
                    "AUC_ActiveRange",
                    "AUC_ActiveRange_CI_Lower",
                    "AUC_ActiveRange_CI_Upper",
                    "AUC_ActiveRange_CI_Available",
                    "AUC_ActiveRange_CI_Reason",
                )
            )
            if ar.height > 0:
                ci_data = {
                    "AUC_ActiveRange": ar["AUC_ActiveRange"][0],
                    "AUC_ActiveRange_CI_Lower": ar["AUC_ActiveRange_CI_Lower"][0],
                    "AUC_ActiveRange_CI_Upper": ar["AUC_ActiveRange_CI_Upper"][0],
                    "AUC_ActiveRange_CI_Available": ar["AUC_ActiveRange_CI_Available"][0],
                    "AUC_ActiveRange_CI_Reason": ar["AUC_ActiveRange_CI_Reason"][0],
                }
        except Exception:
            # Keep model-level maturity rows even when CI cannot be computed.
            pass

        analysis_rows.append(
            {
                "ModelID": model_id,
                "Positives": row["Positives"],
                "ResponseCount": row["ResponseCount"],
                "IsActiveLast30Days": True,
                **ci_data,
            }
        )

    model_level = pl.DataFrame(analysis_rows).with_columns(
        CI_Width=(pl.col("AUC_ActiveRange_CI_Upper") - pl.col("AUC_ActiveRange_CI_Lower")),
        PositivesSegment=pl.when(pl.col("Positives") > positives_maturity_threshold)
        .then(pl.lit(f">{positives_maturity_threshold}"))
        .otherwise(pl.lit(f"<={positives_maturity_threshold}")),
        MaturitySegmentAboveThreshold=pl.col("Positives") > positives_maturity_threshold,
    )

    active_count = model_level.height
    with_ci_count = model_level.filter(pl.col("CI_Width").is_not_null()).height
    above_threshold_count = model_level.filter(pl.col("Positives") > positives_maturity_threshold).height

    ci_non_null = model_level.filter(pl.col("CI_Width").is_not_null())
    mean_above = _float_or_none(
        ci_non_null.filter(pl.col("Positives") > positives_maturity_threshold).get_column("CI_Width").mean()
    )
    mean_at_or_below = _float_or_none(
        ci_non_null.filter(pl.col("Positives") <= positives_maturity_threshold).get_column("CI_Width").mean()
    )
    ratio = (
        (mean_at_or_below / mean_above)
        if mean_above is not None and mean_above > 0 and mean_at_or_below is not None
        else None
    )

    corr_df = ci_non_null.select("Positives", "CI_Width")
    spearman = None
    if corr_df.height >= 2:
        positives_rank = corr_df.get_column("Positives").rank().to_numpy()
        ci_rank = corr_df.get_column("CI_Width").rank().to_numpy()
        corr = np.corrcoef(positives_rank, ci_rank)[0, 1]
        if np.isfinite(corr):
            spearman = float(corr)

    metrics = {
        "Active_NB_Models": active_count,
        "Active_NB_Models_With_CI": with_ci_count,
        "Maturity_Pct_Above_Threshold": (100.0 * above_threshold_count / active_count) if active_count > 0 else 0.0,
        "CI_Width_Mean": _float_or_none(ci_non_null.get_column("CI_Width").mean()) if ci_non_null.height > 0 else None,
        "CI_Width_Median": _float_or_none(ci_non_null.get_column("CI_Width").median())
        if ci_non_null.height > 0
        else None,
        "CI_Width_P90": _float_or_none(ci_non_null.get_column("CI_Width").quantile(0.9))
        if ci_non_null.height > 0
        else None,
        "CI_Width_Mean_AboveThreshold": mean_above,
        "CI_Width_Mean_AtOrBelowThreshold": mean_at_or_below,
        "CI_Width_Ratio_AtOrBelow_over_Above": ratio,
        "Positives_vs_CI_Width_Spearman": spearman,
    }

    return metrics, model_level


def _generate_ci_maturity_plots(
    model_level_df: pl.DataFrame,
    *,
    output_dir: Path,
    positives_maturity_threshold: int,
) -> list[Path]:
    """Generate CI maturity scatter plots from model-level analysis rows.

    Writes a linear-x, log-x, and capped-x(<=10k positives) variant when
    enough data is available.
    """
    if model_level_df.is_empty() or "CI_Width" not in model_level_df.columns:
        return []

    plot_df = model_level_df.filter(
        pl.col("CI_Width").is_not_null() & pl.col("Positives").is_not_null() & (pl.col("Positives") > 0)
    ).with_columns((pl.col("Positives") > positives_maturity_threshold).alias("gt_threshold"))

    if plot_df.height == 0:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ℹ matplotlib not installed — skipping CI maturity PNG generation")
        return []

    segment_summary = (
        plot_df.group_by("gt_threshold")
        .agg(
            pl.len().alias("n_models"),
            pl.col("CI_Width").mean().alias("mean_ci_width"),
        )
        .sort("gt_threshold", descending=True)
    )

    mean_above = None
    n_above = 0
    mean_below_or_equal = None
    n_below_or_equal = 0

    row_above = segment_summary.filter(pl.col("gt_threshold"))
    if row_above.height > 0:
        mean_above = float(row_above["mean_ci_width"][0])
        n_above = int(row_above["n_models"][0])

    row_below_or_equal = segment_summary.filter(~pl.col("gt_threshold"))
    if row_below_or_equal.height > 0:
        mean_below_or_equal = float(row_below_or_equal["mean_ci_width"][0])
        n_below_or_equal = int(row_below_or_equal["n_models"][0])

    def _render_scatter_with_means(
        frame: pl.DataFrame,
        out_path: Path,
        *,
        title: str,
        x_label: str,
        use_log_x: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#1f77b4" if value else "#ff7f0e" for value in frame["gt_threshold"].to_list()]
        ax.scatter(frame["Positives"].to_list(), frame["CI_Width"].to_list(), c=colors, alpha=0.8)
        ax.axvline(positives_maturity_threshold, color="red", linestyle="--", linewidth=1.5)

        if mean_above is not None:
            ax.axhline(mean_above, color="#1f77b4", linestyle="--", linewidth=2)
            ax.text(
                0.99,
                mean_above,
                f"blue mean (>{positives_maturity_threshold}, n={n_above}): {mean_above:.4f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="#1f77b4",
            )

        if mean_below_or_equal is not None:
            ax.axhline(mean_below_or_equal, color="#ff7f0e", linestyle="--", linewidth=2)
            ax.text(
                0.99,
                mean_below_or_equal,
                f"orange mean (<={positives_maturity_threshold}, n={n_below_or_equal}): {mean_below_or_equal:.4f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="#ff7f0e",
            )

        if use_log_x:
            ax.set_xscale("log")

        ax.set_xlabel(x_label)
        ax.set_ylabel("CI Width")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    linear_path = output_dir / "ci_maturity_vs_confidence_intervals.png"
    _render_scatter_with_means(
        plot_df,
        linear_path,
        title="Model CI Width vs Positives",
        x_label="Positives",
    )
    output_files.append(linear_path)

    logx_path = output_dir / "ci_maturity_vs_confidence_intervals_logx.png"
    _render_scatter_with_means(
        plot_df,
        logx_path,
        title="Model CI Width vs Positives (Log X)",
        x_label="Positives (log scale)",
        use_log_x=True,
    )
    output_files.append(logx_path)

    capped_df = plot_df.filter(pl.col("Positives") <= 10000)
    if capped_df.height > 0:
        capped_path = output_dir / "ci_maturity_vs_confidence_intervals_cap10k.png"
        _render_scatter_with_means(
            capped_df,
            capped_path,
            title="Model CI Width vs Positives (Capped at 10k)",
            x_label="Positives (<=10,000)",
        )
        output_files.append(capped_path)

    return output_files


def _generate_cross_dataset_ci_width_plot(
    model_level_df: pl.DataFrame,
    *,
    output_dir: Path,
) -> Path | None:
    """Generate a pooled log-log CI-width versus positives plot.

    Only models with positive outcomes and a positive, available CI width are
    included. The fitted relationship is reported as a power law on the plot.
    """
    if model_level_df.is_empty() or "CI_Width" not in model_level_df.columns:
        return None

    plot_df = model_level_df.filter(
        pl.col("CI_Width").is_not_null()
        & (pl.col("CI_Width") > 0)
        & pl.col("Positives").is_not_null()
        & (pl.col("Positives") > 0)
    )
    if plot_df.height < 2:
        return None

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  ℹ matplotlib/numpy not installed — skipping pooled CI width plot")
        return None

    positives = plot_df["Positives"].to_numpy()
    ci_width = plot_df["CI_Width"].to_numpy()
    log_positives = np.log10(positives)
    log_ci_width = np.log10(ci_width)
    slope, intercept = np.polyfit(log_positives, log_ci_width, 1)
    fitted = slope * log_positives + intercept
    r_squared = 1 - np.sum((log_ci_width - fitted) ** 2) / np.sum((log_ci_width - log_ci_width.mean()) ** 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ci_width_vs_positives_all_datasets.png"
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    datasets = plot_df.get_column("Dataset").unique().sort().to_list() if "Dataset" in plot_df else ["All datasets"]
    color_by_dataset = {name: colors[index % len(colors)] for index, name in enumerate(datasets)}

    if "Dataset" in plot_df:
        for dataset in datasets:
            points = plot_df.filter(pl.col("Dataset") == dataset)
            ax.scatter(
                points["Positives"].to_list(),
                points["CI_Width"].to_list(),
                s=14,
                alpha=0.4,
                color=color_by_dataset[dataset],
                label=dataset,
            )
    else:
        ax.scatter(positives, ci_width, s=14, alpha=0.4, color=colors[0], label="All datasets")

    fit_x = np.logspace(log_positives.min(), log_positives.max(), 200)
    fit_y = 10 ** (intercept + slope * np.log10(fit_x))
    ax.plot(
        fit_x,
        fit_y,
        color="black",
        linewidth=2,
        label=f"Power-law fit (R²={r_squared:.2f})",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Positive outcomes per model")
    ax.set_ylabel("AUC CI width")
    ax.set_title("AUC confidence interval width versus positive volume")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Cross-dataset CI plot: {output_path} (n={plot_df.height}, slope={slope:.3f}, R²={r_squared:.2f})")
    return output_path


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
            return size_mb, "Error", errors_str

        print(f"  ✓ No errors in {label} ({mode})")
        return size_mb, "Success", None

    except Exception as e:
        print(f"  ✗ Error in {label} ({mode}): {e}")
        traceback.print_exc()
        return 0.0, "Error", str(e)


def process_dataset(
    dataset: dict,
    output_dir: Path | None,
    *,
    max_models: int = 3,
    active_window_days: int = 30,
    positives_maturity_threshold: int = 200,
    ci_maturity_dataset_rows: list[dict] | None = None,
    ci_maturity_model_rows: list[dict] | None = None,
) -> dict:
    """Process a single dataset and generate all reports.

    Generates HealthCheck reports (CDN + full-embed), Model Reports for
    selected interesting models, and an Excel export.

    Parameters
    ----------
    dataset : dict
        Dataset information (name, data_dir, model_file, predictor_file,
        prediction_file)
    output_dir : Path, optional
        Directory for output reports. If None, writes to the dataset data directory.
    max_models : int
        Maximum number of model reports to generate
    active_window_days : int, default=30
        Trailing-day window used to classify active NB models.
    positives_maturity_threshold : int, default=200
        Positives threshold used for maturity segmentation.
    ci_maturity_dataset_rows : list[dict] | None, optional
        Optional collector receiving one dataset-level maturity metrics row.
    ci_maturity_model_rows : list[dict] | None, optional
        Optional collector receiving per-model maturity analysis rows.

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

    model_file = dataset["model_file"]
    predictor_file = dataset["predictor_file"]
    prediction_file = dataset.get("prediction_file")

    result = {
        "Dataset": name,
        "Data_Dir": _path_or_none(dataset["data_dir"]),
        "Model_File": _path_or_none(model_file),
        "Predictor_File": _path_or_none(predictor_file),
        "Prediction_File": _path_or_none(prediction_file),
        "Model_File_MB": 0.0,
        "Predictor_File_MB": 0.0,
        "Prediction_File_MB": 0.0,
        "HC_CDN_MB": 0.0,
        "HC_CDN_Status": "Not Found",
        "HC_CDN_Errors": None,
        "HC_Embed_MB": 0.0,
        "HC_Embed_Status": "Not Found",
        "HC_Embed_Errors": None,
        "ModelReport_Models": 0,
        "ModelReport_CDN_MB": 0.0,
        "ModelReport_CDN_Status": "Skipped",
        "ModelReport_CDN_Errors": None,
        "ModelReport_Embed_MB": 0.0,
        "ModelReport_Embed_Status": "Skipped",
        "ModelReport_Embed_Errors": None,
        "Excel_MB": 0.0,
        "Excel_Status": "Skipped",
    }

    result["Model_File_MB"] = get_file_size_mb(model_file)
    result["Predictor_File_MB"] = get_file_size_mb(predictor_file)
    result["Prediction_File_MB"] = get_file_size_mb(prediction_file)

    print(f"  ✓ Model file: {result['Model_File_MB']:.1f} MB")
    if predictor_file:
        print(f"  ✓ Predictor file: {result['Predictor_File_MB']:.1f} MB")
    else:
        print("  ℹ No predictor file found")
    if prediction_file:
        print(f"  ✓ Prediction file: {result['Prediction_File_MB']:.1f} MB")
    else:
        print("  ℹ No prediction file found")

    try:
        print("  → Loading datamart...")
        datamart = ADMDatamart.from_ds_export(
            model_filename=str(model_file),
            predictor_filename=str(predictor_file) if predictor_file else None,
        )
        prediction = Prediction.from_ds_export(str(prediction_file)) if prediction_file else None
        n_models = len(datamart.model_data.collect()) if datamart.model_data is not None else 0
        print(f"  ✓ Datamart loaded: {n_models} models")

        output_dir = Path(dataset["data_dir"]) if output_dir is None else output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Full-embed rendering requires esbuild (Quarto bundles JavaScript for
        # self-contained HTML). Hardened environments (e.g. DJS Docker images)
        # ship Quarto without esbuild, so skip full-embed there rather than
        # letting Quarto fail mid-render and marking the run as failed. See #620.
        esbuild_available = is_esbuild_available()
        if not esbuild_available:
            print("  ℹ esbuild unavailable — full-embed reports will be skipped (CDN-only environment)")

        # ── HealthCheck reports (CDN + full-embed) ──────────────────
        for full_embed, key_prefix in [(False, "HC_CDN"), (True, "HC_Embed")]:
            if full_embed and not esbuild_available:
                result[f"{key_prefix}_Status"] = "Skipped"
                continue
            suffix = "_full" if full_embed else "_cdn"
            mb, status, errors = _generate_quarto_report(
                datamart.generate.health_check,
                "HealthCheck",
                output_dir,
                full_embed=full_embed,
                name=safe_name + suffix,
                title=f"ADM Health Check - {name}",
                subtitle=f"Generated on {timestamp}",
                prediction=prediction,
                model_file_path=model_file,
                predictor_file_path=predictor_file,
                prediction_file_path=prediction_file,
            )
            result[f"{key_prefix}_MB"] = mb
            result[f"{key_prefix}_Status"] = status
            result[f"{key_prefix}_Errors"] = errors

        _print_report_size_comparison("HC", result["HC_CDN_MB"], result["HC_Embed_MB"])

        # ── Model reports for interesting models ────────────────────
        selected_models = select_interesting_models(datamart, max_n=max_models)
        result["ModelReport_Models"] = len(selected_models)

        if selected_models:
            for full_embed, key_prefix in [(False, "ModelReport_CDN"), (True, "ModelReport_Embed")]:
                if full_embed and not esbuild_available:
                    result[f"{key_prefix}_Status"] = "Skipped"
                    continue
                suffix = "_full" if full_embed else "_cdn"
                total_mb = 0.0
                mode_errors = []
                for i, model_id in enumerate(selected_models, start=1):
                    mb, status, errors = _generate_quarto_report(
                        datamart.generate.model_reports,
                        f"ModelReport ({i}/{len(selected_models)})",
                        output_dir,
                        full_embed=full_embed,
                        model_ids=model_id,
                        name=f"{safe_name}_model{suffix}",
                        title=f"Model Report - {name}",
                        subtitle=f"Generated on {timestamp}",
                    )
                    total_mb += mb
                    if status == "Error":
                        mode_errors.append(f"{model_id}: {errors}")

                result[f"{key_prefix}_MB"] = total_mb
                result[f"{key_prefix}_Status"] = "Error" if mode_errors else "Success"
                result[f"{key_prefix}_Errors"] = "; ".join(mode_errors) if mode_errors else None

            _print_report_size_comparison(
                "Model report",
                result["ModelReport_CDN_MB"],
                result["ModelReport_Embed_MB"],
            )

        print("  → Computing CI maturity analysis...")
        ci_metrics, ci_model_df = _compute_ci_maturity_analysis(
            datamart,
            active_window_days=active_window_days,
            positives_maturity_threshold=positives_maturity_threshold,
        )
        result.update(ci_metrics)
        if ci_maturity_dataset_rows is not None:
            dataset_row = {"Dataset": name, **ci_metrics}
            ci_maturity_dataset_rows.append(dataset_row)
        if ci_maturity_model_rows is not None and ci_model_df.height > 0:
            ci_maturity_model_rows.extend(ci_model_df.with_columns(Dataset=pl.lit(name)).to_dicts())

        # Write per-dataset CI maturity plots directly into the dataset HC/data
        # directory so they live next to that dataset's report artifacts.
        dataset_plot_paths = _generate_ci_maturity_plots(
            ci_model_df,
            output_dir=Path(dataset["data_dir"]),
            positives_maturity_threshold=positives_maturity_threshold,
        )
        for plot_path in dataset_plot_paths:
            print(f"  ✓ CI maturity plot: {plot_path}")

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
        default=None,
        help="Output directory for generated reports (default: each dataset's HC/data directory)",
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
    parser.add_argument(
        "--active-window-days",
        type=int,
        default=30,
        help="Trailing active window in days for active NB model definition (default: 30)",
    )
    parser.add_argument(
        "--positives-maturity-threshold",
        type=int,
        default=200,
        help="Positives threshold for maturity segmentation (default: 200)",
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
        print(f"  Prediction: {', '.join(PREDICTION_FILE_PATTERNS)}")
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
    if args.output is None:
        print("Output directory: each dataset's HC/data directory")
    else:
        print(f"Output directory: {args.output.absolute()}")
    print(f"Datasets to process: {len(datasets_to_process)}")
    print(f"Max model reports per dataset: {args.max_models}")
    print("CI maturity analysis: enabled")

    # Process all datasets
    results = []
    ci_maturity_dataset_rows: list[dict] | None = []
    ci_maturity_model_rows: list[dict] | None = []
    summary_dir = args.output if args.output is not None else args.data_path
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "summary.csv"
    for i, dataset in enumerate(datasets_to_process, 1):
        print(f"\n[{i}/{len(datasets_to_process)}]")
        result = process_dataset(
            dataset,
            args.output,
            max_models=args.max_models,
            active_window_days=args.active_window_days,
            positives_maturity_threshold=args.positives_maturity_threshold,
            ci_maturity_dataset_rows=ci_maturity_dataset_rows,
            ci_maturity_model_rows=ci_maturity_model_rows,
        )
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

    # Show rendered-report errors if any
    for mode, col in [
        ("HC CDN", "HC_CDN_Errors"),
        ("HC Embed", "HC_Embed_Errors"),
        ("ModelReport CDN", "ModelReport_CDN_Errors"),
        ("ModelReport Embed", "ModelReport_Embed_Errors"),
    ]:
        errors_df = df.filter(pl.col(col).is_not_null())
        if len(errors_df) > 0:
            print(f"\n{'=' * 60}")
            print(f"Report Errors Detected ({mode})")
            print(f"{'=' * 60}")
            for row in errors_df.iter_rows(named=True):
                print(f"\n{row['Dataset']}:")
                _print_dataset_paths(row)
                for error in row[col].split("; "):
                    print(f"  - {error}")

    print(f"\n✓ Final summary: {summary_file}")

    dataset_summary_file = summary_dir / "ci_maturity_dataset_summary.csv"
    model_level_file = summary_dir / "ci_maturity_model_level.csv"
    if ci_maturity_dataset_rows:
        pl.DataFrame(ci_maturity_dataset_rows).write_csv(dataset_summary_file)
        print(f"✓ CI maturity dataset summary: {dataset_summary_file}")
    if ci_maturity_model_rows:
        ci_model_df = pl.DataFrame(ci_maturity_model_rows)
        ci_model_df.write_csv(model_level_file)
        print(f"✓ CI maturity model-level output: {model_level_file}")
        _generate_cross_dataset_ci_width_plot(
            ci_model_df,
            output_dir=summary_dir,
        )

    # Print statistics
    print(f"\n{'=' * 60}")
    print("Results:")
    generated_report_prefixes = [
        ("HealthCheck", "HC_CDN"),
        ("HealthCheck full-embed", "HC_Embed"),
        ("Model reports", "ModelReport_CDN"),
        ("Model reports full-embed", "ModelReport_Embed"),
    ]
    for report, prefix in generated_report_prefixes:
        s = int((df[f"{prefix}_Status"] == "Success").sum())
        skipped = int((df[f"{prefix}_Status"] == "Skipped").sum())
        failed = len(df) - s - skipped
        print(f"  {report}: {s} success, {skipped} skipped, {failed} failed")
    print(f"  Model reports generated: {df['ModelReport_Models'].sum()} total")
    print(f"  Excel exports: {(df['Excel_Status'] == 'Success').sum()} success")
    print(f"\nTotal HC CDN:    {df['HC_CDN_MB'].sum():.1f} MB")
    print(f"Total HC embed:  {df['HC_Embed_MB'].sum():.1f} MB")
    print(f"Total MR CDN:    {df['ModelReport_CDN_MB'].sum():.1f} MB")
    print(f"Total MR embed:  {df['ModelReport_Embed_MB'].sum():.1f} MB")
    print(f"Total Excel:     {df['Excel_MB'].sum():.1f} MB")
    print(f"{'=' * 60}")

    failed_reports = df.select(
        [
            (pl.col(f"{prefix}_Status") != "Success") & (pl.col(f"{prefix}_Status") != "Skipped")
            for _, prefix in generated_report_prefixes
        ]
    ).sum_horizontal()
    if failed_reports.sum() > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
