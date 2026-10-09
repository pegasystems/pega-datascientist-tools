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
from typing import cast

import polars as pl
from pdstools import ADMDatamart, Prediction
from pdstools.utils import cdh_utils
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


def _active_classifier_model_ids(datamart: ADMDatamart) -> set[str]:
    """Return model IDs with classifier bins for any model technique."""
    if datamart.predictor_data is None:
        return set()

    return set(
        datamart.predictor_data.filter(pl.col("EntryType") == "Classifier")
        .select(pl.col("ModelID").unique())
        .collect()["ModelID"]
        .to_list()
    )


def _empty_ci_metrics(technique: str) -> dict[str, float | int | str | None]:
    """Return empty CI maturity metrics for one model technique."""
    return {
        f"Active_{technique}_Models": 0,
        f"Active_{technique}_Models_With_CI": 0,
        f"{technique}_Maturity_Pct_Above_Threshold": None,
        f"{technique}_CI_Width_Mean": None,
        f"{technique}_CI_Width_Median": None,
        f"{technique}_CI_Width_P90": None,
        f"{technique}_CI_Width_Mean_AboveThreshold": None,
        f"{technique}_CI_Width_Mean_AtOrBelowThreshold": None,
        f"{technique}_CI_Width_Ratio_AtOrBelow_over_Above": None,
        f"{technique}_Positives_vs_CI_Width_Spearman": None,
    }


def _normalise_model_technique(column: pl.Expr) -> pl.Expr:
    """Treat missing model technique values as NaiveBayes."""
    return pl.coalesce(column, pl.lit("NaiveBayes")).alias("ModelTechnique")


def _compute_ci_maturity_analysis(
    datamart: ADMDatamart,
    *,
    active_window_days: int,
    positives_maturity_threshold: int,
) -> tuple[dict[str, float | int | str | None], pl.DataFrame]:
    """Compute maturity-versus-CI analysis for all classifier-bearing models."""
    import numpy as np

    def _float_or_none(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))

    techniques = ("NB", "AGB")
    if datamart.model_data is None:
        return {
            key: value for technique in techniques for key, value in _empty_ci_metrics(technique).items()
        }, pl.DataFrame()

    model_ids = _active_classifier_model_ids(datamart)
    if not model_ids:
        return {
            key: value for technique in techniques for key, value in _empty_ci_metrics(technique).items()
        }, pl.DataFrame()

    model_columns = datamart.model_data.collect_schema().names()
    select_exprs = [pl.col("ModelID")]
    if "ModelTechnique" in model_columns:
        select_exprs.append(_normalise_model_technique(pl.col("ModelTechnique")))
    else:
        select_exprs.append(pl.lit("NaiveBayes").alias("ModelTechnique"))
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

    model_rows = datamart.model_data.filter(pl.col("ModelID").is_in(list(model_ids))).select(select_exprs).collect()

    if model_rows.height == 0:
        return {
            key: value for technique in techniques for key, value in _empty_ci_metrics(technique).items()
        }, pl.DataFrame()

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

    active_agg = active_candidates.group_by("ModelID", "ModelTechnique").agg(
        Positives=pl.col("Positives").sum(),
        ResponseCount=pl.col("ResponseCount").sum(),
    )

    if active_agg.height == 0:
        return {
            key: value for technique in techniques for key, value in _empty_ci_metrics(technique).items()
        }, pl.DataFrame()

    model_ids_needing_ci = active_agg["ModelID"].to_list()
    empty_ci_columns = {
        "AUC_ActiveRange": pl.Series([], dtype=pl.Float64),
        "AUC_ActiveRange_CI_Lower": pl.Series([], dtype=pl.Float64),
        "AUC_ActiveRange_CI_Upper": pl.Series([], dtype=pl.Float64),
        "AUC_ActiveRange_CI_Available": pl.Series([], dtype=pl.Boolean),
        "AUC_ActiveRange_CI_Reason": pl.Series([], dtype=pl.Utf8),
    }
    try:
        # Compute active ranges for every model in one batched call instead of
        # once per model: active_ranges() re-scans the (potentially huge)
        # predictor data on every invocation, so calling it per model in a
        # Python loop makes this step scale with model count times predictor
        # size rather than just predictor size, and can take hours on large
        # customer exports.
        active_ranges_df = (
            datamart.active_ranges(model_ids_needing_ci)
            .select(
                "ModelID",
                "AUC_ActiveRange",
                "AUC_ActiveRange_CI_Lower",
                "AUC_ActiveRange_CI_Upper",
                "AUC_ActiveRange_CI_Available",
                "AUC_ActiveRange_CI_Reason",
            )
            .collect()
        )
    except Exception:
        active_ranges_df = pl.DataFrame(
            {"ModelID": pl.Series([], dtype=active_agg["ModelID"].dtype), **empty_ci_columns}
        )

    model_level = (
        active_agg.join(active_ranges_df, on="ModelID", how="left")
        .with_columns(
            IsActiveLast30Days=pl.lit(True),
            AUC_ActiveRange_CI_Available=pl.col("AUC_ActiveRange_CI_Available").fill_null(False),
            AUC_ActiveRange_CI_Reason=pl.col("AUC_ActiveRange_CI_Reason").fill_null("analysis_error"),
        )
        .with_columns(
            CI_Width=(pl.col("AUC_ActiveRange_CI_Upper") - pl.col("AUC_ActiveRange_CI_Lower")),
            PositivesSegment=pl.when(pl.col("Positives") > positives_maturity_threshold)
            .then(pl.lit(f">{positives_maturity_threshold}"))
            .otherwise(pl.lit(f"<={positives_maturity_threshold}")),
            MaturitySegmentAboveThreshold=pl.col("Positives") > positives_maturity_threshold,
        )
    )

    metrics = {}
    for technique_key, technique_df in model_level.group_by("ModelTechnique", maintain_order=True):
        technique = technique_key[0]
        prefix = "AGB" if technique == "GradientBoost" else "NB"
        active_count = technique_df.height
        ci_non_null = technique_df.filter(pl.col("CI_Width").is_not_null())
        mean_above = _float_or_none(
            ci_non_null.filter(pl.col("Positives") > positives_maturity_threshold).get_column("CI_Width").mean()
        )
        mean_at_or_below = _float_or_none(
            ci_non_null.filter(pl.col("Positives") <= positives_maturity_threshold).get_column("CI_Width").mean()
        )
        mean_ratio = (
            mean_at_or_below / mean_above
            if mean_above is not None and mean_above > 0 and mean_at_or_below is not None
            else None
        )
        corr_df = ci_non_null.select("Positives", "CI_Width")
        spearman = None
        if corr_df.height >= 2:
            # A constant column (zero stddev) makes corrcoef divide by zero and
            # return NaN, which is already handled below via np.isfinite.
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(corr_df["Positives"].rank().to_numpy(), corr_df["CI_Width"].rank().to_numpy())[0, 1]
            if np.isfinite(corr):
                spearman = float(corr)
        metrics.update(
            {
                f"Active_{prefix}_Models": active_count,
                f"Active_{prefix}_Models_With_CI": ci_non_null.height,
                f"{prefix}_Maturity_Pct_Above_Threshold": (
                    100.0
                    * technique_df.filter(pl.col("Positives") > positives_maturity_threshold).height
                    / active_count
                    if active_count > 0
                    else 0.0
                ),
                f"{prefix}_CI_Width_Mean": _float_or_none(ci_non_null["CI_Width"].mean())
                if ci_non_null.height > 0
                else None,
                f"{prefix}_CI_Width_Median": _float_or_none(ci_non_null["CI_Width"].median())
                if ci_non_null.height > 0
                else None,
                f"{prefix}_CI_Width_P90": _float_or_none(ci_non_null["CI_Width"].quantile(0.9))
                if ci_non_null.height > 0
                else None,
                f"{prefix}_CI_Width_Mean_AboveThreshold": mean_above,
                f"{prefix}_CI_Width_Mean_AtOrBelowThreshold": mean_at_or_below,
                f"{prefix}_CI_Width_Ratio_AtOrBelow_over_Above": mean_ratio,
                f"{prefix}_Positives_vs_CI_Width_Spearman": spearman,
            }
        )
    for prefix in ("NB", "AGB"):
        metrics = {**_empty_ci_metrics(prefix), **metrics}

    # Keep the original aggregate keys as NB aliases for existing consumers.
    metrics.update(
        {
            "Active_NB_Models": metrics["Active_NB_Models"],
            "Active_NB_Models_With_CI": metrics["Active_NB_Models_With_CI"],
            "Maturity_Pct_Above_Threshold": metrics["NB_Maturity_Pct_Above_Threshold"],
            "CI_Width_Mean": metrics["NB_CI_Width_Mean"],
            "CI_Width_Median": metrics["NB_CI_Width_Median"],
            "CI_Width_P90": metrics["NB_CI_Width_P90"],
            "CI_Width_Mean_AboveThreshold": metrics["NB_CI_Width_Mean_AboveThreshold"],
            "CI_Width_Mean_AtOrBelowThreshold": metrics["NB_CI_Width_Mean_AtOrBelowThreshold"],
            "CI_Width_Ratio_AtOrBelow_over_Above": metrics["NB_CI_Width_Ratio_AtOrBelow_over_Above"],
            "Positives_vs_CI_Width_Spearman": metrics["NB_Positives_vs_CI_Width_Spearman"],
        }
    )

    return metrics, model_level


def compute_auc_rollup_comparison(datamart: ADMDatamart) -> pl.DataFrame:
    """Compare AUC roll-up weighting schemes per model configuration.

    Reports, for each ``Configuration``, the per-model AUC aggregated with
    seven weighting schemes: an unweighted (naive) average, the current
    ResponseCount-weighted average, a pair-count Positives*Negatives-weighted
    average (the ratio of total concordant positive-negative pairs to total
    eligible pairs under conventional empirical AUC), a Positives-only-weighted
    average (a potentially closer approximation to the conditional
    inverse-variance benchmark than Positives*Negatives when Negatives >>
    Positives, since AUC variance is then governed almost entirely by
    Positives; see
    docs/plans/adm/auc-rollup-weighting.md), the same weighted averages
    restricted to models with at least one positive (a cheap way to check
    whether zero-positive models are what drives the response-count-weighted
    average away from the others), and a DeLong inverse-variance-weighted
    average based on grouped-bin variance estimates. The latter is a
    conditional fixed-effect benchmark for independent estimates of one
    common AUC, not a universally correct target for heterogeneous or
    correlated model rows.

    Parameters
    ----------
    datamart : ADMDatamart
        The loaded datamart

    Returns
    -------
    pl.DataFrame
        One row per Configuration with the aggregated AUC values.
    """
    if datamart.model_data is None:
        return pl.DataFrame()

    per_model = (
        datamart.aggregates.last()
        .filter(pl.col("ResponseCount") > 0)
        .select(
            "ModelID",
            "Configuration",
            "Performance",
            "Positives",
            "ResponseCount",
        )
        .with_columns(Negatives=pl.col("ResponseCount") - pl.col("Positives"))
        .collect()
    )
    if per_model.height == 0:
        return pl.DataFrame()

    if datamart.predictor_data is not None:
        classifier_model_ids = _active_classifier_model_ids(datamart)
        variance_model_ids = list(set(per_model["ModelID"]) & classifier_model_ids)
    else:
        variance_model_ids = []

    if variance_model_ids:
        active_range = (
            datamart.active_ranges(variance_model_ids)
            .filter(pl.col("AUC_ActiveRange_CI_Available"))
            .select(
                "ModelID",
                "AUC_ActiveRange",
                "AUC_ActiveRange_CI_Variance",
                "AUC_ActiveRange_CI_Estimate",
                "AUC_ActiveRange_CI_Scope",
            )
            .collect()
        )
        per_model = per_model.join(active_range, on="ModelID", how="left")
    else:
        per_model = per_model.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("AUC_ActiveRange"),
            pl.lit(None, dtype=pl.Float64).alias("AUC_ActiveRange_CI_Variance"),
            pl.lit(None, dtype=pl.Float64).alias("AUC_ActiveRange_CI_Estimate"),
            pl.lit(None, dtype=pl.String).alias("AUC_ActiveRange_CI_Scope"),
        )

    rows = []
    for (configuration,), group in per_model.group_by("Configuration", maintain_order=True):
        naive_mean = cast(float, group.select(pl.col("Performance").cast(pl.Float64).mean()).item())
        weighted_response_count = float(group.select(cdh_utils.weighted_performance_polars()).item())
        weighted_pos_neg = float(
            group.select(
                cdh_utils.weighted_average_polars("Performance", pl.col("Positives") * pl.col("Negatives"))
            ).item()
        )
        weighted_positives = float(group.select(cdh_utils.weighted_average_polars("Performance", "Positives")).item())

        positives_group = group.filter(pl.col("Positives") > 0)
        if positives_group.height > 0:
            weighted_response_count_pos_only = float(
                positives_group.select(cdh_utils.weighted_performance_polars()).item()
            )
            weighted_pos_neg_pos_only = float(
                positives_group.select(
                    cdh_utils.weighted_average_polars("Performance", pl.col("Positives") * pl.col("Negatives"))
                ).item()
            )
        else:
            weighted_response_count_pos_only = None
            weighted_pos_neg_pos_only = None

        variance_group = group.filter(
            pl.col("AUC_ActiveRange_CI_Variance").is_not_null() & (pl.col("AUC_ActiveRange_CI_Variance") > 0)
        )
        if variance_group.height > 0:
            # AGB segments sharing one fitted ensemble already carry the same
            # pooled configuration-level CI (AUC_ActiveRange_CI_Scope ==
            # "configuration"); take it once instead of re-pooling it once per
            # segment row, which would double-count the same evidence and
            # understate the resulting uncertainty. NB rows (scope == "model")
            # are treated as independent per-model estimates by this
            # conditional benchmark and are pooled as before.
            configuration_scoped = variance_group.filter(pl.col("AUC_ActiveRange_CI_Scope") == "configuration")
            model_scoped = variance_group.filter(pl.col("AUC_ActiveRange_CI_Scope") != "configuration")
            estimate_auc = model_scoped["AUC_ActiveRange"].to_list()
            estimate_variance = model_scoped["AUC_ActiveRange_CI_Variance"].to_list()
            if configuration_scoped.height > 0:
                estimate_auc.append(configuration_scoped["AUC_ActiveRange_CI_Estimate"][0])
                estimate_variance.append(configuration_scoped["AUC_ActiveRange_CI_Variance"][0])
            inverse_variance_result = cdh_utils.weighted_auc_ci_from_estimates(
                auc=estimate_auc,
                variance=estimate_variance,
                weights=[1.0 / v for v in estimate_variance],
            )
            n_delong_estimates = model_scoped.height + int(configuration_scoped.height > 0)
        else:
            inverse_variance_result = {"auc": None, "ci_lower": None, "ci_upper": None}
            n_delong_estimates = 0

        rows.append(
            {
                "Configuration": configuration,
                "N_Models": group.height,
                "AUC_Naive_Mean": naive_mean,
                "AUC_Weighted_ResponseCount": weighted_response_count,
                "AUC_Weighted_PosNeg": weighted_pos_neg,
                "AUC_Weighted_Positives": weighted_positives,
                "N_Models_With_Positives": positives_group.height,
                "AUC_Weighted_ResponseCount_PositivesOnly": weighted_response_count_pos_only,
                "AUC_Weighted_PosNeg_PositivesOnly": weighted_pos_neg_pos_only,
                "N_DeLong_Estimates": n_delong_estimates,
                "AUC_Weighted_InverseVariance_DeLong": inverse_variance_result["auc"],
                "AUC_InverseVariance_DeLong_CI_Lower": inverse_variance_result["ci_lower"],
                "AUC_InverseVariance_DeLong_CI_Upper": inverse_variance_result["ci_upper"],
            }
        )

    return pl.DataFrame(rows)


def _print_auc_rollup_table(df: pl.DataFrame) -> None:
    """Print a compact per-configuration AUC roll-up comparison table."""
    if df.height == 0:
        return
    print("  AUC roll-up comparison (Pega 50-100 scale):")
    display_df = df.sort("Configuration").select(
        pl.col("Configuration"),
        pl.col("N_Models").alias("N"),
        pl.col("N_Models_With_Positives").alias("N_pos"),
        (pl.col("AUC_Naive_Mean") * 100).round(1).alias("Naive"),
        (pl.col("AUC_Weighted_ResponseCount") * 100).round(1).alias("RespCnt"),
        (pl.col("AUC_Weighted_PosNeg") * 100).round(1).alias("PosNeg"),
        (pl.col("AUC_Weighted_Positives") * 100).round(1).alias("Positives"),
        (pl.col("AUC_Weighted_ResponseCount_PositivesOnly") * 100).round(1).alias("RespCnt+"),
        (pl.col("AUC_Weighted_PosNeg_PositivesOnly") * 100).round(1).alias("PosNeg+"),
        (pl.col("AUC_Weighted_InverseVariance_DeLong") * 100).round(1).alias("DeLong"),
        pl.col("N_DeLong_Estimates").alias("N_DeLong"),
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=-1, fmt_str_lengths=40):
        print(display_df)


# Conditional comparison benchmark: inverse-variance pooling of eligible
# model- or configuration-scoped AUC estimates (see
# docs/plans/adm/auc-rollup-weighting.md). All other aggregation methods are
# compared against it below.
AUC_ROLLUP_REFERENCE_COLUMN = "AUC_Weighted_InverseVariance_DeLong"
AUC_ROLLUP_CANDIDATE_COLUMNS = [
    "AUC_Weighted_ResponseCount",
    "AUC_Weighted_PosNeg",
    "AUC_Weighted_Positives",
    "AUC_Weighted_ResponseCount_PositivesOnly",
    "AUC_Weighted_PosNeg_PositivesOnly",
    "AUC_Naive_Mean",
]


def _lin_ccc(x, y) -> float:
    """Lin's concordance correlation coefficient between x and y.

    Unlike Pearson r, this drops when two series are highly correlated but
    systematically offset or differently scaled, which is what "how good an
    approximation" actually requires here.
    """
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    return float((2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2))


def _compare_auc_rollup_to_reference(df: pl.DataFrame, candidate_column: str) -> dict[str, float | int | str]:
    """Compute agreement statistics against the inverse-variance benchmark."""
    import numpy as np

    pair = df.select(candidate_column, AUC_ROLLUP_REFERENCE_COLUMN).drop_nulls()
    empty_stats = {"Bias": None, "MAE": None, "RMSE": None, "Pearson_r": None, "Lin_CCC": None}
    if pair.height < 2:
        return {"Metric": candidate_column, "N": pair.height, **empty_stats}

    candidate = pair[candidate_column].to_numpy()
    reference = pair[AUC_ROLLUP_REFERENCE_COLUMN].to_numpy()
    diff = candidate - reference

    return {
        "Metric": candidate_column,
        "N": pair.height,
        "Bias": float(diff.mean()),
        "MAE": float(np.abs(diff).mean()),
        "RMSE": float(np.sqrt((diff**2).mean())),
        "Pearson_r": float(np.corrcoef(candidate, reference)[0, 1]),
        "Lin_CCC": _lin_ccc(candidate, reference),
    }


def print_auc_rollup_agreement_table(df: pl.DataFrame, *, min_delong_estimates: int) -> pl.DataFrame:
    """Print how closely each AUC roll-up method agrees with the benchmark.

    Only configurations whose benchmark combines at least
    ``min_delong_estimates`` estimates are included, since the comparison
    is unstable with very few estimates. See
    docs/plans/adm/auc-rollup-weighting.md for the statistical rationale.

    Parameters
    ----------
    df : pl.DataFrame
        Rows from ``compute_auc_rollup_comparison`` across all datasets.
    min_delong_estimates : int
        Minimum ``N_DeLong_Estimates`` for a configuration to be included in
        the comparison.

    Returns
    -------
    pl.DataFrame
        The eligible subset of rows used for the comparison (for reuse by
        the Bland-Altman plot).
    """
    eligible = df.filter(pl.col("N_DeLong_Estimates") >= min_delong_estimates)
    print(
        f"\n{eligible.height} of {df.height} configurations have an inverse-variance "
        f"estimate combining >= {min_delong_estimates} estimates."
    )
    if eligible.height == 0:
        return eligible

    results = [_compare_auc_rollup_to_reference(eligible, column) for column in AUC_ROLLUP_CANDIDATE_COLUMNS]
    results_df = pl.DataFrame(results).select(
        "Metric",
        "N",
        pl.col("Bias").cast(pl.Float64).round(4),
        pl.col("MAE").cast(pl.Float64).round(4),
        pl.col("RMSE").cast(pl.Float64).round(4),
        pl.col("Pearson_r").cast(pl.Float64).round(4),
        pl.col("Lin_CCC").cast(pl.Float64).round(4),
    )
    print(
        f"Agreement with {AUC_ROLLUP_REFERENCE_COLUMN} "
        "(conditional DeLong-style inverse-variance benchmark; Bias/MAE/RMSE on the 0-1 AUC scale):"
    )
    print(results_df)
    return eligible


def generate_auc_rollup_bland_altman_plot(
    eligible_df: pl.DataFrame,
    *,
    output_dir: Path,
) -> Path | None:
    """Render a Bland-Altman plot against the conditional inverse-variance benchmark.

    For each candidate method, plots (mean of candidate & reference) on the
    x-axis against (candidate - reference) on the y-axis, with the mean
    bias and +/-1.96 SD limits of agreement annotated. This makes any
    systematic offset or scale-dependent disagreement visible directly,
    which plain scatter/correlation plots do not.

    Parameters
    ----------
    eligible_df : pl.DataFrame
        Rows already filtered to configurations with an eligible benchmark
        estimate (see ``print_auc_rollup_agreement_table``).
    output_dir : Path
        Directory to write the plot to.

    Returns
    -------
    Path | None
        The written plot path, or None if there wasn't enough data.
    """
    if eligible_df.height < 2:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ℹ matplotlib not installed — skipping AUC roll-up Bland-Altman plot")
        return None

    candidates = [
        ("AUC_Weighted_ResponseCount", "#e11d48", "RespCnt (current)"),
        ("AUC_Weighted_PosNeg", "#2563eb", "PosNeg (pair-pooled)"),
        ("AUC_Weighted_Positives", "#16a34a", "Positives-only"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "auc_rollup_bland_altman.png"
    fig, axes = plt.subplots(1, len(candidates), figsize=(6 * len(candidates), 5), dpi=160, sharey=True)
    for ax, (column, color, label) in zip(axes, candidates, strict=True):
        pair = eligible_df.select(column, AUC_ROLLUP_REFERENCE_COLUMN).drop_nulls()
        if pair.height < 2:
            continue
        candidate = pair[column].to_numpy()
        reference = pair[AUC_ROLLUP_REFERENCE_COLUMN].to_numpy()
        mean_of_pair = (candidate + reference) / 2
        diff = candidate - reference
        bias = diff.mean()
        limit = 1.96 * diff.std()

        ax.scatter(mean_of_pair, diff, s=16, alpha=0.5, color=color)
        ax.axhline(bias, color="black", linewidth=1.5, label=f"Bias={bias:.3f}")
        ax.axhline(bias + limit, color="black", linestyle="--", linewidth=1, label=f"+-1.96 SD={limit:.3f}")
        ax.axhline(bias - limit, color="black", linestyle="--", linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel(f"Mean of ({label}, DeLong-style IVW)")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Difference (candidate - DeLong-style IVW)")
    fig.suptitle("AUC roll-up agreement with conditional DeLong-style IVW benchmark")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✓ AUC roll-up Bland-Altman plot: {output_path}")
    return output_path


def _generate_ci_maturity_plots(
    model_level_df: pl.DataFrame,
    *,
    output_dir: Path,
    positives_maturity_threshold: int,
) -> list[Path]:
    """Generate one CI maturity scatter plot from model-level rows."""
    output_path = _generate_ci_width_plot(
        model_level_df,
        output_dir=output_dir,
        output_filename="ci_maturity_vs_confidence_intervals.png",
        positives_maturity_threshold=positives_maturity_threshold,
        title="AUC confidence interval width versus positive volume",
    )
    return [output_path] if output_path is not None else []


def _generate_cross_dataset_ci_width_plot(
    model_level_df: pl.DataFrame,
    *,
    output_dir: Path,
) -> Path | None:
    """Generate a pooled log-log CI-width versus positives plot.

    Only models with positive outcomes and a positive, available CI width are
    included. The fitted relationship is reported as a power law on the plot.
    """
    return _generate_ci_width_plot(
        model_level_df,
        output_dir=output_dir,
        output_filename="ci_width_vs_positives_all_datasets.png",
        positives_maturity_threshold=200,
        title="AUC confidence interval width versus positive volume",
    )


def _generate_ci_width_plot(
    model_level_df: pl.DataFrame,
    *,
    output_dir: Path,
    output_filename: str,
    positives_maturity_threshold: int,
    title: str,
) -> Path | None:
    """Render the shared log-log CI-width versus positives visual."""
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

    # Constrained fit assuming the theoretical Var(AUC) ~ 1/Positives law
    # (slope = -0.5, see docs/plans/adm/auc-rollup-weighting.md), to compare
    # directly against a historical "CI ~= C / sqrt(Positives)" rule of thumb.
    constrained_constant = 10 ** (log_ci_width + 0.5 * log_positives).mean()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    technique_colors = {
        "NaiveBayes": ("#2563eb", "Naive Bayes"),
        "GradientBoost": ("#e11d48", "AGB"),
    }
    if "ModelTechnique" in plot_df.columns:
        for technique, technique_df in plot_df.group_by("ModelTechnique", maintain_order=True):
            color, label = technique_colors.get(technique[0], ("#64748b", technique[0]))
            ax.scatter(
                technique_df["Positives"].to_numpy(),
                technique_df["CI_Width"].to_numpy(),
                s=14,
                alpha=0.45,
                color=color,
                label=label,
            )
    else:
        ax.scatter(positives, ci_width, s=14, alpha=0.35, color="#2563eb", label="Models")

    fit_x = np.logspace(log_positives.min(), log_positives.max(), 200)
    fit_y = 10 ** (intercept + slope * np.log10(fit_x))
    ax.plot(
        fit_x,
        fit_y,
        color="black",
        linewidth=2,
        label=f"Power-law fit (R²={r_squared:.2f})",
    )
    ax.axvline(
        positives_maturity_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{positives_maturity_threshold} positives",
    )
    for above_threshold, color, label in [
        (True, "#1f77b4", "Mean CI width >200"),
        (False, "#ff7f0e", "Mean CI width <=200"),
    ]:
        mean_width = plot_df.filter((pl.col("Positives") > positives_maturity_threshold) == above_threshold)[
            "CI_Width"
        ].mean()
        if mean_width is not None:
            ax.axhline(mean_width, color=color, linestyle="--", linewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Positive outcomes per model")
    ax.set_ylabel("AUC CI width")
    ax.set_title(title)
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Cross-dataset CI plot: {output_path} (n={plot_df.height}, slope={slope:.3f}, R²={r_squared:.2f})")
    print(
        f"  ℹ 1/sqrt(Positives)-constrained fit: CI_Width ≈ {constrained_constant:.3f}/sqrt(Positives) "
        f"(0-1 scale) ≈ {constrained_constant * 100:.2f}/sqrt(Positives) (Pega points scale); "
        "compare to any historical CI ≈ C/sqrt(Positives) rule of thumb (free-fit slope above "
        "should be close to -0.5 for that comparison to be meaningful)."
    )
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
    auc_rollup_rows: list[dict] | None = None,
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
        Trailing-day window used to classify active models.
    positives_maturity_threshold : int, default=200
        Positives threshold used for maturity segmentation.
    ci_maturity_dataset_rows : list[dict] | None, optional
        Optional collector receiving one dataset-level maturity metrics row.
    ci_maturity_model_rows : list[dict] | None, optional
        Optional collector receiving per-model maturity analysis rows.
    auc_rollup_rows : list[dict] | None, optional
        Optional collector receiving per-configuration AUC roll-up
        weighting comparison rows (see docs/plans/adm/auc-rollup-weighting.md).

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

        print("  → Computing AUC roll-up weighting comparison...")
        auc_rollup_df = compute_auc_rollup_comparison(datamart)
        _print_auc_rollup_table(auc_rollup_df)
        if auc_rollup_rows is not None and auc_rollup_df.height > 0:
            auc_rollup_rows.extend(auc_rollup_df.with_columns(Dataset=pl.lit(name)).to_dicts())

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
        help="Trailing active window in days for active model definition (default: 30)",
    )
    parser.add_argument(
        "--positives-maturity-threshold",
        type=int,
        default=200,
        help="Positives threshold for maturity segmentation (default: 200)",
    )
    parser.add_argument(
        "--min-delong-estimates",
        "--min-delong-models",
        dest="min_delong_estimates",
        type=int,
        default=5,
        help=(
            "Minimum inverse-variance estimates combined for a configuration to be "
            "included in the AUC roll-up agreement analysis (default: 5)"
        ),
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
    auc_rollup_rows: list[dict] | None = []
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
            auc_rollup_rows=auc_rollup_rows,
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

    auc_rollup_file = summary_dir / "auc_rollup_comparison.csv"
    if auc_rollup_rows:
        auc_rollup_df = pl.DataFrame(auc_rollup_rows)
        auc_rollup_df.write_csv(auc_rollup_file)
        print(f"✓ AUC roll-up weighting comparison: {auc_rollup_file}")

        eligible_df = print_auc_rollup_agreement_table(
            auc_rollup_df,
            min_delong_estimates=args.min_delong_estimates,
        )
        generate_auc_rollup_bland_altman_plot(eligible_df, output_dir=summary_dir)

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
