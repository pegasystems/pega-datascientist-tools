__all__ = ["Analysis", "Finding"]

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from ..utils import cdh_utils, report_utils
from ..utils.metric_limits import MetricLimits

if TYPE_CHECKING:  # pragma: no cover
    from ..adm.ADMDatamart import ADMDatamart
    from ..prediction import Prediction

logger = logging.getLogger(__name__)

Severity = Literal["critical", "warning", "info"]
Category = Literal[
    "channel",
    "model",
    "predictor",
    "prediction",
    "taxonomy",
    "response_distribution",
    "trend",
    "configuration",
]


@dataclass(frozen=True)
class Finding:
    """A single diagnostic finding from ADM health analysis.

    Parameters
    ----------
    severity : Severity
        One of "critical", "warning", or "info".
    category : Category
        The area of the analysis this finding relates to.
    title : str
        A short, one-line summary of the finding.
    detail : str
        A longer explanation with context and recommended action.
    data : dict
        Structured data for programmatic consumption.
    """

    severity: Severity
    category: Category
    title: str
    detail: str
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        icon = {"critical": "❌", "warning": "⚠️", "info": "ℹ️"}[self.severity]
        return f"{icon} [{self.category}] {self.title}"


def _gini(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values."""
    v = np.sort(np.array(values, dtype=float))
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


class Analysis:
    """Automated diagnostic analysis for ADM model data.

    Accessed as ``datamart.analysis``. Provides programmatic health findings
    that would otherwise require a human to scan through charts and tables.

    Examples
    --------
    >>> from pdstools import datasets
    >>> dm = datasets.cdh_sample()
    >>> for f in dm.analysis.findings():
    ...     print(f)
    """

    def __init__(self, datamart: "ADMDatamart") -> None:
        self.datamart = datamart

    # ── public API ──────────────────────────────────────────────────────

    def findings(
        self,
        *,
        active_filter: pl.Expr | None = None,
        active_threshold_days: int = 30,
        prediction: "Prediction | None" = None,
        response_threshold: int = 100,
    ) -> list[Finding]:
        """Run all diagnostic checks and return a list of findings.

        Parameters
        ----------
        active_filter : pl.Expr, optional
            A Polars expression that filters to "active" models.
            If not provided, a default filter based on
            ``active_threshold_days`` is constructed.
        active_threshold_days : int, default 30
            If ``active_filter`` is not given, models not updated in this
            many days are considered inactive.
        prediction : Prediction, optional
            A :class:`~pdstools.prediction.Prediction` instance. If
            provided, prediction-level findings are included.
        response_threshold : int, default 100
            Minimum number of responses for a model to be included in
            success-rate analyses.

        Returns
        -------
        list[Finding]
            Sorted by severity (critical first, then warning, then info).
        """
        if active_filter is None:
            active_filter = (
                pl.col("LastUpdate") > (pl.col("LastUpdate").max() - datetime.timedelta(days=active_threshold_days))
            ).fill_null(True)

        last_data = self._get_last_data()

        results: list[Finding] = []
        results.extend(self._check_channels(active_filter))
        results.extend(self._check_configurations(last_data))
        results.extend(self._check_model_maturity(last_data))
        results.extend(self._check_model_performance(last_data, active_filter))
        results.extend(self._check_taxonomy())
        results.extend(self._check_response_distribution(last_data, active_filter))
        results.extend(self._check_trends(active_filter))

        if self.datamart.predictor_data is not None:
            results.extend(self._check_predictors())

        if prediction is not None:
            results.extend(self._check_predictions(prediction))

        severity_order = {"critical": 0, "warning": 1, "info": 2}
        results.sort(key=lambda f: severity_order[f.severity])
        return results

    # ── private check methods ───────────────────────────────────────────

    def _get_last_data(self) -> pl.DataFrame:
        return (
            self.datamart.aggregates.last()
            .with_columns(pl.col(pl.Categorical).cast(pl.Utf8))
            .with_columns(
                [
                    pl.col(pl.Utf8).fill_null("NA"),
                    pl.col(pl.Null).fill_null("NA"),
                    pl.col("SuccessRate").fill_nan(0).fill_null(0),
                    pl.col("Performance").fill_nan(0).fill_null(0),
                    pl.col("ResponseCount").fill_null(0),
                ]
            )
            .collect()
        )

    def _check_channels(self, active_filter: pl.Expr) -> list[Finding]:
        findings: list[Finding] = []
        try:
            channel_summary = self.datamart.aggregates.summary_by_channel(query=active_filter).collect()
        except Exception as e:
            logger.debug("Could not compute channel summary: %s", e)
            return findings

        for row in channel_summary.iter_rows(named=True):
            ch = f"{row['Channel']}/{row['Direction']}"
            responses = row.get("Responses", 0) or 0
            positives = row.get("Positives", 0) or 0
            perf = row.get("Performance")
            # ctr = row.get("CTR")
            omni = row.get("OmniChannel")

            if responses == 0:
                findings.append(
                    Finding(
                        severity="critical",
                        category="channel",
                        title=f'Channel "{ch}" has zero responses',
                        detail=(
                            "Models exist for this channel but have never received "
                            "any responses. Either the channel is not active or the "
                            "decisioning integration is not triggering ADM models."
                        ),
                        data={"channel": ch, "responses": 0},
                    )
                )
            elif positives == 0:
                findings.append(
                    Finding(
                        severity="critical",
                        category="channel",
                        title=f'Channel "{ch}" has zero positives',
                        detail=(
                            f"Channel has {responses:,} responses but no positive "
                            "outcomes. Check that the response loop is correctly "
                            "configured and outcome labels match the model setup."
                        ),
                        data={"channel": ch, "responses": responses, "positives": 0},
                    )
                )

            if perf is not None and positives > 0:
                rag = MetricLimits.evaluate_metric_rag("ModelPerformance", perf)
                if rag == "RED":
                    findings.append(
                        Finding(
                            severity="warning",
                            category="channel",
                            title=f'Channel "{ch}" has very low average performance (AUC {perf * 100:.1f})',
                            detail=(
                                "The weighted average model performance for this "
                                "channel is below the minimum threshold. Consider "
                                "reviewing predictor availability and data quality."
                            ),
                            data={"channel": ch, "performance": perf},
                        )
                    )

            if omni is not None and omni < 0.2 and positives > 0:
                findings.append(
                    Finding(
                        severity="info",
                        category="channel",
                        title=f'Channel "{ch}" has low cross-channel overlap ({omni:.0%})',
                        detail=(
                            "This channel shares few actions with other channels. "
                            "If an omni-channel strategy is intended, consider "
                            "aligning actions across channels."
                        ),
                        data={"channel": ch, "omni_channel_pct": omni},
                    )
                )

            configs_per_channel = row.get("Configuration", "")
            if configs_per_channel and configs_per_channel.count(",") >= 2:
                n_configs = configs_per_channel.count(",") + 1
                findings.append(
                    Finding(
                        severity="info",
                        category="channel",
                        title=f'Channel "{ch}" has {n_configs} model configurations',
                        detail=(
                            "Typical channels have 1-2 model configurations. More "
                            "may add complexity. Verify this is intentional (e.g., "
                            "different predictors for different issues)."
                        ),
                        data={
                            "channel": ch,
                            "n_configurations": n_configs,
                            "configurations": configs_per_channel,
                        },
                    )
                )

        return findings

    def _check_configurations(self, last_data: pl.DataFrame) -> list[Finding]:
        findings: list[Finding] = []

        for config in self.datamart.unique_configurations:
            config_str = str(config)
            config_lower = config_str.lower()
            if any(
                x in config_lower
                for x in [
                    "default_inbound",
                    "default_outbound",
                    "default_model",
                ]
            ):
                findings.append(
                    Finding(
                        severity="warning",
                        category="configuration",
                        title=f'Default configuration name: "{config_str}"',
                        detail=(
                            "Using default model configuration names is not "
                            "recommended. Rename to a channel-specific name for "
                            "clarity and to avoid unintended model sharing."
                        ),
                        data={"configuration": config_str},
                    )
                )

            config_data = last_data.filter(pl.col("Configuration") == config_str)
            total_responses = config_data.select(pl.sum("ResponseCount")).item() or 0
            total_positives = config_data.select(pl.sum("Positives")).item() or 0

            if total_responses > 0 and total_positives == 0:
                findings.append(
                    Finding(
                        severity="critical",
                        category="configuration",
                        title=f'Configuration "{config_str}" has no positives',
                        detail=(
                            f"Configuration has {total_responses:,} responses but "
                            "zero positive outcomes. Check the outcome label "
                            "configuration and response loop."
                        ),
                        data={
                            "configuration": config_str,
                            "responses": total_responses,
                            "positives": 0,
                        },
                    )
                )

        return findings

    def _check_model_maturity(self, last_data: pl.DataFrame) -> list[Finding]:
        findings: list[Finding] = []
        total = last_data.height
        if total == 0:
            return findings

        bp_min_positives = MetricLimits.best_practice_min("TotalPositiveCount")
        min_perf = MetricLimits.minimum("ModelPerformance")

        # Unused models
        unused = last_data.filter(pl.col("ResponseCount") == 0).height
        if unused > 0:
            pct = unused / total * 100
            findings.append(
                Finding(
                    severity="warning" if pct > 20 else "info",
                    category="model",
                    title=f"{unused:,} models ({pct:.0f}%) have never been used",
                    detail=(
                        "These models have zero responses — they exist but were "
                        "never selected in decisioning. This is common for actions "
                        "that were only tested or never activated."
                    ),
                    data={"count": unused, "total": total, "percentage": pct},
                )
            )

        # Models with responses but no positives
        no_positives = last_data.filter((pl.col("ResponseCount") > 0) & (pl.col("Positives") == 0)).height
        if no_positives > 0:
            pct = no_positives / total * 100
            findings.append(
                Finding(
                    severity="warning" if pct > 10 else "info",
                    category="model",
                    title=f"{no_positives:,} models ({pct:.0f}%) have responses but zero positives",
                    detail=(
                        "These models have been used in decisions but never received "
                        "a positive response. This could indicate unattractive "
                        "propositions or a broken response loop."
                    ),
                    data={"count": no_positives, "total": total, "percentage": pct},
                )
            )

        # Immature models
        if bp_min_positives is not None:
            min_pos_count = MetricLimits.minimum("TotalPositiveCount") or 1
            immature = last_data.filter(
                (pl.col("Positives") < bp_min_positives) & (pl.col("Positives") >= min_pos_count)
            ).height
            if immature > 0:
                pct = immature / total * 100
                findings.append(
                    Finding(
                        severity="info",
                        category="model",
                        title=f"{immature:,} models ({pct:.0f}%) are still immature (positives < {int(bp_min_positives)})",
                        detail=(
                            "These models are learning but have not yet received "
                            "enough positive responses to be considered mature. "
                            "This is normal for recently introduced actions."
                        ),
                        data={
                            "count": immature,
                            "total": total,
                            "percentage": pct,
                            "threshold": bp_min_positives,
                        },
                    )
                )

        # Stuck at AUC=50
        if bp_min_positives is not None:
            stuck = last_data.filter((pl.col("Performance") == 0.5) & (pl.col("Positives") >= bp_min_positives)).height
            if stuck > 0:
                findings.append(
                    Finding(
                        severity="warning",
                        category="model",
                        title=f"{stuck:,} models stuck at minimum performance (AUC=50) despite sufficient data",
                        detail=(
                            f"These models have ≥{int(bp_min_positives)} positives "
                            "but model performance has not moved from the initial "
                            "value. This may indicate data quality issues, lack of "
                            "predictor variation, or technical problems."
                        ),
                        data={
                            "count": stuck,
                            "total": total,
                            "threshold_positives": bp_min_positives,
                        },
                    )
                )

        # Low performance
        if min_perf is not None and bp_min_positives is not None:
            low_perf = last_data.filter(
                (pl.col("Performance") > 0.5)
                & (pl.col("Performance") < min_perf)
                & (pl.col("Positives") >= bp_min_positives)
            ).height
            if low_perf > 0:
                findings.append(
                    Finding(
                        severity="warning",
                        category="model",
                        title=(f"{low_perf:,} mature models have low performance (AUC < {min_perf * 100:.0f})"),
                        detail=(
                            "These models have sufficient positive responses but "
                            "still show poor predictive performance. Consider "
                            "reviewing predictor data or adding better features."
                        ),
                        data={
                            "count": low_perf,
                            "total": total,
                            "min_performance": min_perf,
                        },
                    )
                )

        # Suspiciously high performance
        if bp_min_positives is not None:
            bp_max_perf = MetricLimits.best_practice_max("ModelPerformance")
            if bp_max_perf is not None:
                suspicious = last_data.filter(
                    (pl.col("Performance") > bp_max_perf) & (pl.col("Positives") >= bp_min_positives)
                ).height
                if suspicious > 0:
                    findings.append(
                        Finding(
                            severity="warning",
                            category="model",
                            title=(
                                f"{suspicious:,} models have suspiciously high "
                                f"performance (AUC > {bp_max_perf * 100:.0f})"
                            ),
                            detail=(
                                "Very high AUC values may indicate outcome leakers "
                                "in the predictor set, or models with very few "
                                "responses where the metric is unreliable."
                            ),
                            data={
                                "count": suspicious,
                                "total": total,
                                "max_performance": bp_max_perf,
                            },
                        )
                    )

        # Healthy models summary
        if min_perf is not None and bp_min_positives is not None:
            healthy = last_data.filter(
                (pl.col("Performance") >= min_perf) & (pl.col("Positives") >= bp_min_positives)
            ).height
            if healthy > 0:
                pct = healthy / total * 100
                findings.append(
                    Finding(
                        severity="info",
                        category="model",
                        title=f"{healthy:,} models ({pct:.0f}%) are mature with decent performance",
                        detail=(f"These models have ≥{int(bp_min_positives)} positives and AUC ≥{min_perf * 100:.0f}."),
                        data={
                            "count": healthy,
                            "total": total,
                            "percentage": pct,
                        },
                    )
                )

        return findings

    def _check_model_performance(
        self,
        last_data: pl.DataFrame,
        active_filter: pl.Expr,
    ) -> list[Finding]:
        findings: list[Finding] = []

        try:
            active_data = last_data.filter(active_filter)
        except Exception:
            return findings

        if active_data.height == 0:
            return findings

        avg_perf = active_data.select(cdh_utils.weighted_average_polars("Performance", "ResponseCount")).item()

        if avg_perf is not None and not math.isnan(avg_perf):
            rag = MetricLimits.evaluate_metric_rag("ModelPerformance", avg_perf)
            if rag == "GREEN":
                findings.append(
                    Finding(
                        severity="info",
                        category="model",
                        title=f"Overall average model performance is healthy (AUC {avg_perf * 100:.1f})",
                        detail="The weighted average performance across active models is within acceptable range.",
                        data={"weighted_avg_performance": avg_perf},
                    )
                )

        return findings

    def _check_taxonomy(self) -> list[Finding]:
        findings: list[Finding] = []
        all_columns = (
            self.datamart.combined_data.collect_schema().names()
            if self.datamart.predictor_data is not None
            else self.datamart.model_data.collect_schema().names()
        )

        checks = [
            ("ActionCount", "Name", "Total actions"),
            ("TreatmentCount", "Treatment", "Total treatments"),
            ("IssueCount", "Issue", "Unique issues"),
            ("ChannelsUsingADM", ["Channel", "Direction"], "Channels using ADM"),
        ]

        for metric_id, field_name, label in checks:
            try:
                value = report_utils.n_unique_values(self.datamart, all_columns, field_name)
            except Exception:
                continue

            rag = MetricLimits.evaluate_metric_rag(metric_id, value)
            if rag == "RED":
                bp_min = MetricLimits.best_practice_min(metric_id)
                bp_max = MetricLimits.best_practice_max(metric_id)
                findings.append(
                    Finding(
                        severity="warning",
                        category="taxonomy",
                        title=f"{label} ({value:,}) is outside recommended limits",
                        detail=(
                            f"The count of {value:,} is outside the recommended "
                            f"range (min: {bp_min}, max: {bp_max}). "
                            "Check Pega cloud service health limits."
                        ),
                        data={
                            "metric_id": metric_id,
                            "value": value,
                            "best_practice_min": bp_min,
                            "best_practice_max": bp_max,
                        },
                    )
                )
            elif rag == "AMBER":
                bp_min = MetricLimits.best_practice_min(metric_id)
                bp_max = MetricLimits.best_practice_max(metric_id)
                findings.append(
                    Finding(
                        severity="info",
                        category="taxonomy",
                        title=f"{label} ({value:,}) is outside best practice range",
                        detail=(
                            f"The count of {value:,} is outside the recommended "
                            f"best practice range (min: {bp_min}, max: {bp_max})."
                        ),
                        data={
                            "metric_id": metric_id,
                            "value": value,
                            "best_practice_min": bp_min,
                            "best_practice_max": bp_max,
                        },
                    )
                )

        # Check predictor counts per configuration
        if self.datamart.predictor_data is not None:
            try:
                pred_counts = (
                    self.datamart.combined_data.filter(pl.col("EntryType") != "Classifier")
                    .group_by("Configuration")
                    .agg(pl.col("PredictorName").n_unique().alias("n_predictors"))
                    .collect()
                )
                for row in pred_counts.iter_rows(named=True):
                    rag = MetricLimits.evaluate_metric_rag("PredictorCount", row["n_predictors"])
                    if rag in ("RED", "AMBER"):
                        findings.append(
                            Finding(
                                severity="warning" if rag == "RED" else "info",
                                category="taxonomy",
                                title=(f'Configuration "{row["Configuration"]}" has {row["n_predictors"]} predictors'),
                                detail=(
                                    f"The recommended range is "
                                    f"{MetricLimits.best_practice_min('PredictorCount'):.0f}"
                                    f"-{MetricLimits.best_practice_max('PredictorCount'):.0f} "
                                    f"predictors per configuration."
                                ),
                                data={
                                    "configuration": row["Configuration"],
                                    "n_predictors": row["n_predictors"],
                                },
                            )
                        )
            except Exception:
                pass

        return findings

    def _check_response_distribution(
        self,
        last_data: pl.DataFrame,
        active_filter: pl.Expr,
    ) -> list[Finding]:
        findings: list[Finding] = []

        try:
            active_data = last_data.filter(active_filter)
        except Exception:
            return findings

        if active_data.height == 0:
            return findings

        resp = active_data["ResponseCount"].to_list()
        gini_resp = _gini(resp)

        if gini_resp > 0.9:
            findings.append(
                Finding(
                    severity="warning",
                    category="response_distribution",
                    title=f"Response distribution is highly skewed (Gini={gini_resp:.2f})",
                    detail=(
                        "A very small number of models receive the vast majority "
                        "of responses. This may indicate over-leveraging of certain "
                        "actions or that prioritization rules are too aggressive."
                    ),
                    data={"gini_responses": gini_resp},
                )
            )
        elif gini_resp > 0.7:
            findings.append(
                Finding(
                    severity="info",
                    category="response_distribution",
                    title=f"Response distribution is moderately skewed (Gini={gini_resp:.2f})",
                    detail=(
                        "Some actions receive substantially more responses than "
                        "others. This is common but worth monitoring — verify that "
                        "high-volume actions are intentionally prioritized."
                    ),
                    data={"gini_responses": gini_resp},
                )
            )

        # Check positives distribution too
        pos = active_data["Positives"].to_list()
        gini_pos = _gini(pos)
        if gini_pos > 0.9:
            findings.append(
                Finding(
                    severity="warning",
                    category="response_distribution",
                    title=f"Positive response distribution is highly skewed (Gini={gini_pos:.2f})",
                    detail=(
                        "Positive responses are concentrated in a few models. "
                        "Check whether the high-converting actions are truly the "
                        "best options or if this is driven by exposure bias."
                    ),
                    data={"gini_positives": gini_pos},
                )
            )

        # Performance vs volume: what % of volume is at low AUC
        total_resp = active_data.select(pl.sum("ResponseCount")).item() or 0
        if total_resp > 0:
            low_auc_resp = active_data.filter(pl.col("Performance") < 0.55).select(pl.sum("ResponseCount")).item() or 0
            low_auc_pct = low_auc_resp / total_resp * 100
            if low_auc_pct > 50:
                findings.append(
                    Finding(
                        severity="warning",
                        category="response_distribution",
                        title=(f"{low_auc_pct:.0f}% of response volume is driven by low-performance models (AUC < 55)"),
                        detail=(
                            "Most responses come from models with very low "
                            "predictive performance. Targeting is sub-optimal. "
                            "Consider improving predictor data or allowing models "
                            "more time to mature."
                        ),
                        data={
                            "low_auc_response_pct": low_auc_pct,
                            "low_auc_responses": low_auc_resp,
                            "total_responses": total_resp,
                        },
                    )
                )
            elif low_auc_pct > 30:
                findings.append(
                    Finding(
                        severity="info",
                        category="response_distribution",
                        title=(f"{low_auc_pct:.0f}% of response volume is from low-performance models (AUC < 55)"),
                        detail=("A notable portion of responses comes from models that are not yet performing well."),
                        data={
                            "low_auc_response_pct": low_auc_pct,
                            "low_auc_responses": low_auc_resp,
                            "total_responses": total_resp,
                        },
                    )
                )

        return findings

    def _check_trends(self, active_filter: pl.Expr) -> list[Finding]:
        findings: list[Finding] = []

        try:
            snapshots = self.datamart.model_data.select(pl.col("SnapshotTime").unique()).collect()
        except Exception:
            return findings

        n_snapshots = snapshots.height
        if n_snapshots < 3:
            return findings

        try:
            by_expr = pl.concat_str(pl.col("Channel"), pl.col("Direction"), separator="/").alias("Channel/Direction")

            trend = (
                self.datamart.model_data.with_columns(by_expr)
                .group_by(["SnapshotTime", "Channel/Direction"])
                .agg(
                    (cdh_utils.weighted_average_polars("Performance", "ResponseCount") * 100).round(1).alias("AvgPerf"),
                    pl.sum("ResponseCount").alias("TotalResp"),
                )
                .sort(["Channel/Direction", "SnapshotTime"])
                .collect()
            )
        except Exception:
            return findings

        for channel in trend["Channel/Direction"].unique().sort():
            ch_trend = trend.filter(pl.col("Channel/Direction") == channel)
            if ch_trend.height < 3:
                continue

            perfs = ch_trend["AvgPerf"].to_list()
            first_half = perfs[: len(perfs) // 2]
            second_half = perfs[len(perfs) // 2 :]

            avg_first = sum(first_half) / len(first_half) if first_half else 0
            avg_second = sum(second_half) / len(second_half) if second_half else 0
            change = avg_second - avg_first

            if change < -3:
                findings.append(
                    Finding(
                        severity="warning",
                        category="trend",
                        title=(f'Performance declining for channel "{channel}" (Δ{change:+.1f} AUC)'),
                        detail=(
                            f"Average model performance dropped from "
                            f"{avg_first:.1f} to {avg_second:.1f} over the data "
                            "period. This may indicate data quality degradation, "
                            "concept drift, or configuration changes."
                        ),
                        data={
                            "channel": channel,
                            "start_perf": avg_first,
                            "end_perf": avg_second,
                            "change": change,
                        },
                    )
                )

            # Check for response volume drops
            resps = ch_trend["TotalResp"].to_list()
            if len(resps) >= 3:
                resp_first = resps[:3]
                resp_last = resps[-3:]
                avg_resp_first = sum(resp_first) / len(resp_first)
                avg_resp_last = sum(resp_last) / len(resp_last)
                if avg_resp_first > 0:
                    resp_change = (avg_resp_last - avg_resp_first) / avg_resp_first
                    if resp_change < -0.5:
                        findings.append(
                            Finding(
                                severity="warning",
                                category="trend",
                                title=(
                                    f"Response volume dropped significantly for "
                                    f'channel "{channel}" ({resp_change:+.0%})'
                                ),
                                detail=(
                                    "Response counts have dropped by more than 50% "
                                    "between the start and end of the data period. "
                                    "This may indicate that models were reset, "
                                    "actions were deactivated, or traffic patterns changed."
                                ),
                                data={
                                    "channel": channel,
                                    "start_avg_responses": avg_resp_first,
                                    "end_avg_responses": avg_resp_last,
                                    "change_pct": resp_change,
                                },
                            )
                        )

        return findings

    def _check_predictors(self) -> list[Finding]:
        findings: list[Finding] = []
        min_perf = MetricLimits.minimum("ModelPerformance")

        # Poor predictors
        try:
            predictor_overview = self.datamart.aggregates.predictors_global_overview().collect()
            poor_count = predictor_overview.filter(pl.col("Mean") < (min_perf * 100)).height
            total_predictors = predictor_overview.height

            if poor_count > 0:
                pct = poor_count / total_predictors * 100 if total_predictors > 0 else 0
                findings.append(
                    Finding(
                        severity="warning" if pct > 30 else "info",
                        category="predictor",
                        title=(
                            f"{poor_count} of {total_predictors} predictors "
                            f"({pct:.0f}%) have poor performance "
                            f"(mean AUC < {min_perf * 100:.0f})"
                        ),
                        detail=(
                            "These predictors consistently perform below the "
                            "minimum threshold across all models. They may "
                            "indicate data sourcing issues. Be cautious with "
                            "removal — they may be valuable for specific actions."
                        ),
                        data={
                            "poor_predictor_count": poor_count,
                            "total_predictors": total_predictors,
                            "threshold": min_perf * 100,
                        },
                    )
                )
        except Exception:
            pass

        # Missing predictor data
        try:
            all_columns = self.datamart.combined_data.collect_schema().names()
            gb_cols = report_utils.polars_subset_to_existing_cols(all_columns, ["PredictorCategory", "PredictorName"])

            missing = (
                self.datamart.aggregates.last(table="predictor_data")
                .filter(pl.col("PredictorName") != "Classifier")
                .filter(pl.col("PredictorCategory") != "IH")
                .group_by(gb_cols)
                .agg(
                    pl.col("BinResponseCount").filter(pl.col("BinSymbol") == "MISSING").sum().alias("MissingCount"),
                    pl.sum("BinResponseCount").alias("TotalResponses"),
                )
                .with_columns((pl.col("MissingCount") / pl.col("TotalResponses")).alias("MissingPct"))
                .filter(~pl.col("MissingPct").is_nan())
                .filter(pl.col("MissingPct") > 0.5)
                .collect()
            )

            if missing.height > 0:
                names = missing["PredictorName"].to_list()
                findings.append(
                    Finding(
                        severity="warning",
                        category="predictor",
                        title=f"{missing.height} predictors have >50% missing values",
                        detail=(
                            "These predictors (excluding IH) have high rates of "
                            "missing data, which may indicate data pipeline issues. "
                            f"Examples: {', '.join(names[:5])}"
                        ),
                        data={
                            "count": missing.height,
                            "predictor_names": names[:20],
                        },
                    )
                )
        except Exception:
            pass

        # Check for IH predictor proportion
        try:
            ih_counts = (
                self.datamart.combined_data.filter(pl.col("EntryType") != "Classifier")
                .group_by("Configuration")
                .agg(
                    pl.col("PredictorName").filter(pl.col("PredictorCategory") == "IH").n_unique().alias("IH_count"),
                    pl.col("PredictorName").n_unique().alias("Total"),
                )
                .collect()
            )

            for row in ih_counts.iter_rows(named=True):
                if row["IH_count"] > 100:
                    findings.append(
                        Finding(
                            severity="info",
                            category="predictor",
                            title=(f'Configuration "{row["Configuration"]}" has {row["IH_count"]} IH predictors'),
                            detail=(
                                "More than 100 Interaction History (IH) predictors "
                                "may cause performance issues. Consider reducing "
                                "the number of IH predictors."
                            ),
                            data={
                                "configuration": row["Configuration"],
                                "ih_predictor_count": row["IH_count"],
                                "total_predictors": row["Total"],
                            },
                        )
                    )
                elif row["IH_count"] == 0 and row["Total"] > 0:
                    findings.append(
                        Finding(
                            severity="info",
                            category="predictor",
                            title=(f'Configuration "{row["Configuration"]}" has no IH predictors'),
                            detail=(
                                "No Interaction History predictors are present. "
                                "IH predictors are typically valuable for "
                                "personalization and are recommended."
                            ),
                            data={
                                "configuration": row["Configuration"],
                                "ih_predictor_count": 0,
                                "total_predictors": row["Total"],
                            },
                        )
                    )
        except Exception:
            pass

        return findings

    def _check_predictions(self, prediction: "Prediction") -> list[Finding]:
        findings: list[Finding] = []

        try:
            pred_summary = prediction.summary_by_channel().collect()
        except Exception:
            return findings

        if "Lift" not in pred_summary.columns:
            return findings

        for row in pred_summary.iter_rows(named=True):
            ch = f"{row.get('Channel', '?')}/{row.get('Direction', '?')}"
            lift = row.get("Lift")

            if lift is not None and not math.isnan(lift):
                if lift < 0:
                    findings.append(
                        Finding(
                            severity="critical",
                            category="prediction",
                            title=f'Negative engagement lift ({lift * 100:.0f}%) for "{ch}"',
                            detail=(
                                "Models are performing worse than random selection. "
                                "This indicates misconfiguration, data problems, or "
                                "that the control group setup needs review."
                            ),
                            data={"channel": ch, "lift": lift},
                        )
                    )
                elif lift < 0.1:
                    findings.append(
                        Finding(
                            severity="warning",
                            category="prediction",
                            title=f'Very low engagement lift ({lift * 100:.0f}%) for "{ch}"',
                            detail=(
                                "The model-driven group is barely outperforming "
                                "the random group. The models may need better "
                                "predictors or more training data."
                            ),
                            data={"channel": ch, "lift": lift},
                        )
                    )

            # Check for default prediction names
            pred_name = row.get("Prediction", "")
            if pred_name and "default" in str(pred_name).lower():
                findings.append(
                    Finding(
                        severity="warning",
                        category="prediction",
                        title=f'Default prediction name: "{pred_name}"',
                        detail=(
                            "Using default prediction names (e.g., "
                            "'PredictInboundDefaultPropensity') may indicate "
                            "that the prediction configuration was not customized."
                        ),
                        data={"prediction_name": pred_name, "channel": ch},
                    )
                )

            # Check control group size
            ctrl_pct = row.get("ControlPercentage")
            if ctrl_pct is not None and not math.isnan(ctrl_pct):
                if ctrl_pct > 0.10:
                    findings.append(
                        Finding(
                            severity="warning",
                            category="prediction",
                            title=(f'Large control group ({ctrl_pct:.1%}) for "{ch}"'),
                            detail=(
                                "The control group is larger than typical (1-5%). "
                                "A larger control group means more customers are "
                                "receiving random rather than optimized offers."
                            ),
                            data={"channel": ch, "control_pct": ctrl_pct},
                        )
                    )
                elif ctrl_pct < 0.005:
                    findings.append(
                        Finding(
                            severity="info",
                            category="prediction",
                            title=(f'Very small control group ({ctrl_pct:.2%}) for "{ch}"'),
                            detail=(
                                "The control group may be too small for reliable "
                                "lift measurement. Consider increasing to 1-2%."
                            ),
                            data={"channel": ch, "control_pct": ctrl_pct},
                        )
                    )

        return findings
