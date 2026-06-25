"""Programmatic ADM health findings built on top of ``ADMDatamart``."""

from __future__ import annotations

__all__ = ["Analysis", "Finding", "HealthCheckPreAggregates", "_format_markdown_value"]

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import polars as pl

from .HealthCheckMarkdown import HealthCheckMarkdownRenderer, _format_markdown_value
from ..utils import cdh_utils, report_utils
from ..utils.metric_limits import MetricLimits

if TYPE_CHECKING:  # pragma: no cover
    from ..adm.ADMDatamart import ADMDatamart
    from ..prediction.Prediction import Prediction

logger = logging.getLogger(__name__)
RECOVERABLE_ANALYSIS_ERRORS = (
    pl.exceptions.PolarsError,
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)

Severity = Literal["critical", "warning", "info", "success"]
Category = Literal[
    "channel",
    "model",
    "predictor",
    "prediction",
    "taxonomy",
    "response_distribution",
    "trend",
    "configuration",
    "data_quality",
    "summary",
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
        if self.category == "summary":
            icon = "🩺"
        else:
            icon = {"critical": "❌", "warning": "⚠️", "info": "ℹ️", "success": "✅"}[self.severity]
        return f"{icon} [{self.category}] {self.title}"


@dataclass
class HealthCheckPreAggregates:
    """Precomputed summaries reused across health-check outputs."""

    last_data: pl.DataFrame
    date_start: object | None = None
    date_end: object | None = None
    total_models: int | None = None
    active_models: int | None = None
    channel_count: int | None = None
    configuration_count: int | None = None
    action_count: int | None = None
    treatment_count: int | None = None
    response_count: int | None = None
    positive_count: int | None = None
    overall_avg_auc: float | None = None
    active_avg_auc: float | None = None
    predictor_count: int | None = None
    channel_overview: pl.DataFrame | None = None
    channel_summary: pl.DataFrame | None = None
    configuration_summary: pl.DataFrame | None = None
    predictor_overview: pl.DataFrame | None = None
    predictor_categories: pl.DataFrame | None = None
    prediction_summary: pl.DataFrame | None = None
    taxonomy_counts: dict[str, int] = field(default_factory=dict)


def _gini(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values."""
    v = sorted(float(value) for value in values if not math.isnan(float(value)))
    n = len(v)
    total = sum(v)
    if n == 0 or total == 0:
        return 0.0
    weighted_sum = sum((index + 1) * value for index, value in enumerate(v))
    return float((2 * weighted_sum - (n + 1) * total) / (n * total))


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

    def _model_data(self) -> pl.LazyFrame:
        if self.datamart.model_data is None:
            raise ValueError("Model data is not available.")
        return self.datamart.model_data

    def _combined_data(self) -> pl.LazyFrame:
        if self.datamart.combined_data is None:
            raise ValueError("Combined predictor/model data is not available.")
        return self.datamart.combined_data

    def _markdown_renderer(self) -> HealthCheckMarkdownRenderer:
        return HealthCheckMarkdownRenderer(self)

    def health_check_active_filter(self, *, active_threshold_days: int = 30) -> pl.Expr:
        """Return the shared active-model filter used by health checks."""
        return (
            pl.col("LastUpdate") > (pl.col("LastUpdate").max() - datetime.timedelta(days=active_threshold_days))
        ).fill_null(True)

    def health_check_active_threshold_date_string(self, *, active_threshold_days: int = 30) -> str:
        """Return the shared cutoff-date string for the active-model filter."""
        return (
            self._model_data()
            .select(
                (pl.col("LastUpdate").max() - datetime.timedelta(days=active_threshold_days))
                .dt.strftime("%v")
                .alias("cutoff")
            )
            .collect()
            .item()
            .strip()
        )

    def health_check_maturity_criteria(self) -> list[tuple[str, str, pl.Expr]]:
        """Return the shared maturity bucket definitions for health checks."""
        min_responses = MetricLimits.minimum("TotalResponseCount")
        min_positives = MetricLimits.minimum("TotalPositiveCount")
        bp_min_positives = MetricLimits.best_practice_min("TotalPositiveCount")
        min_perf = MetricLimits.minimum("ModelPerformance")
        if bp_min_positives is None or min_perf is None:
            raise ValueError("Missing maturity thresholds in MetricLimits.")

        return [
            ("last_snapshot", "Number of models in last snapshot", pl.lit(True)),
            (
                "never_used",
                "Models that have never been used (responses = 0)",
                pl.col("ResponseCount") < min_responses,
            ),
            (
                "responses_no_positives",
                "Models that have been used (responses > 0) but never received a positive response",
                (pl.col("Positives") < min_positives) & (pl.col("ResponseCount") >= min_responses),
            ),
            (
                "immature",
                f"Models that are still in an immature phase of learning (positives < {int(bp_min_positives)})",
                (pl.col("Positives") < bp_min_positives) & (pl.col("Positives") >= min_positives),
            ),
            (
                "stuck_at_50",
                "Models that have received sufficient responses but are still at their minimum performance (AUC = 50)",
                (pl.col("Performance") == 0.5) & (pl.col("Positives") >= bp_min_positives),
            ),
            (
                "low_performance",
                f"Models that have received sufficient responses but still have a low performance (AUC < {min_perf})",
                (pl.col("Performance") > 0.5)
                & (pl.col("Performance") < min_perf)
                & (pl.col("Positives") >= bp_min_positives),
            ),
            (
                "mature_decent",
                f"Models with sufficient positive responses (≥ {int(bp_min_positives)}) and a decent performance (≥ {min_perf})",
                (pl.col("Performance") >= min_perf) & (pl.col("Positives") >= bp_min_positives),
            ),
        ]

    def health_check_maturity_overview(
        self,
        *,
        last_data: pl.DataFrame | None = None,
        active_filter: pl.Expr | None = None,
        active_threshold_days: int = 30,
    ) -> pl.DataFrame:
        """Return the shared maturity-overview table used by health checks."""
        if last_data is None:
            last_data = self._get_last_data()
        if active_filter is None:
            active_filter = self.health_check_active_filter(active_threshold_days=active_threshold_days)

        active_last_data = last_data.lazy().filter(active_filter)
        return pl.concat(
            [
                active_last_data.group_by(None)
                .agg(
                    pl.lit(label).alias("Category"),
                    pl.col("Name").filter(filter_expression).len().alias("Number of Models"),
                    (
                        cdh_utils.weighted_average_polars(
                            pl.col("Performance").filter(filter_expression),
                            pl.col("ResponseCount").filter(filter_expression),
                        )
                        * 100.0
                    ).alias("Average Performance"),
                )
                .drop("literal")
                .collect()
                for _, label, filter_expression in self.health_check_maturity_criteria()
            ]
        )

    def compute_health_check_preaggregates(
        self,
        *,
        active_filter: pl.Expr | None = None,
        active_threshold_days: int = 30,
        prediction: "Prediction | None" = None,
        include_markdown_sections: bool = True,
    ) -> HealthCheckPreAggregates:
        """Compute reusable summaries for health-check generation.

        Parameters
        ----------
        active_filter : pl.Expr, optional
            Custom Polars expression defining which models count as active.
            If not provided, a default filter based on ``active_threshold_days``
            is constructed.
        active_threshold_days : int, default 30
            Default recency window used when ``active_filter`` is not provided.
        prediction : Prediction, optional
            Optional prediction data to summarize alongside ADM data.
        include_markdown_sections : bool, default True
            Whether to also precompute the compact tables currently used by the
            Markdown health check (for example configuration summaries).

        Returns
        -------
        HealthCheckPreAggregates
            Materialized summaries that can be reused across multiple report
            outputs within one run.
        """
        if active_filter is None:
            active_filter = self.health_check_active_filter(active_threshold_days=active_threshold_days)
        return self._build_health_check_preaggregates(
            active_filter=active_filter,
            prediction=prediction,
            include_markdown_sections=include_markdown_sections,
        )

    def _build_health_check_preaggregates(
        self,
        *,
        active_filter: pl.Expr | None,
        prediction: "Prediction | None" = None,
        include_markdown_sections: bool = False,
    ) -> HealthCheckPreAggregates:
        """Precompute shared summaries reused across health-check outputs."""
        last_data = self._get_last_data()
        preaggregates = HealthCheckPreAggregates(last_data=last_data)

        try:
            metrics = last_data.select(
                pl.col("ModelID").n_unique().alias("total_models"),
                pl.col("Configuration").n_unique().alias("configuration_count"),
                pl.sum("ResponseCount").alias("response_count"),
                pl.sum("Positives").alias("positive_count"),
            ).row(0, named=True)
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute health-check pre-aggregates: %s", exc)
        else:
            preaggregates.total_models = metrics["total_models"]
            preaggregates.configuration_count = metrics["configuration_count"]
            preaggregates.response_count = metrics["response_count"]
            preaggregates.positive_count = metrics["positive_count"]

        try:
            all_columns = (
                self._combined_data().collect_schema().names()
                if self.datamart.predictor_data is not None
                else self._model_data().collect_schema().names()
            )
            taxonomy_fields: list[tuple[str, str | list[str]]] = [
                ("ActionCount", "Name"),
                ("IssueCount", "Issue"),
                ("ChannelsUsingADM", ["Channel", "Direction"]),
            ]
            if "Treatment" in all_columns:
                taxonomy_fields.append(("TreatmentCount", "Treatment"))
            preaggregates.taxonomy_counts = {
                metric_id: report_utils.n_unique_values(self.datamart, all_columns, field_name)
                for metric_id, field_name in taxonomy_fields
            }
            preaggregates.action_count = preaggregates.taxonomy_counts.get("ActionCount")
            preaggregates.treatment_count = preaggregates.taxonomy_counts.get("TreatmentCount")
            preaggregates.channel_count = preaggregates.taxonomy_counts.get("ChannelsUsingADM")
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute taxonomy-style pre-aggregate counts: %s", exc)

        try:
            preaggregates.overall_avg_auc = last_data.select(
                cdh_utils.weighted_average_polars("Performance", "ResponseCount")
            ).item()
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute overall AUC for health-check pre-aggregates: %s", exc)

        try:
            if active_filter is not None:
                active_last_data = last_data.lazy().filter(active_filter)
                preaggregates.active_models = active_last_data.select(pl.col("ModelID").n_unique()).collect().item()
                preaggregates.active_avg_auc = (
                    active_last_data.select(cdh_utils.weighted_average_polars("Performance", "ResponseCount"))
                    .collect()
                    .item()
                )
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute active model count for health-check pre-aggregates: %s", exc)

        try:
            date_range = (
                self._model_data()
                .select(
                    pl.col("SnapshotTime").min().alias("start"),
                    pl.col("SnapshotTime").max().alias("end"),
                )
                .collect()
                .row(0, named=True)
            )
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute snapshot date range for health-check pre-aggregates: %s", exc)
        else:
            preaggregates.date_start = date_range["start"]
            preaggregates.date_end = date_range["end"]

        try:
            preaggregates.channel_summary = self._health_check_channel_summary(
                active_filter=active_filter,
            )
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute health-check channel pre-aggregates: %s", exc)

        if include_markdown_sections:
            try:
                preaggregates.channel_overview = self.datamart.aggregates.channel_overview(query=active_filter)
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not compute shared health-check channel overview: %s", exc)
            try:
                preaggregates.configuration_summary = self.datamart.aggregates.configuration_overview(
                    query=active_filter
                )
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not compute health-check configuration pre-aggregates: %s", exc)

        if self.datamart.predictor_data is not None:
            try:
                preaggregates.predictor_overview = self.datamart.aggregates.global_predictor_overview(
                    query=active_filter
                )
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not compute health-check predictor pre-aggregates: %s", exc)
            else:
                preaggregates.predictor_count = preaggregates.predictor_overview.height
                if include_markdown_sections:
                    try:
                        preaggregates.predictor_categories = (
                            preaggregates.predictor_overview.lazy()
                            .group_by("PredictorCategory")
                            .agg(
                                pl.col("PredictorName").n_unique().alias("Predictors"),
                                pl.col("Mean").mean().alias("AvgPerformance"),
                            )
                            .sort("Predictors", descending=True)
                            .collect()
                        )
                    except RECOVERABLE_ANALYSIS_ERRORS as exc:
                        logger.debug("Could not compute health-check predictor-category pre-aggregates: %s", exc)

        if prediction is not None:
            try:
                preaggregates.prediction_summary = prediction.summary_by_channel().collect()
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not compute health-check prediction pre-aggregates: %s", exc)

        return preaggregates

    # ── public API ──────────────────────────────────────────────────────

    def findings(
        self,
        *,
        active_filter: pl.Expr | None = None,
        active_threshold_days: int = 30,
        prediction: "Prediction | None" = None,
        preaggregates: HealthCheckPreAggregates | None = None,
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

        Returns
        -------
        list[Finding]
            Sorted by severity (critical first, then warning, then info).
        """
        if active_filter is None:
            active_filter = self.health_check_active_filter(active_threshold_days=active_threshold_days)

        preaggregates = preaggregates or self.compute_health_check_preaggregates(
            active_filter=active_filter,
            active_threshold_days=active_threshold_days,
            prediction=prediction,
            include_markdown_sections=False,
        )
        last_data = preaggregates.last_data

        results: list[Finding] = []
        dq_findings, invalid_channels = self._check_data_quality()
        results.extend(dq_findings)
        channel_findings, dead_channel_count, has_mature_low_perf_channel = self._check_channels(
            active_filter,
            invalid_channels=invalid_channels,
            channel_summary=preaggregates.channel_summary,
        )
        results.extend(channel_findings)
        results.extend(self._check_configurations(last_data))
        results.extend(self._check_model_maturity(last_data))
        perf_findings, avg_auc = self._check_model_performance(last_data, active_filter)
        results.extend(perf_findings)
        results.extend(self._check_taxonomy(preaggregates=preaggregates))
        results.extend(self._check_response_distribution(last_data, active_filter))
        results.extend(self._check_trends())

        if self.datamart.predictor_data is not None:
            results.extend(self._check_predictors(predictor_overview=preaggregates.predictor_overview))

        if prediction is not None:
            results.extend(self._check_predictions(prediction, pred_summary=preaggregates.prediction_summary))

        severity_order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
        results.sort(key=lambda f: severity_order[f.severity])

        headline = self._build_headline(
            results,
            last_data,
            preaggregates.overall_avg_auc,
            dead_channel_count=dead_channel_count,
            has_mature_low_perf_channel=has_mature_low_perf_channel,
        )
        if headline is not None:
            results.insert(0, headline)
        return results

    def markdown(
        self,
        *,
        title: str = "ADM Health Check",
        subtitle: str = "",
        disclaimer: str = "",
        active_filter: pl.Expr | None = None,
        active_threshold_days: int = 30,
        prediction: "Prediction | None" = None,
        preaggregates: HealthCheckPreAggregates | None = None,
    ) -> str:
        """Render findings as agent-friendly GitHub-flavored Markdown.

        Parameters
        ----------
        title : str, default "ADM Health Check"
            Report title shown at the top of the markdown document.
        subtitle : str, default ""
            Optional subtitle shown below the title.
        disclaimer : str, default ""
            Optional disclaimer shown as a blockquote near the top.
        active_filter : pl.Expr, optional
            Custom Polars expression defining which models count as active.
        active_threshold_days : int, default 30
            Default recency window used when ``active_filter`` is not provided.
        prediction : Prediction, optional
            Optional prediction data for prediction-level findings.

        Returns
        -------
        str
            Markdown document summarizing the findings.
        """
        return self._markdown_renderer().render(
            title=title,
            subtitle=subtitle,
            disclaimer=disclaimer,
            active_filter=active_filter,
            active_threshold_days=active_threshold_days,
            prediction=prediction,
            preaggregates=preaggregates,
        )

    def _key_metrics_table(
        self,
        last_data: pl.DataFrame,
        findings: list[Finding],
        active_filter: pl.Expr | None,
        *,
        preaggregates: HealthCheckPreAggregates | None = None,
    ) -> pl.DataFrame:
        """Build a compact key-metrics table for markdown output."""
        return self._markdown_renderer().key_metrics_table(
            last_data,
            findings,
            active_filter,
            preaggregates=preaggregates,
        )

    def _estate_snapshot_sections(
        self,
        active_filter: pl.Expr | None,
        *,
        preaggregates: HealthCheckPreAggregates | None = None,
    ) -> list[tuple[str, pl.DataFrame]]:
        """Build compact orientation tables for the markdown report."""
        return self._markdown_renderer().estate_snapshot_sections(
            active_filter,
            preaggregates=preaggregates,
        )

    @staticmethod
    def _findings_by_severity_table(findings: list[Finding]) -> pl.DataFrame:
        """Summarize findings by severity."""
        return HealthCheckMarkdownRenderer.findings_by_severity_table(findings)

    @staticmethod
    def _findings_by_category_table(findings: list[Finding]) -> pl.DataFrame:
        """Summarize findings by category."""
        return HealthCheckMarkdownRenderer.findings_by_category_table(findings)

    # ── private check methods ───────────────────────────────────────────

    def _get_last_data(self) -> pl.DataFrame:
        last_data = self.datamart.aggregates.last()
        keep_columns = [
            "ModelID",
            "Configuration",
            "Channel",
            "Direction",
            "Issue",
            "Group",
            "Name",
            "Treatment",
            "ResponseCount",
            "Positives",
            "Performance",
            "SuccessRate",
            "LastUpdate",
        ]
        existing_columns = last_data.collect_schema().names()

        return (
            last_data.select([col for col in keep_columns if col in existing_columns])
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

    def _health_check_channel_summary(
        self,
        *,
        active_filter: pl.Expr | None,
    ) -> pl.DataFrame:
        model_data = self.datamart._require_model_data()
        if active_filter is not None:
            model_data = model_data.filter(active_filter)

        per_model = model_data.group_by(["Channel", "Direction", "ModelID"]).agg(
            (pl.col("Positives").max() - pl.col("Positives").min()).alias("Positives"),
            (pl.col("ResponseCount").max() - pl.col("ResponseCount").min()).alias("Responses"),
            pl.col("Positives").max().alias("TotalPositives"),
            pl.col("ResponseCount").max().alias("TotalResponseCount"),
            pl.col("Performance").mean().alias("Performance"),
        )

        channel_metrics = per_model.group_by(["Channel", "Direction"]).agg(
            pl.sum("Positives", "Responses", "TotalPositives", "TotalResponseCount"),
            cdh_utils.weighted_performance_polars("Performance", "Responses").alias("Performance"),
        )
        channel_actions = model_data.group_by(["Channel", "Direction"]).agg(
            pl.col("Name").n_unique().alias("Actions"),
            pl.col("Name").unique().alias("AllActions"),
        )

        channel_summary = (
            channel_metrics.join(
                channel_actions,
                on=["Channel", "Direction"],
                nulls_equal=True,
                how="left",
            )
            .with_columns(
                CTR=pl.when(pl.col("Responses") > 0)
                .then(pl.col("Positives") / pl.col("Responses"))
                .otherwise(pl.lit(None)),
                isValid=(pl.col("TotalPositives") >= 200) & (pl.col("TotalResponseCount") >= 1000),
                ChannelDirection=pl.format("{}/{}", pl.col("Channel"), pl.col("Direction")),
            )
            .collect()
        )

        valid_channels = channel_summary.filter(pl.col("isValid"))
        if valid_channels.height > 0:
            omni = (
                valid_channels.select("Channel", "Direction", "AllActions")
                .with_columns(
                    pl.col("AllActions")
                    .map_batches(cdh_utils.overlap_lists_polars, return_dtype=pl.Float64)
                    .alias("OmniChannel")
                )
                .drop("AllActions")
            )
            channel_summary = channel_summary.join(
                omni,
                on=["Channel", "Direction"],
                nulls_equal=True,
                how="left",
            )
        else:
            channel_summary = channel_summary.with_columns(pl.lit(None).alias("OmniChannel"))

        return channel_summary.select(
            "ChannelDirection",
            "Channel",
            "Direction",
            "Responses",
            "Positives",
            "Performance",
            "CTR",
            "Actions",
            "OmniChannel",
        )

    def _build_headline(
        self,
        results: list[Finding],
        last_data: pl.DataFrame,
        avg_auc: float | None,
        *,
        dead_channel_count: int,
        has_mature_low_perf_channel: bool,
    ) -> Finding | None:
        """Compose a single one-line summary finding from underlying check output."""
        total = last_data.height
        if total == 0:
            return None

        # Pull the underlying numbers we need straight from the data so we
        # don't depend on whether a particular check fired.
        never_used = last_data.filter(pl.col("ResponseCount") == 0).height
        pct_never_used = never_used / total * 100

        auc100 = avg_auc * 100 if avg_auc is not None else None

        severity_order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
        highest_severity = min(
            (f.severity for f in results),
            key=lambda severity: severity_order[severity],
            default="success",
        )
        severity: Severity = cast(Severity, highest_severity)
        verdict = {
            "critical": "NEEDS ATTENTION",
            "warning": "MIXED",
            "info": "HEALTHY",
            "success": "HEALTHY",
        }[severity]

        critical_count = sum(1 for finding in results if finding.severity == "critical")
        warning_count = sum(1 for finding in results if finding.severity == "warning")
        facts: list[str] = []
        if critical_count > 0:
            noun = "critical finding" if critical_count == 1 else "critical findings"
            facts.append(f"{critical_count} {noun}")
        elif warning_count > 0:
            noun = "warning" if warning_count == 1 else "warnings"
            facts.append(f"{warning_count} {noun}")
        if pct_never_used > 10:
            facts.append(f"{pct_never_used:.0f}% of models never used")
        if dead_channel_count >= 1:
            noun = "dead channel" if dead_channel_count == 1 else "dead channels"
            facts.append(f"{dead_channel_count} {noun}")
        if has_mature_low_perf_channel and dead_channel_count == 0:
            facts.append("low-performing channel present")
        if auc100 is not None and (severity != "success" or not facts):
            facts.append(f"AUC {auc100:.1f}")
        if not facts:
            facts.append(f"{100 - pct_never_used:.0f}% of models in use")

        # Cap at 3 facts to keep the headline scannable.
        facts = facts[:3]
        body = ", ".join(facts) if facts else "no issues detected"
        return Finding(
            severity=severity,
            category="summary",
            title=f"Health: {verdict} — {body}",
            detail=(
                "Headline derived from the rest of the findings. Open the full list below for the per-area detail."
            ),
            data={
                "verdict": verdict,
                "pct_never_used": pct_never_used,
                "dead_channel_count": dead_channel_count,
                "has_mature_low_perf_channel": has_mature_low_perf_channel,
                "avg_auc": avg_auc,
                "facts": facts,
            },
        )

    @staticmethod
    def _is_invalid_channel_name(value: str | None) -> bool:
        """Return True if a Channel/Direction value looks like missing metadata."""
        if value is None:
            return True
        s = str(value).strip()
        if not s or s == "/":
            return True
        # Catch values like "/Inbound" or "Web/" that come from concatenating
        # an empty Channel with a Direction in upstream summaries.
        if s.startswith("/") or s.endswith("/"):
            return True
        return False

    def _check_data_quality(self) -> tuple[list[Finding], set[str]]:
        """Detect malformed Channel/Direction values.

        Returns
        -------
        list[Finding]
            One ``data_quality`` finding if invalid values exist, else empty.
        set[str]
            Set of ``"Channel/Direction"`` strings considered invalid; callers
            should exclude these from downstream channel-level findings to
            avoid duplicating noise.
        """
        findings: list[Finding] = []
        invalid: set[str] = set()
        try:
            rows = self._model_data().select(["Channel", "Direction"]).unique().collect()
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not collect channel/direction values: %s", exc)
            return findings, invalid

        invalid_pairs: list[tuple[str, str]] = []
        for row in rows.iter_rows(named=True):
            ch = row.get("Channel")
            dr = row.get("Direction")
            if self._is_invalid_channel_name(ch) or self._is_invalid_channel_name(dr):
                ch_str = "" if ch is None else str(ch)
                dr_str = "" if dr is None else str(dr)
                invalid.add(f"{ch_str}/{dr_str}")
                invalid_pairs.append((ch_str, dr_str))

        if not invalid_pairs:
            return findings, invalid

        try:
            invalid_df = pl.DataFrame(
                {
                    "Channel": [p[0] for p in invalid_pairs],
                    "Direction": [p[1] for p in invalid_pairs],
                }
            )
            n_models = (
                self._model_data()
                .select(
                    pl.col("ModelID"),
                    pl.col("Channel").cast(pl.Utf8),
                    pl.col("Direction").cast(pl.Utf8),
                )
                .join(invalid_df.lazy(), on=["Channel", "Direction"], how="inner", nulls_equal=True)
                .select(pl.col("ModelID").n_unique())
                .collect()
                .item()
            )
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not count models with invalid channel metadata: %s", exc)
            n_models = 0

        if n_models <= 0:
            return findings, invalid

        examples = ", ".join(f'"{(ch + "/" + dr) if ch or dr else "/"}"' for ch, dr in invalid_pairs[:3])
        findings.append(
            Finding(
                severity="critical",
                category="data_quality",
                title=(
                    f"{n_models:,} models have invalid Channel/Direction values "
                    f"(e.g. {examples}) — likely missing channel metadata"
                ),
                detail=(
                    "Some models have Channel or Direction set to an empty "
                    "string, '/', or another malformed value. This usually "
                    "means the channel was not populated when the model was "
                    "created. Fix the upstream metadata or filter these "
                    "models before reporting."
                ),
                data={
                    "model_count": n_models,
                    "invalid_count": len(invalid_pairs),
                    "examples": [f"{ch}/{dr}" for ch, dr in invalid_pairs[:10]],
                },
            )
        )
        return findings, invalid

    def _check_channels(
        self,
        active_filter: pl.Expr,
        *,
        invalid_channels: set[str] | None = None,
        channel_summary: pl.DataFrame | None = None,
    ) -> tuple[list[Finding], int, bool]:
        findings: list[Finding] = []
        invalid_channels = invalid_channels or set()
        dead_channel_count = 0
        has_mature_low_perf_channel = False
        if channel_summary is None:
            try:
                channel_summary = self.datamart.aggregates.summary_by_channel(query=active_filter).collect()
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not compute channel summary: %s", exc)
                return findings, dead_channel_count, has_mature_low_perf_channel

        # Collect dead channels into per-direction aggregate findings instead
        # of one ❌ per channel — the per-channel detail is rarely actionable
        # and dwarfs everything else in the output.
        dead_by_direction: dict[str, list[str]] = {}

        for row in channel_summary.iter_rows(named=True):
            channel = row.get("Channel")
            direction = row.get("Direction")
            ch = f"{channel}/{direction}"
            if (
                ch in invalid_channels
                or self._is_invalid_channel_name(channel)
                or self._is_invalid_channel_name(direction)
            ):
                continue
            responses = row.get("Responses", 0) or 0
            positives = row.get("Positives", 0) or 0
            perf = row.get("Performance")
            omni = row.get("OmniChannel")

            if responses == 0:
                direction = row.get("Direction") or "Unknown"
                dead_by_direction.setdefault(direction, []).append(ch)
                continue

            if positives == 0:
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
                perf_rate = perf / 100 if perf > 1 else perf
                rag = MetricLimits.evaluate_metric_rag("ModelPerformance", perf_rate)
                if rag == "RED":
                    has_mature_low_perf_channel = True
                    findings.append(
                        Finding(
                            severity="warning",
                            category="channel",
                            title=f'Channel "{ch}" has very low average performance (AUC {perf:.1f})',
                            detail=(
                                "The weighted average model performance for this "
                                "channel is below the minimum threshold. Consider "
                                "reviewing predictor availability and data quality."
                            ),
                            data={"channel": ch, "performance": perf_rate, "mature_low_perf": True},
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

        # Emit one collapsed dead-channel finding per direction.
        for direction in sorted(dead_by_direction):
            channels = sorted(dead_by_direction[direction])
            n = len(channels)
            dead_channel_count += n
            preview = channels[:5]
            preview_str = ", ".join(preview)
            if n > len(preview):
                preview_str += f" (+{n - len(preview)} more)"
            noun = "channel has" if n == 1 else "channels have"
            findings.append(
                Finding(
                    severity="warning" if n == 1 else "critical",
                    category="channel",
                    title=(f"{n} {direction} {noun} zero responses (configured but inactive): {preview_str}"),
                    detail=(
                        "Models exist for these channels but have never received "
                        "any responses. Either the channel is not active or the "
                        "decisioning integration is not triggering ADM models."
                    ),
                    data={
                        "direction": direction,
                        "dead_channel_count": n,
                        "channels": channels,
                    },
                )
            )

        return findings, dead_channel_count, has_mature_low_perf_channel

    def _check_configurations(self, last_data: pl.DataFrame) -> list[Finding]:
        findings: list[Finding] = []

        # Pre-compute the dominant Channel/Direction per Configuration so we
        # can give callers a quick hint of which channel a problematic
        # configuration belongs to. We pick the (Channel, Direction) pair
        # with the highest model count, breaking ties on response volume.
        try:
            dominant = (
                last_data.group_by(["Configuration", "Channel", "Direction"])
                .agg(
                    pl.len().alias("_n_models"),
                    pl.sum("ResponseCount").alias("_resp"),
                )
                .sort(["Configuration", "_n_models", "_resp"], descending=[False, True, True])
                .group_by("Configuration", maintain_order=True)
                .agg(pl.first("Channel"), pl.first("Direction"))
            )
            dominant_map = {
                row["Configuration"]: (row["Channel"], row["Direction"]) for row in dominant.iter_rows(named=True)
            }
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute dominant channel per configuration: %s", exc)
            dominant_map = {}

        for config in last_data["Configuration"].unique().sort():
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
                ch_dr = dominant_map.get(config_str)
                ch_hint = f" ({ch_dr[0]}/{ch_dr[1]})" if ch_dr else ""
                findings.append(
                    Finding(
                        severity="critical",
                        category="configuration",
                        title=f'Configuration "{config_str}"{ch_hint} has no positives',
                        detail=(
                            f"Configuration has {total_responses:,} responses but "
                            "zero positive outcomes. Check the outcome label "
                            "configuration and response loop."
                        ),
                        data={
                            "configuration": config_str,
                            "responses": total_responses,
                            "positives": 0,
                            "channel": ch_dr[0] if ch_dr else None,
                            "direction": ch_dr[1] if ch_dr else None,
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
        criteria = {key: expr for key, _, expr in self.health_check_maturity_criteria()}

        # Unused models
        bp_max_unused = MetricLimits.best_practice_max("ModelsWithoutResponsesPercentage")
        unused = last_data.filter(criteria["never_used"]).height
        if unused > 0:
            pct = unused / total * 100
            warn_threshold_pct = (bp_max_unused or 0.2) * 100
            findings.append(
                Finding(
                    severity="warning" if pct > warn_threshold_pct else "info",
                    category="model",
                    title=f"{unused:,} models ({pct:.0f}%) have never been used",
                    detail=(
                        "These models have zero responses — they exist but were "
                        "never selected in decisioning. This is common for actions "
                        "that were only tested or never activated."
                    ),
                    data={"count": unused, "total": total, "percentage": pct, "bucket": "never_used"},
                )
            )

        # Models with responses but no positives
        no_positives = last_data.filter(criteria["responses_no_positives"]).height
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
                    data={
                        "count": no_positives,
                        "total": total,
                        "percentage": pct,
                        "bucket": "responses_no_positives",
                    },
                )
            )

        # Immature models — Positives in [1, bp_min) (mutually exclusive
        # with the never-used and zero-positives buckets above).
        if bp_min_positives is not None:
            immature = last_data.filter(criteria["immature"]).height
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
                            "bucket": "immature",
                        },
                    )
                )

        # Stuck at AUC=50
        if bp_min_positives is not None:
            stuck = last_data.filter(criteria["stuck_at_50"]).height
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

        if min_perf is not None and bp_min_positives is not None:
            low_perf = last_data.filter(criteria["low_performance"]).height
            if low_perf > 0:
                findings.append(
                    Finding(
                        severity="warning",
                        category="model",
                        title=(f"{low_perf:,} mature models have low performance (AUC < {min_perf * 100:.0f})"),
                        detail=(
                            "These models have sufficient positive responses but "
                            "still fall below the minimum AUC threshold used in the "
                            "health check. "
                            "Consider reviewing predictor data or adding better "
                            "features."
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

        # Mature low-performance models — count-only finding follows in
        # _check_model_performance; not emitted here so the maturity
        # percentage buckets remain a deliberate "visible cohorts" view
        # rather than a forced 100 % decomposition.

        # Healthy models summary
        if min_perf is not None and bp_min_positives is not None:
            healthy = last_data.filter(criteria["mature_decent"]).height
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
                            "bucket": "mature_decent",
                        },
                    )
                )

        return findings

    def _check_model_performance(
        self,
        last_data: pl.DataFrame,
        active_filter: pl.Expr,
    ) -> tuple[list[Finding], float | None]:
        """Emit a tiered AUC finding and return the weighted average.

        Returns
        -------
        list[Finding]
            One finding describing the active-model AUC, if computable.
        float | None
            The weighted average AUC across active models on a 0-1 scale,
            or ``None`` when not computable (no active data, all-NaN, etc.).
        """
        findings: list[Finding] = []

        try:
            active_data = last_data.filter(active_filter)
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not apply active filter for performance analysis: %s", exc)
            return findings, None

        if active_data.height == 0:
            return findings, None

        avg_perf = active_data.select(cdh_utils.weighted_average_polars("Performance", "ResponseCount")).item()

        if avg_perf is None or math.isnan(avg_perf):
            return findings, None

        auc100 = avg_perf * 100
        if auc100 < 58:
            severity: Severity = "critical"
            label = "low"
            tail = "Investigate predictor availability and data quality."
        elif auc100 < 62:
            severity = "warning"
            label = "borderline"
            tail = "Performance is below the typical healthy range; review predictors and outcome labelling."
        elif auc100 < 70:
            severity = "info"
            label = "healthy"
            tail = "Performance is within the typical healthy range."
        else:
            severity = "success"
            label = "strong"
            tail = "Performance is well above the typical healthy range."

        findings.append(
            Finding(
                severity=severity,
                category="model",
                title=f"Average performance across active models is {label} (AUC {auc100:.1f})",
                detail=(
                    f"Weighted average AUC across active models is {auc100:.1f}. "
                    f"Tiers: <58 low, 58-62 borderline, 62-70 healthy, ≥70 strong. {tail}"
                ),
                data={"active_weighted_avg_performance": avg_perf, "tier": label},
            )
        )

        return findings, avg_perf

    def _check_taxonomy(
        self,
        *,
        preaggregates: HealthCheckPreAggregates | None = None,
    ) -> list[Finding]:
        findings: list[Finding] = []
        all_columns = (
            self._combined_data().collect_schema().names()
            if self.datamart.predictor_data is not None
            else self._model_data().collect_schema().names()
        )

        checks = [
            (
                "ActionCount",
                "Name",
                "Total actions",
                preaggregates.taxonomy_counts.get("ActionCount") if preaggregates is not None else None,
            ),
            (
                "TreatmentCount",
                "Treatment",
                "Total treatments",
                preaggregates.taxonomy_counts.get("TreatmentCount") if preaggregates is not None else None,
            ),
            (
                "IssueCount",
                "Issue",
                "Unique issues",
                preaggregates.taxonomy_counts.get("IssueCount") if preaggregates is not None else None,
            ),
            (
                "ChannelsUsingADM",
                ["Channel", "Direction"],
                "Channels using ADM",
                preaggregates.taxonomy_counts.get("ChannelsUsingADM") if preaggregates is not None else None,
            ),
        ]

        for metric_id, field_name, label, precomputed_value in checks:
            if precomputed_value is not None:
                value = precomputed_value
            else:
                try:
                    value = report_utils.n_unique_values(self.datamart, all_columns, field_name)
                except RECOVERABLE_ANALYSIS_ERRORS as exc:
                    logger.debug("Could not compute taxonomy metric %s: %s", metric_id, exc)
                    continue

            rag = MetricLimits.evaluate_metric_rag(metric_id, value)
            if rag not in ("RED", "AMBER"):
                continue

            bp_min = MetricLimits.best_practice_min(metric_id)
            bp_max = MetricLimits.best_practice_max(metric_id)
            hard_min = MetricLimits.minimum(metric_id)
            hard_max = MetricLimits.maximum(metric_id)

            severity: Severity = "warning" if rag == "RED" else "info"

            if rag == "RED" and hard_min is not None and value < hard_min:
                title = f"{label} ({value:,}) is below minimum ({hard_min:.0f})"
            elif rag == "RED" and hard_max is not None and value > hard_max:
                title = f"{label} ({value:,}) is above maximum ({hard_max:.0f})"
            elif rag == "AMBER" and bp_min is not None and value < bp_min:
                title = f"{label} ({value:,}) is below recommended (≥{bp_min:.0f})"
            elif rag == "AMBER" and bp_max is not None and value > bp_max:
                title = f"{label} ({value:,}) is above typical range (>{bp_max:.0f})"
            elif bp_min is not None and value < bp_min:
                title = f"{label} ({value:,}) is below recommended (≥{bp_min:.0f})"
            elif bp_max is not None and value > bp_max:
                title = f"{label} ({value:,}) is above typical range (>{bp_max:.0f})"
            else:
                # Both bounds undefined for this rag — fall back to a clear summary.
                title = f"{label} ({value:,}) is outside the recommended range"

            findings.append(
                Finding(
                    severity=severity,
                    category="taxonomy",
                    title=title,
                    detail=(
                        f"Recommended range: "
                        f"{('≥' + format(bp_min, '.0f')) if bp_min is not None else '-'}"
                        f" to "
                        f"{('≤' + format(bp_max, '.0f')) if bp_max is not None else '-'}. "
                        f"Hard limits: "
                        f"{('≥' + format(hard_min, '.0f')) if hard_min is not None else '-'}"
                        f" to "
                        f"{('≤' + format(hard_max, '.0f')) if hard_max is not None else '-'}."
                    ),
                    data={
                        "metric_id": metric_id,
                        "value": value,
                        "best_practice_min": bp_min,
                        "best_practice_max": bp_max,
                        "minimum": hard_min,
                        "maximum": hard_max,
                    },
                )
            )

        # Check predictor counts per configuration
        if self.datamart.predictor_data is not None:
            try:
                pred_counts = (
                    self._combined_data()
                    .filter(pl.col("EntryType") != "Classifier")
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
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not compute predictor counts per configuration: %s", exc)

        return findings

    def _check_response_distribution(
        self,
        last_data: pl.DataFrame,
        active_filter: pl.Expr,
    ) -> list[Finding]:
        findings: list[Finding] = []

        try:
            active_data = last_data.filter(active_filter)
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not apply active filter for response distribution: %s", exc)
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
        bp_min_perf = MetricLimits.best_practice_min("ModelPerformance")
        total_resp = active_data.select(pl.sum("ResponseCount")).item() or 0
        if total_resp > 0 and bp_min_perf is not None:
            low_auc_resp = (
                active_data.filter(pl.col("Performance") < bp_min_perf).select(pl.sum("ResponseCount")).item() or 0
            )
            low_auc_pct = low_auc_resp / total_resp * 100
            auc_threshold_str = f"AUC < {bp_min_perf * 100:.0f}"
            if low_auc_pct > 50:
                findings.append(
                    Finding(
                        severity="warning",
                        category="response_distribution",
                        title=(
                            f"{low_auc_pct:.0f}% of response volume is driven by "
                            f"low-performance models ({auc_threshold_str})"
                        ),
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
                        title=(
                            f"{low_auc_pct:.0f}% of response volume is from "
                            f"low-performance models ({auc_threshold_str})"
                        ),
                        detail=("A notable portion of responses comes from models that are not yet performing well."),
                        data={
                            "low_auc_response_pct": low_auc_pct,
                            "low_auc_responses": low_auc_resp,
                            "total_responses": total_resp,
                        },
                    )
                )

        return findings

    def _check_trends(self) -> list[Finding]:
        findings: list[Finding] = []

        try:
            snapshots = self._model_data().select(pl.col("SnapshotTime").unique()).collect()
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not collect snapshot times for trend analysis: %s", exc)
            return findings

        n_snapshots = snapshots.height
        if n_snapshots < 3:
            return findings

        try:
            by_expr = pl.concat_str(pl.col("Channel"), pl.col("Direction"), separator="/").alias("Channel/Direction")

            trend = (
                self._model_data()
                .with_columns(by_expr)
                .group_by(["SnapshotTime", "Channel/Direction"])
                .agg(
                    (cdh_utils.weighted_average_polars("Performance", "ResponseCount") * 100).round(1).alias("AvgPerf"),
                    pl.sum("ResponseCount").alias("TotalResp"),
                )
                .sort(["Channel/Direction", "SnapshotTime"])
                .collect()
            )
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not build per-channel trend data: %s", exc)
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

    def _check_predictors(self, *, predictor_overview: pl.DataFrame | None = None) -> list[Finding]:
        findings: list[Finding] = []
        min_perf = MetricLimits.minimum("ModelPerformance")
        if min_perf is None:
            return findings

        # Poor predictors
        if predictor_overview is None:
            try:
                predictor_overview = self.datamart.aggregates.predictors_global_overview().collect()
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not compute predictor performance overview: %s", exc)
                predictor_overview = None
        if predictor_overview is not None:
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

        # Missing predictor data
        try:
            all_columns = self._combined_data().collect_schema().names()
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
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute missing predictor data rates: %s", exc)

        # Check for IH predictor proportion
        try:
            ih_counts = (
                self._combined_data()
                .filter(pl.col("EntryType") != "Classifier")
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
        except RECOVERABLE_ANALYSIS_ERRORS as exc:
            logger.debug("Could not compute IH predictor proportions: %s", exc)

        return findings

    def _check_predictions(
        self,
        prediction: "Prediction",
        *,
        pred_summary: pl.DataFrame | None = None,
    ) -> list[Finding]:
        findings: list[Finding] = []

        if pred_summary is None:
            try:
                pred_summary = prediction.summary_by_channel().collect()
            except RECOVERABLE_ANALYSIS_ERRORS as exc:
                logger.debug("Could not summarize prediction data by channel: %s", exc)
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
                if ctrl_pct > 10:
                    findings.append(
                        Finding(
                            severity="warning",
                            category="prediction",
                            title=(f'Large control group ({ctrl_pct:.1f}%) for "{ch}"'),
                            detail=(
                                "The control group is larger than typical (1-5%). "
                                "A larger control group means more customers are "
                                "receiving random rather than optimized offers."
                            ),
                            data={"channel": ch, "control_pct": ctrl_pct},
                        )
                    )
                elif ctrl_pct < 0.5:
                    findings.append(
                        Finding(
                            severity="info",
                            category="prediction",
                            title=(f'Very small control group ({ctrl_pct:.1f}%) for "{ch}"'),
                            detail=(
                                "The control group may be too small for reliable "
                                "lift measurement. Consider increasing to 1-2%."
                            ),
                            data={"channel": ch, "control_pct": ctrl_pct},
                        )
                    )

        return findings
