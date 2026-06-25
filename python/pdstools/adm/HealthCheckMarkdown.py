from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .Analysis import Analysis, Finding, HealthCheckPreAggregates, Severity


def _format_markdown_value(value: object) -> str:
    """Format a finding payload value for markdown output."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        if value == 0:
            return "0"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    if isinstance(value, list):
        preview = ", ".join(_format_markdown_value(item) for item in value[:5])
        if len(value) > 5:
            preview += f" (+{len(value) - 5} more)"
        return preview
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(df: pl.DataFrame, *, max_rows: int | None = None) -> str:
    """Render a small Polars DataFrame as GitHub-flavored Markdown."""
    if df.height == 0:
        return "*No data available.*"

    truncated = max_rows is not None and df.height > max_rows
    table_df = df.head(max_rows) if truncated else df
    header = "| " + " | ".join(table_df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table_df.columns) + " |"
    rows = ["| " + " | ".join(_format_markdown_value(value) for value in row) + " |" for row in table_df.iter_rows()]
    markdown = "\n".join([header, separator, *rows])
    if truncated:
        markdown += f"\n\n*And {df.height - max_rows} more rows.*"
    return markdown


class HealthCheckMarkdownRenderer:
    """Render health-check findings as GitHub-flavored Markdown."""

    def __init__(self, analysis: Analysis) -> None:
        self.analysis = analysis

    def render(
        self,
        *,
        title: str = "ADM Health Check",
        subtitle: str = "",
        disclaimer: str = "",
        active_filter: pl.Expr | None = None,
        active_threshold_days: int = 30,
        prediction=None,
        preaggregates: HealthCheckPreAggregates | None = None,
    ) -> str:
        """Render the health-check findings as a markdown document."""
        preaggregates = preaggregates or self.analysis.compute_health_check_preaggregates(
            active_filter=active_filter,
            active_threshold_days=active_threshold_days,
            prediction=prediction,
            include_markdown_sections=True,
        )
        findings = self.analysis.findings(
            active_filter=active_filter,
            active_threshold_days=active_threshold_days,
            prediction=prediction,
            preaggregates=preaggregates,
        )
        headline = next((finding for finding in findings if finding.category == "summary"), None)
        body_findings = [finding for finding in findings if finding.category != "summary"]

        lines = [f"# {title}", ""]
        if subtitle:
            lines.extend([f"## {subtitle}", ""])
        if disclaimer:
            lines.extend([f"> **Disclaimer:** {disclaimer}", ""])

        if headline is not None:
            lines.extend(["## Summary", "", f"**{headline.title}**", "", headline.detail, ""])

        lines.extend(
            [
                "## Key Metrics",
                "",
                _markdown_table(
                    self.analysis._key_metrics_table(
                        preaggregates.last_data,
                        findings,
                        active_filter,
                        preaggregates=preaggregates,
                    )
                ),
                "",
            ]
        )

        estate_sections = self.analysis._estate_snapshot_sections(active_filter, preaggregates=preaggregates)
        if estate_sections:
            lines.extend(["## Estate Snapshot", ""])
            for section_title, section_table in estate_sections:
                lines.extend([f"### {section_title}", "", _markdown_table(section_table, max_rows=5), ""])

        lines.extend(["## Findings Overview", ""])
        lines.extend(
            [
                "### Findings by Severity",
                "",
                _markdown_table(self.analysis._findings_by_severity_table(body_findings)),
                "",
            ]
        )
        lines.extend(
            [
                "### Findings by Category",
                "",
                _markdown_table(self.analysis._findings_by_category_table(body_findings), max_rows=10),
                "",
            ]
        )

        if not body_findings:
            lines.extend(["## Findings", "", "No findings were generated.", ""])
            return "\n".join(lines).strip() + "\n"

        severity_order: list[Severity] = ["critical", "warning", "info", "success"]
        section_titles = {
            "critical": "Critical Findings",
            "warning": "Warning Findings",
            "info": "Informational Findings",
            "success": "Positive Findings",
        }

        for severity in severity_order:
            section_findings = [finding for finding in body_findings if finding.severity == severity]
            if not section_findings:
                continue
            lines.extend([f"## {section_titles[severity]}", ""])
            for finding in section_findings:
                lines.extend(
                    [
                        f"### {finding.title}",
                        "",
                        f"- **Category:** `{finding.category}`",
                        f"- **Severity:** `{finding.severity}`",
                        "",
                        finding.detail,
                    ]
                )
                if finding.data:
                    lines.append("")
                    lines.append("Supporting data:")
                    for key, value in finding.data.items():
                        lines.append(f"- **{key}**: {_format_markdown_value(value)}")
                lines.append("")

        return "\n".join(lines).strip() + "\n"

    def key_metrics_table(
        self,
        last_data: pl.DataFrame,
        findings: list[Finding],
        active_filter: pl.Expr | None,
        *,
        preaggregates: HealthCheckPreAggregates | None = None,
    ) -> pl.DataFrame:
        """Build a compact key-metrics table for markdown output."""
        if preaggregates is None:
            preaggregates = self.analysis._build_health_check_preaggregates(active_filter=active_filter)

        avg_auc = next(
            (finding.data.get("avg_auc") for finding in findings if finding.category == "summary"),
            None,
        )

        metrics: list[tuple[str, object]] = [
            (
                "Snapshot range",
                (
                    f"{preaggregates.date_start:%Y-%m-%d} to {preaggregates.date_end:%Y-%m-%d}"
                    if preaggregates.date_start is not None and preaggregates.date_end is not None
                    else None
                ),
            ),
            ("Models (latest snapshot)", preaggregates.total_models),
            ("Active models", preaggregates.active_models),
            ("Channels", preaggregates.channel_count),
            ("Configurations", preaggregates.configuration_count),
            ("Actions", preaggregates.action_count),
            ("Treatments", preaggregates.treatment_count),
            ("Responses", preaggregates.response_count),
            ("Positives", preaggregates.positive_count),
            (
                "Average AUC (latest snapshot)",
                avg_auc * 100 if isinstance(avg_auc, float) else None,
            ),
        ]

        if isinstance(preaggregates.active_avg_auc, float):
            metrics.append(("Average AUC (active)", preaggregates.active_avg_auc * 100))

        if self.analysis.datamart.predictor_data is not None:
            metrics.append(("Predictors", preaggregates.predictor_count))

        return pl.DataFrame(
            {
                "Metric": [metric for metric, _ in metrics],
                "Value": [_format_markdown_value(value) for _, value in metrics],
            }
        )

    def estate_snapshot_sections(
        self,
        active_filter: pl.Expr | None,
        *,
        preaggregates: HealthCheckPreAggregates | None = None,
    ) -> list[tuple[str, pl.DataFrame]]:
        """Build compact orientation tables for the markdown report."""
        sections: list[tuple[str, pl.DataFrame]] = []
        if preaggregates is None:
            preaggregates = self.analysis._build_health_check_preaggregates(
                active_filter=active_filter,
                include_markdown_sections=True,
            )

        if preaggregates.channel_overview is not None:
            channel_summary = (
                preaggregates.channel_overview.filter(
                    ~(
                        pl.col("Channel").map_elements(self.analysis._is_invalid_channel_name, return_dtype=pl.Boolean)
                        | pl.col("Direction").map_elements(
                            self.analysis._is_invalid_channel_name,
                            return_dtype=pl.Boolean,
                        )
                    )
                )
                .select(
                    "ChannelDirection",
                    "Responses",
                    "Positives",
                    "Performance",
                    "CTR",
                    pl.col("Used Actions").alias("Actions"),
                )
                .sort("Responses", descending=True)
            )
            if channel_summary.height > 0:
                sections.append(("Top Channels by Responses", channel_summary))
        elif preaggregates.channel_summary is not None:
            channel_summary = preaggregates.channel_summary.select(
                "ChannelDirection", "Responses", "Positives", "Performance", "CTR", "Actions"
            ).sort("Responses", descending=True)
            if channel_summary.height > 0:
                sections.append(("Top Channels by Responses", channel_summary))

        if preaggregates.configuration_summary is not None:
            configuration_columns = [
                col
                for col in [
                    "Configuration",
                    "Channel",
                    "Direction",
                    "ResponseCount",
                    "Positives",
                    "Performance",
                ]
                if col in preaggregates.configuration_summary.columns
            ]
            configuration_summary = preaggregates.configuration_summary.select(configuration_columns).sort(
                "ResponseCount", descending=True
            )
            if configuration_summary.height > 0:
                sections.append(("Top Configurations by Responses", configuration_summary))

        if preaggregates.predictor_categories is not None and preaggregates.predictor_categories.height > 0:
            sections.append(("Predictor Categories", preaggregates.predictor_categories))

        return sections

    @staticmethod
    def findings_by_severity_table(findings: list[Finding]) -> pl.DataFrame:
        """Summarize findings by severity."""
        severity_order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        rows = sorted(counts.items(), key=lambda item: severity_order[item[0]])
        return pl.DataFrame({"Severity": [severity for severity, _ in rows], "Count": [count for _, count in rows]})

    @staticmethod
    def findings_by_category_table(findings: list[Finding]) -> pl.DataFrame:
        """Summarize findings by category."""
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return pl.DataFrame({"Category": [category for category, _ in rows], "Count": [count for _, count in rows]})
