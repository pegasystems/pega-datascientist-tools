"""Validated, immutable specification of the synthetic retail-banking catalog."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from math import isclose, isfinite

from ._actions import ACTIONS
from ._calendar import DEFAULT_DAYS
from ._configurations import CONFIGURATIONS
from ._predictors import PREDICTORS
from ._stories import STORIES
from ._treatments import TREATMENTS
from .schema import (
    Action,
    Channel,
    Configuration,
    ModelContext,
    Predictor,
    PredictorGroup,
    Story,
    Treatment,
)

CHANNELS = (
    Channel(
        "Web",
        "Inbound",
        0.03,
        150_000,
        None,
        "Established AGB channel with a separate, strictly restricted Loans route.",
        ("health_check", "specialized_models", "success_rate"),
    ),
    Channel(
        "Mobile",
        "Inbound",
        0.02,
        60_000,
        None,
        "New NB models and a direct 50/50 AGB challenger from week 13.",
        ("learning", "champion_challenger", "model_report"),
    ),
    Channel(
        "Email",
        "Outbound",
        0.015,
        100_000,
        None,
        "Established NB channel with a temporary feed-quality dip and recovery.",
        ("predictor_binning", "prediction_lift", "performance_trend"),
    ),
    Channel(
        "Call Center",
        "Outbound",
        0.1,
        None,
        700,
        "Deliberately tiny volume demonstrates excluded channels and configurations.",
        ("excluded_channels", "excluded_configurations", "sparse_groups"),
    ),
)

PREDICTOR_GROUPS = (
    PredictorGroup(
        "customer",
        "General relationship attributes; excluded fields remain audit-only, never model inputs.",
        ("predictor_overview", "numeric_binning", "missing_values"),
    ),
    PredictorGroup(
        "account",
        "Synthetic primary-account scalar projections with shared financial drivers.",
        ("numeric_binning", "correlation_groups", "missing_values"),
    ),
    PredictorGroup(
        "clickstream",
        "Correlated engagement windows; only neutral own-Web activity is permitted for Loans.",
        ("engagement", "correlation_groups", "specialized_models"),
    ),
    PredictorGroup(
        "ih",
        "Channel-specific historical response summaries and recency drivers.",
        ("predictor_importance", "champion_challenger", "specialized_models"),
    ),
    PredictorGroup(
        "param",
        "Synthetic creative and placement attributes, shared across modeling techniques.",
        ("predictor_importance", "symbolic_binning", "specialized_models"),
    ),
)


def _unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} identifiers must be unique.")


def _probability(value: float, label: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be a finite probability between zero and one.")


@dataclass(frozen=True)
class SampleSpec:
    """Keep the approved catalog and its policy constraints together.

    All nested records contain tuples rather than mutable dictionaries or lists.
    Constructing a modified specification validates cross-references immediately.
    Generated samples and private source-document metadata are not stored here.
    """

    channels: tuple[Channel, ...] = CHANNELS
    """Channel baselines, volumes and demonstration rationales."""
    configurations: tuple[Configuration, ...] = CONFIGURATIONS
    """Techniques, predictor membership and lifecycle."""
    actions: tuple[Action, ...] = ACTIONS
    """Fictional taxonomy, channel memberships and lifetimes."""
    treatments: tuple[Treatment, ...] = TREATMENTS
    """Literal creative/placement baselines."""
    predictors: tuple[Predictor, ...] = PREDICTORS
    """Feature eligibility, marginal distributions and shared driver metadata."""
    predictor_groups: tuple[PredictorGroup, ...] = PREDICTOR_GROUPS
    """Rationales shared by each family of predictors."""
    stories: tuple[Story, ...] = STORIES
    """Timed underlying-rate and feedback-measurement events."""

    def __post_init__(self) -> None:
        _unique(tuple(c.name for c in self.channels), "Channel")
        _unique(tuple(c.name for c in self.configurations), "Configuration")
        _unique(tuple(a.id for a in self.actions), "Action")
        _unique(tuple(t.id for t in self.treatments), "Treatment")
        _unique(tuple(p.id for p in self.predictors), "Predictor")
        _unique(tuple(p.name for p in self.predictor_groups), "Predictor group")
        _unique(tuple(s.name for s in self.stories), "Story")
        for records in (self.channels, self.configurations, self.actions, self.predictor_groups, self.stories):
            for record in records:
                if not record.rationale.strip() or not record.serves or any(not tag.strip() for tag in record.serves):
                    raise ValueError(f"{record.name} must have a rationale and non-empty serves tags.")
        for channel in self.channels:
            _probability(channel.baseline_probability, channel.name)
            targets = (channel.daily_responses, channel.window_responses)
            if sum(value is not None for value in targets) != 1 or any(
                value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0)
                for value in targets
            ):
                raise ValueError(f"{channel.name} must have exactly one positive integer volume target.")
        self._validate_predictors()
        self._validate_actions()
        for story in self.stories:
            if story.channel not in {c.name for c in self.channels}:
                raise ValueError(f"{story.name} references an unknown channel.")
            if (
                story.start_day < 0
                or story.duration_days <= 0
                or story.recovery_days < 0
                or story.start_day + story.duration_days + story.recovery_days > DEFAULT_DAYS
            ):
                raise ValueError(f"{story.name} has an invalid canonical lifetime.")
            _probability(story.ctr_multiplier, f"{story.name} response multiplier")

    def _validate_predictors(self) -> None:
        predictors = {p.id: p for p in self.predictors}
        families = {p.name for p in self.predictor_groups}
        for predictor in self.predictors:
            if predictor.family not in families:
                raise ValueError(f"{predictor.id} has no predictor-group rationale.")
            if predictor.baseline_eligible:
                if predictor.distribution is None or predictor.missing_probability is None:
                    raise ValueError(f"{predictor.id} needs a distribution and missing probability.")
                _probability(predictor.missing_probability, f"{predictor.id} missingness")
            elif not predictor.exclusion_reason:
                raise ValueError(f"{predictor.id} needs an exclusion reason.")
            if predictor.loan_allowed and not predictor.baseline_eligible:
                raise ValueError(f"{predictor.id} cannot be allowed for Loans but excluded from the baseline.")
            if predictor.latent_effect_size is not None and not isfinite(predictor.latent_effect_size):
                raise ValueError(f"{predictor.id} must have a finite latent effect.")
            self._validate_distribution(predictor)
        for configuration in self.configurations:
            _unique(configuration.predictor_ids, f"{configuration.name} predictor")
            if not 0 <= configuration.start_day < DEFAULT_DAYS:
                raise ValueError(f"{configuration.name} has an invalid introduction day.")
            for predictor_id in configuration.predictor_ids:
                if predictor_id not in predictors or not predictors[predictor_id].baseline_eligible:
                    raise ValueError(f"{configuration.name} uses an unknown or excluded predictor: {predictor_id}.")
                if configuration.role == "single-group specialization" and not predictors[predictor_id].loan_allowed:
                    raise ValueError(f"{configuration.name} violates the Loans allowlist: {predictor_id}.")

    @staticmethod
    def _validate_distribution(predictor: Predictor) -> None:
        distribution = predictor.distribution
        if distribution is None:
            return
        weights = distribution.weights
        if not weights or any(not isfinite(w) or w < 0 for w in weights) or not isclose(sum(weights), 1):
            raise ValueError(f"{predictor.id} distribution weights must be finite, non-negative and sum to one.")
        if distribution.kind == "binned_numeric":
            edges = distribution.edges
            if (
                predictor.data_type != "numeric"
                or distribution.values
                or len(edges) != len(weights) + 1
                or any(not isfinite(edge) for edge in edges)
                or any(left >= right for left, right in pairwise(edges))
            ):
                raise ValueError(f"{predictor.id} needs increasing numeric edges matching its weights.")
        elif distribution.kind == "categorical":
            if (
                predictor.data_type != "symbolic"
                or distribution.edges
                or len(distribution.values) != len(weights)
                or len(set(distribution.values)) != len(distribution.values)
            ):
                raise ValueError(f"{predictor.id} needs unique categorical values matching its weights.")
        else:
            raise ValueError(f"{predictor.id} has an unknown distribution kind.")

    def _validate_actions(self) -> None:
        channels = {c.name: c for c in self.channels}
        configurations = {c.name: c for c in self.configurations}
        treatments = {t.id: t for t in self.treatments}
        claimed: list[str] = []
        for action in self.actions:
            stop = action.stop_day if action.stop_day is not None else DEFAULT_DAYS
            if not 0 <= action.start_day < stop <= DEFAULT_DAYS:
                raise ValueError(f"{action.id} has an invalid lifetime.")
            if not isfinite(action.rate_multiplier) or action.rate_multiplier <= 0:
                raise ValueError(f"{action.id} must have a positive finite rate multiplier.")
            _unique(tuple(c.channel for c in action.channels), f"{action.id} channel")
            if not action.channels:
                raise ValueError(f"{action.id} needs at least one channel.")
            for membership in action.channels:
                if membership.channel not in channels or membership.direction != channels[membership.channel].direction:
                    raise ValueError(f"{action.id} has an unknown channel or mismatched direction.")
                if membership.configuration not in configurations:
                    raise ValueError(f"{action.id} routes to an unknown configuration.")
                configuration = configurations[membership.configuration]
                is_loans = membership.channel == "Web" and action.issue == "Sales" and action.group == "Loans"
                if is_loans != (configuration.role == "single-group specialization"):
                    raise ValueError(f"{action.id} violates exclusive Web Loans routing.")
                if membership.challenger_configuration is not None and (
                    membership.channel != "Mobile" or membership.challenger_configuration not in configurations
                ):
                    raise ValueError(f"{action.id} has an invalid challenger route.")
                if not isfinite(membership.volume_weight) or membership.volume_weight <= 0:
                    raise ValueError(f"{action.id} must have a positive finite volume weight.")
                expected = channels[membership.channel].baseline_probability * action.rate_multiplier
                if not isclose(membership.baseline_probability, expected, abs_tol=1e-12):
                    raise ValueError(f"{action.id} baseline does not match the channel and action multiplier.")
                if not membership.treatment_ids or any(t not in treatments for t in membership.treatment_ids):
                    raise ValueError(f"{action.id} references an empty or unknown treatment set.")
                shares = []
                for treatment_id in membership.treatment_ids:
                    treatment = treatments[treatment_id]
                    if (
                        treatment.action_id != action.id
                        or treatment.channel != membership.channel
                        or treatment.direction != membership.direction
                        or treatment.configuration != membership.configuration
                    ):
                        raise ValueError(f"{treatment_id} does not match its owning action/channel.")
                    _probability(treatment.baseline_probability, treatment_id)
                    _probability(treatment.volume_share, f"{treatment_id} volume share")
                    if not isclose(
                        treatment.baseline_probability,
                        membership.baseline_probability * treatment.rate_multiplier,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(f"{treatment_id} baseline does not match its multipliers.")
                    shares.append(treatment.volume_share)
                    claimed.append(treatment_id)
                if not isclose(sum(shares), 1):
                    raise ValueError(f"{action.id}/{membership.channel} treatment volume shares must sum to one.")
        if set(claimed) != set(treatments) or len(claimed) != len(treatments):
            raise ValueError("Every treatment must be owned by exactly one action/channel.")

    @property
    def model_contexts(self) -> tuple[ModelContext, ...]:
        """Enumerate primary, challenger and derived Omni action-level models."""
        configurations = {c.name: c for c in self.configurations}
        contexts = []
        for action in self.actions:
            for membership in action.channels:
                names = [membership.configuration]
                if membership.challenger_configuration is not None:
                    names.append(membership.challenger_configuration)
                if "OmniAdaptiveModel" in configurations:
                    names.append("OmniAdaptiveModel")
                for name in names:
                    configuration = configurations[name]
                    treatment_ids = (None,) if name == "OmniAdaptiveModel" else membership.treatment_ids
                    for treatment_id in treatment_ids:
                        contexts.append(
                            ModelContext(
                                name,
                                action.id,
                                membership.channel,
                                membership.direction,
                                treatment_id,
                                max(action.start_day, configuration.start_day),
                                action.stop_day,
                                configuration.prehistory and action.start_day == 0,
                            ),
                        )
        return tuple(contexts)

    @property
    def manifest(self) -> dict[str, int | dict[str, int]]:
        """Return a fresh catalog manifest, excluding AGB summary model rows."""
        return {
            "issues": len({a.issue for a in self.actions}),
            "groups": len({(a.issue, a.group) for a in self.actions}),
            "actions": len(self.actions),
            "treatments": len(self.treatments),
            "predictors": len(self.predictors),
            "configurations": len(self.configurations),
            "stories": len(self.stories),
            "model_contexts": len(self.model_contexts),
            "contexts_by_configuration": dict(sorted(Counter(c.configuration for c in self.model_contexts).items())),
        }

    def describe(self) -> str:
        """Render the catalog's counts and rationale tables as Markdown."""
        lines = [
            "# Synthetic sample specification",
            "",
            "## Catalog counts",
            "",
            "| Entity | Count |",
            "| --- | ---: |",
        ]
        lines.extend(f"| {name} | {value} |" for name, value in self.manifest.items() if isinstance(value, int))
        for title, records in (
            ("Channels", self.channels),
            ("Configurations", self.configurations),
            ("Predictor groups", self.predictor_groups),
            ("Stories", self.stories),
        ):
            lines.extend(("", f"## {title}", "", "| Name | Rationale | Serves |", "| --- | --- | --- |"))
            lines.extend(f"| {r.name} | {r.rationale} | {', '.join(r.serves)} |" for r in records)
        return "\n".join(lines) + "\n"


DEFAULT_SPEC = SampleSpec()
