"""Immutable records for the synthetic sample specification, not sample rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["Inbound", "Outbound"]
Technique = Literal["NaiveBayes", "GradientBoost"]
Family = Literal["customer", "account", "clickstream", "ih", "param", "external_model"]


@dataclass(frozen=True)
class Channel:
    """Define one channel's volume, baseline and demonstration purpose."""

    name: str
    """Display name."""
    direction: Direction
    """Interaction direction."""
    baseline_probability: float
    """Mature positive-response probability."""
    daily_responses: int | None
    """Daily opportunity target, absent for a window-total target."""
    window_responses: int | None
    """Total opportunity target for the canonical window, if applicable."""
    rationale: str
    """Reason this channel is in the demo."""
    serves: tuple[str, ...]
    """Report sections or demonstrations exercised."""


@dataclass(frozen=True)
class ActionChannel:
    """Bind an action to a channel and exactly one primary configuration."""

    channel: str
    """Channel name."""
    direction: Direction
    """Direction for this membership."""
    configuration: str
    """Primary configuration, never an unsafe fallback."""
    challenger_configuration: str | None
    """Optional Mobile challenger."""
    baseline_probability: float
    """Mature action-level response probability."""
    volume_weight: float
    """Relative weight normalized across active actions."""
    treatment_ids: tuple[str, ...]
    """Explicit treatments for this action/channel."""


@dataclass(frozen=True)
class Action:
    """Define a fictional action's taxonomy and canonical lifetime."""

    id: str
    """Stable taxonomy identifier."""
    issue: str
    """Business issue."""
    group: str
    """Business group."""
    name: str
    """Clean action name before export-only naming stories."""
    start_day: int
    """Inclusive canonical launch day."""
    stop_day: int | None
    """Exclusive canonical stop day; None means no stop."""
    rate_multiplier: float
    """Multiplier on each channel's mature baseline."""
    model_category: str | None
    """Native model-category proposal for the specialized route."""
    channels: tuple[ActionChannel, ...]
    """Explicit channel memberships."""
    rationale: str
    """Reason for this catalog row."""
    serves: tuple[str, ...]
    """Demonstrations exercised by this action."""


@dataclass(frozen=True)
class Treatment:
    """Define one action/channel creative and its baseline probability."""

    id: str
    """Stable treatment identifier."""
    action_id: str
    """Owning action identifier."""
    name: str
    """Exported treatment name."""
    channel: str
    """Channel name."""
    direction: Direction
    """Interaction direction."""
    configuration: str
    """Primary configuration."""
    rate_multiplier: float
    """Multiplier on the action/channel response probability."""
    volume_share: float
    """Share within the action/channel; sibling shares total one."""
    baseline_probability: float
    """Literal mature response probability."""


@dataclass(frozen=True)
class Distribution:
    """Describe synthetic numeric bins or categorical values."""

    kind: Literal["binned_numeric", "categorical"]
    """Distribution vocabulary, not an instruction to sample independently."""
    weights: tuple[float, ...]
    """Bin or category probabilities."""
    edges: tuple[float, ...] = ()
    """Numeric interval boundaries."""
    values: tuple[str | bool, ...] = ()
    """Symbolic category values."""


@dataclass(frozen=True)
class Predictor:
    """Define a feature's synthetic distribution and model-input policy."""

    id: str
    """Python/export-facing predictor name."""
    family: Family
    """Predictor family with shared rationale and latent drivers."""
    data_type: Literal["numeric", "symbolic"]
    """Predictor data type."""
    baseline_eligible: bool
    """Whether general configurations may use this field."""
    exclusion_reason: str | None
    """Why the field is excluded from the general baseline."""
    loan_allowed: bool
    """Membership in the stricter neutral Web Loans allowlist."""
    distribution: Distribution | None
    """Synthetic marginal shape; correlations are modeled separately."""
    missing_probability: float | None
    """Probability of a missing value."""
    signal_role: str
    """Shared latent driver role."""
    correlation_family: str
    """Driver family shared by related windows or financial attributes."""
    latent_effect_size: float | None
    """Log-odds effect per standardized driver, not observed importance."""


@dataclass(frozen=True)
class PredictorGroup:
    """Explain a predictor family's contribution to the demo."""

    name: Family
    """Family identifier."""
    rationale: str
    """Why this family is configured."""
    serves: tuple[str, ...]
    """Demonstrations exercised."""


@dataclass(frozen=True)
class Configuration:
    """Define model technique, configured inputs and lifecycle."""

    name: str
    """Configuration name."""
    technique: Technique
    """Adaptive modeling technique."""
    context_keys: tuple[str, ...]
    """Context dimensions of a model."""
    predictor_ids: tuple[str, ...]
    """Complete configured feature set."""
    start_day: int
    """Inclusive canonical introduction day."""
    prehistory: bool
    """Whether day-zero actions may have pre-window feedback."""
    role: str
    """Standard, specialized or champion/challenger purpose."""
    rationale: str
    """Why this configuration exists."""
    serves: tuple[str, ...]
    """Report sections or demonstrations exercised."""


@dataclass(frozen=True)
class ModelContext:
    """Identify an action-level model before export-only perturbations."""

    configuration: str
    """Configuration name."""
    action_id: str
    """Clean action identifier."""
    channel: str
    """Channel of the opportunity stream."""
    direction: Direction
    """Direction of the opportunity stream."""
    treatment_id: str | None
    """Treatment identifier; absent for Omni contexts."""
    start_day: int
    """Later of the action and configuration launch."""
    stop_day: int | None
    """Action stop day; snapshots continue with frozen counts."""
    prehistory: bool
    """Whether this particular context has pre-window feedback."""


@dataclass(frozen=True)
class Story:
    """Define one timed effect on the canonical opportunity stream."""

    name: str
    """Stable story identifier."""
    channel: str
    """Affected channel."""
    technique: Technique
    """Affected model technique."""
    start_day: int
    """Inclusive canonical start day."""
    duration_days: int
    """Duration of the event before recovery."""
    recovery_days: int
    """Duration of recovery or delayed negative-feedback catch-up."""
    ctr_multiplier: float
    """Multiplier on underlying response probability during the event."""
    positive_only: bool
    """Whether negative outcomes are delayed while positives continue."""
    rationale: str
    """Why the story is included."""
    serves: tuple[str, ...]
    """Report sections or demonstrations exercised."""
