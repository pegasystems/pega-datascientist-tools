"""Pega standard outcome domain knowledge.

Provides channel-aware defaults for outcome label resolution, based on
Pega CDH standard outcomes documentation. Designed for use by any pdstools
component that maps raw Pega outcome strings to business metrics.

The Impact Analyzer uses this for Impressions/Accepts per Channel/Direction.
The IH class uses it for per-channel Engagement labels and direction-aware
OpenRate applicability.
"""

from __future__ import annotations

# Maps channel name prefix (part before '/') to (default_impressions, default_accepts).
# Based on Pega CDH '25 "Standard outcomes" documentation, channel-specific table.
# Both old-style (Accept, Click) and new-style (Accepted, Clicked) variants are
# included to handle different CDH versions in the field.
_CHANNEL_OUTCOME_DEFAULTS: dict[str, tuple[list[str], list[str]]] = {
    "Call Center": (["Impression"], ["Accept", "Accepted"]),
    "Retail": (["Impression"], ["Accept", "Accepted"]),
    "Direct Mail": (["Pending"], ["Click", "Clicked"]),
}

# Fallback for all channels not listed above (Web, Mobile, Email, SMS,
# Push, ATM, IVR, Paid, and any custom channels).
_DEFAULT_IMPRESSIONS: list[str] = ["Impression"]
_DEFAULT_ACCEPTS: list[str] = ["Click", "Clicked"]


def resolve_outcome_labels(
    channels_with_outcomes: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    """Resolve explicit per-channel outcome label mappings from data.

    For each channel, looks up Pega standard outcome defaults by matching
    the channel prefix (the part before '/'), then filters to only outcome
    values actually present in the data. Channels not in the defaults table
    fall back to the global digital-channel defaults.

    Parameters
    ----------
    channels_with_outcomes : dict[str, list[str]]
        Mapping of channel (Channel/Direction format) to the distinct
        outcome strings present in the data for that channel.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Per-channel mapping of the form::

            {
                "Web/Inbound": {"Impressions": ["Impression"], "Accepts": ["Clicked"]},
                "Call Center/Inbound": {"Impressions": ["Impression"], "Accepts": ["Accepted"]},
            }

        Only outcome values present in the data are included.

    Examples
    --------
    >>> resolve_outcome_labels({"Web/Inbound": ["Impression", "Clicked", "Rejected"]})
    {'Web/Inbound': {'Impressions': ['Impression'], 'Accepts': ['Clicked']}}

    >>> resolve_outcome_labels({"Call Center/Inbound": ["Impression", "Accepted"]})
    {'Call Center/Inbound': {'Impressions': ['Impression'], 'Accepts': ['Accepted']}}
    """
    result: dict[str, dict[str, list[str]]] = {}
    for channel, available in channels_with_outcomes.items():
        prefix = channel.split("/")[0]
        default_imp, default_acc = _CHANNEL_OUTCOME_DEFAULTS.get(prefix, (_DEFAULT_IMPRESSIONS, _DEFAULT_ACCEPTS))
        result[channel] = {
            "Impressions": [o for o in default_imp if o in available],
            "Accepts": [o for o in default_acc if o in available],
        }
    return result


def get_channel_defaults(channel: str) -> dict[str, list[str]]:
    """Return the unfiltered Pega standard outcome defaults for a channel.

    Looks up the channel prefix (part before '/') in the defaults table.
    Returns all candidate outcome values regardless of what is in the data.

    Parameters
    ----------
    channel : str
        Channel in "Channel/Direction" format (e.g. "Web/Inbound").

    Returns
    -------
    dict[str, list[str]]
        ``{"Impressions": [...], "Accepts": [...]}`` with all standard
        candidate values for this channel type.
    """
    prefix = channel.split("/")[0]
    default_imp, default_acc = _CHANNEL_OUTCOME_DEFAULTS.get(prefix, (_DEFAULT_IMPRESSIONS, _DEFAULT_ACCEPTS))
    return {"Impressions": list(default_imp), "Accepts": list(default_acc)}


# OpenRate outcomes — only meaningful for outbound channels (Email, SMS, Push, etc.).
# Both old-style (Open) and new-style (Opened) variants included.
_OPENRATE_POSITIVE: list[str] = ["Opened", "Open"]
_OPENRATE_NEGATIVE: list[str] = ["Impression", "Pending"]


def get_openrate_labels(channel: str) -> dict[str, list[str]] | None:
    """Return OpenRate outcome labels if applicable for this channel.

    OpenRate (whether a pushed message was opened) only applies to outbound
    channels. Returns None for inbound channels or channels without a
    direction component.

    Parameters
    ----------
    channel : str
        Channel in "Channel/Direction" format (e.g. "Email/Outbound").

    Returns
    -------
    dict[str, list[str]] or None
        ``{"positive": [...], "negative": [...]}`` for outbound channels,
        None for inbound or unknown direction.
    """
    parts = channel.split("/", 1)
    if len(parts) < 2 or parts[1] != "Outbound":
        return None
    return {"positive": list(_OPENRATE_POSITIVE), "negative": list(_OPENRATE_NEGATIVE)}
