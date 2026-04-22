from __future__ import annotations

import json
import random
import uuid
from typing import Any

SAMPLE_EVENTS = [
    {"experimentKey": "nba_random_relevant_action", "arm": "test", "eventType": "container", "customerId": "C1"},
    {"experimentKey": "nba_random_relevant_action", "arm": "test", "eventType": "container", "customerId": "C2"},
    {"experimentKey": "nba_random_relevant_action", "arm": "test", "eventType": "capture", "outcome": "Accepted", "value": 75, "customerId": "C1"},
    {"experimentKey": "nba_random_relevant_action", "arm": "control", "eventType": "container", "customerId": "C3"},

    {"experimentKey": "nba_propensity_only", "arm": "test", "eventType": "container", "customerId": "P1"},
    {"experimentKey": "nba_propensity_only", "arm": "test", "eventType": "container", "customerId": "P2"},
    {"experimentKey": "nba_propensity_only", "arm": "test", "eventType": "capture", "outcome": "Clicked", "value": 75, "customerId": "P2"},
    {"experimentKey": "nba_propensity_only", "arm": "control", "eventType": "container", "customerId": "P3"},
    {"experimentKey": "nba_propensity_only", "arm": "control", "eventType": "capture", "outcome": "Accepted", "value": 75, "customerId": "P3"},

    {"experimentKey": "nba_no_levers", "arm": "test", "eventType": "container", "customerId": "N1"},
    {"experimentKey": "nba_no_levers", "arm": "test", "eventType": "capture", "outcome": "Accepted", "value": 75, "customerId": "N1"},
    {"experimentKey": "nba_no_levers", "arm": "control", "eventType": "container", "customerId": "N2"},

    {"experimentKey": "adaptive_vs_random_propensity", "arm": "test", "eventType": "container", "customerId": "A1"},
    {"experimentKey": "adaptive_vs_random_propensity", "arm": "test", "eventType": "capture", "outcome": "Accepted", "value": 75, "customerId": "A1"},
    {"experimentKey": "adaptive_vs_random_propensity", "arm": "control", "eventType": "container", "customerId": "A2"},

    {"experimentKey": "nba_eligibility_only", "arm": "test", "eventType": "container", "customerId": "E1"},
    {"experimentKey": "nba_eligibility_only", "arm": "test", "eventType": "container", "customerId": "E2"},
    {"experimentKey": "nba_eligibility_only", "arm": "test", "eventType": "capture", "outcome": "Accepted", "value": 75, "customerId": "E1"},
    {"experimentKey": "nba_eligibility_only", "arm": "control", "eventType": "container", "customerId": "E3"},
    {"experimentKey": "nba_eligibility_only", "arm": "control", "eventType": "capture", "outcome": "Accepted", "value": 75, "customerId": "E3"},
]

SAMPLE_PAYLOAD = json.dumps(SAMPLE_EVENTS, indent=2)


# ── Configurable live data generator ──────────────────────────────────────
# Each experiment has: (key, test_accept_rate, control_accept_rate, test_traffic_share)
# test_traffic_share controls ratio of test vs control impressions (e.g. 0.98 = 98% test)

GENERATOR_PROFILES: dict[str, dict[str, Any]] = {
    "Realistic NBA (asymmetric traffic)": {
        "description": "Mirrors real Pega IA: ~98% test, ~2% control, low accept rates, NBA wins on most tests",
        "action_value": 75.0,
        "experiments": [
            # (key,                              test_rate, ctrl_rate, test_share)
            ("nba_random_relevant_action",           0.0115,   0.010,   0.98),
            ("nba_propensity_only",                  0.0109,   0.020,   0.92),
            ("nba_no_levers",                        0.0115,   0.010,   0.98),
            ("adaptive_vs_random_propensity",        0.0985,   0.096,   0.98),
            ("nba_eligibility_only",                 0.0985,   0.096,   0.98),
        ],
    },
    "Clear NBA winner (high lift)": {
        "description": "NBA clearly beats all baselines — significance reached quickly",
        "action_value": 75.0,
        "experiments": [
            ("nba_random_relevant_action",           0.12,    0.03,    0.90),
            ("nba_propensity_only",                  0.12,    0.06,    0.90),
            ("nba_no_levers",                        0.12,    0.05,    0.90),
            ("adaptive_vs_random_propensity",        0.10,    0.04,    0.90),
            ("nba_eligibility_only",                 0.12,    0.08,    0.90),
        ],
    },
    "Neck-and-neck (tiny lift)": {
        "description": "Very small differences — takes many events to reach significance",
        "action_value": 75.0,
        "experiments": [
            ("nba_random_relevant_action",           0.051,   0.050,   0.90),
            ("nba_propensity_only",                  0.052,   0.050,   0.90),
            ("nba_no_levers",                        0.050,   0.050,   0.90),
            ("adaptive_vs_random_propensity",        0.049,   0.050,   0.90),
            ("nba_eligibility_only",                 0.053,   0.050,   0.90),
        ],
    },
    "NBA losing (negative lift)": {
        "description": "Control outperforms NBA — shows negative lift and red indicators",
        "action_value": 75.0,
        "experiments": [
            ("nba_random_relevant_action",           0.03,    0.05,    0.90),
            ("nba_propensity_only",                  0.04,    0.06,    0.90),
            ("nba_no_levers",                        0.03,    0.04,    0.90),
            ("adaptive_vs_random_propensity",        0.02,    0.05,    0.90),
            ("nba_eligibility_only",                 0.04,    0.05,    0.90),
        ],
    },
}

DEFAULT_PROFILE = "Realistic NBA (asymmetric traffic)"


def generate_batch(
    batch_size: int = 50,
    profile_name: str = DEFAULT_PROFILE,
    counter: int = 0,
) -> list[dict[str, Any]]:
    """Generate a batch of realistic IA events.

    Each call produces `batch_size` impression events spread across all 5
    experiments (with the configured test/control traffic split), and then
    probabilistically generates accept captures based on the configured
    accept rates.

    Parameters
    ----------
    batch_size : int
        Number of *impressions* to generate per experiment per call.
    profile_name : str
        Which generator profile to use (key into GENERATOR_PROFILES).
    counter : int
        Running counter for unique customer IDs.

    Returns
    -------
    list[dict]
        List of event dicts ready for ``parse_payload()``.
    """
    profile = GENERATOR_PROFILES.get(profile_name, GENERATOR_PROFILES[DEFAULT_PROFILE])
    action_value = profile["action_value"]
    events: list[dict[str, Any]] = []

    for exp_key, test_rate, ctrl_rate, test_share in profile["experiments"]:
        n_test = max(1, int(batch_size * test_share))
        n_ctrl = max(1, batch_size - n_test)

        # Test arm impressions + captures
        for i in range(n_test):
            cid = f"G{counter + i:06d}"
            events.append({
                "experimentKey": exp_key,
                "arm": "test",
                "eventType": "container",
                "customerId": cid,
            })
            if random.random() < test_rate:
                outcome = random.choice(["Accepted", "Clicked"])
                events.append({
                    "experimentKey": exp_key,
                    "arm": "test",
                    "eventType": "capture",
                    "outcome": outcome,
                    "value": action_value,
                    "customerId": cid,
                })

        # Control arm impressions + captures
        for i in range(n_ctrl):
            cid = f"G{counter + n_test + i:06d}"
            events.append({
                "experimentKey": exp_key,
                "arm": "control",
                "eventType": "container",
                "customerId": cid,
            })
            if random.random() < ctrl_rate:
                outcome = random.choice(["Accepted", "Clicked"])
                events.append({
                    "experimentKey": exp_key,
                    "arm": "control",
                    "eventType": "capture",
                    "outcome": outcome,
                    "value": action_value,
                    "customerId": cid,
                })

    random.shuffle(events)  # interleave experiments naturally
    return events
