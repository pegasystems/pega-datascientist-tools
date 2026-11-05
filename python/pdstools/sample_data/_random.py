"""Independent deterministic random streams for synthetic entities."""

from __future__ import annotations

import hashlib
import json
import random

DEFAULT_SEED = 42


def entity_seed(seed: int, *identity: str) -> int:
    """Derive a stable seed without Python's process-randomized hash.

    Parameters
    ----------
    seed : int
        Root seed for the synthetic world.
    *identity : str
        Ordered identity components, such as a view, configuration and model.
        Components are encoded separately so separators cannot cause collisions.

    Returns
    -------
    int
        SHA-256-derived unsigned integer independent of catalog order.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not identity or any(not isinstance(part, str) or not part for part in identity):
        raise ValueError("Provide at least one non-empty string identity component.")
    payload = json.dumps([seed, *identity], ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def entity_rng(seed: int, *identity: str) -> random.Random:
    """Return an independent random stream for one synthetic entity.

    Parameters
    ----------
    seed : int
        Root seed.
    *identity : str
        Non-empty ordered entity identity components.

    Returns
    -------
    random.Random
        Fresh local stream that does not mutate global random state.
    """
    return random.Random(entity_seed(seed, *identity))
