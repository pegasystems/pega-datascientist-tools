#!/usr/bin/env python3
"""
One-off script to produce data/agb/ModelExportWithSampleCount.json from a
raw production AGB model export.

DO NOT COMMIT production model exports or raw customer data.
This script is kept in source control as documentation of the anonymization
steps applied; it cannot be re-run without the original un-anonymized export.

Anonymization steps applied (in order):
  1. Drop the last N trees (keep only KEEP_TREES trees).
     Late trees have tiny gain and removing them changes the tree-count
     fingerprint while preserving all statistically interesting patterns.
  2. Rename predictors that contain non-generic business names so that
     no client-specific field names remain in split strings.
  3. Scale training response counts by a non-round factor so the exact
     positive / negative tallies don't match the source.
  4. Round all float values to MAX_DECIMALS significant decimal places to
     reduce file size without losing any precision relevant to the plots.
  5. Write compact JSON (no extra whitespace) to minimise committed size.
"""

import json
import pathlib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = pathlib.Path("REPLACE_WITH_ORIGINAL_EXPORT_PATH.json")
OUTPUT_FILE = pathlib.Path(__file__).parent.parent / "data" / "agb" / "ModelExportWithSampleCount.json"

KEEP_TREES = 83  # drop trees 83-99 (tiny gain, near-zero contribution)
VOLUME_SCALE = 0.71  # multiply pos/neg/total counts by this factor
MAX_DECIMALS = 5  # decimal places for score, gain, threshold floats

# Map of predictor names to replace.  Keys are literal prefixes or full names
# that appear at the start of split strings (before the operator).
# Anything listed here is replaced in-place in every split string.
PREDICTOR_RENAMES: dict[str, str] = {
    # Looks like a client-specific segment/pricing score
    "Customer.Scores.Cisegmpricesen": "Customer.Scores.ExtModelScore0",
    # Channel name that could identify the client's system
    "IH.Millenet.Inbound.": "IH.Web.Inbound.",
    # Business-specific field names
    "Customer.RetailTrxDays": "Customer.AccountAgeDays",
    "Param.ClNewAbandonCnt1d": "Param.SessionAbandonCnt1d",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rename_split(split: str, renames: dict[str, str]) -> str:
    """Replace a predictor name at the start of a split string."""
    for old, new in renames.items():
        if split.startswith(old):
            return new + split[len(old) :]
    return split


def round_floats(obj: object, ndigits: int) -> object:
    """Recursively round all float values in a JSON-compatible structure."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def anonymize_tree(node: object, renames: dict[str, str]) -> object:
    """Rename predictors in all split strings within a tree node."""
    if not isinstance(node, dict):
        return node
    result = {}
    for k, v in node.items():
        if k == "split" and isinstance(v, str):
            result[k] = rename_split(v, renames)
        elif k in ("left", "right"):
            result[k] = anonymize_tree(v, renames)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Raw export not found: {INPUT_FILE}\nThis script requires the original un-anonymized model export."
        )

    print(f"Reading {INPUT_FILE} …")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    trees: list[dict] = data["model"]["booster"]["trees"]
    original_tree_count = len(trees)

    # Step 1 — trim trailing trees
    trees = trees[:KEEP_TREES]
    print(f"  Trees: {original_tree_count} → {len(trees)} (dropped last {original_tree_count - KEEP_TREES})")

    # Step 2 — rename predictors in split strings
    trees = [anonymize_tree(t, PREDICTOR_RENAMES) for t in trees]
    print(f"  Predictor renames applied: {list(PREDICTOR_RENAMES.keys())}")

    # Step 3 — scale response volumes
    ts = data["trainingStats"]
    original_pos = ts["positiveCount"]
    original_neg = ts["negativeCount"]
    new_pos = round(original_pos * VOLUME_SCALE)
    new_neg = round(original_neg * VOLUME_SCALE)
    new_total = new_pos + new_neg
    ts["positiveCount"] = new_pos
    ts["negativeCount"] = new_neg
    ts["totalCount"] = new_total
    data["successRate"] = round(new_pos / new_total, MAX_DECIMALS)
    print(f"  Volumes scaled by {VOLUME_SCALE}: pos {original_pos:,} → {new_pos:,}, neg {original_neg:,} → {new_neg:,}")

    # Step 4 — round all floats
    data["model"] = {"booster": {"trees": trees}}
    data = round_floats(data, MAX_DECIMALS)
    print(f"  Floats rounded to {MAX_DECIMALS} decimal places")

    # Step 5 — write compact JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(data, separators=(",", ":"))
    OUTPUT_FILE.write_text(output_text, encoding="utf-8")

    size_mb = len(output_text) / 1024 / 1024
    print(f"Written {OUTPUT_FILE}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
