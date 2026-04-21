"""Regression tests for the algorithmic rewrite of ``IH.get_sequences``.

The ``ngrams_and_bigrams`` inner helper was rewritten to iterate over
positive-outcome end positions rather than enumerating every (start, length)
pair and filtering. Emission order into the working lists changed, but the
final frequency counts must be bit-identical.

These tests pin that invariant by running the OLD implementation (copied
verbatim below) against the NEW one over hand-crafted, randomised, and
mock-data inputs.
"""

from __future__ import annotations

import random
from collections import defaultdict

import polars as pl
import pytest

from pdstools.ih.IH import IH


# ---------------------------------------------------------------------------
# Reference implementation: the pre-rewrite O(L^2) version of
# ``ngrams_and_bigrams`` plus the surrounding bookkeeping. Copied verbatim
# from ``IH.get_sequences`` prior to the perf(ih) commit so we can compare
# output element-for-element.
# ---------------------------------------------------------------------------


def _get_sequences_reference(
    ih: IH,
    positive_outcome_label: str,
    level: str,
    outcome_column: str,
    customerid_column: str,
):
    cols = [customerid_column, level, outcome_column]
    df = ih.data.select(cols).sort([customerid_column]).collect()

    count_actions = [defaultdict(int), defaultdict(int)]
    count_sequences = [
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    ]
    customer_sequences = []
    customer_outcomes = []

    for _user_id, user_df in df.group_by(customerid_column):
        user_actions = user_df[level].to_list()
        outcome_actions = user_df[outcome_column].to_list()
        outcome_actions = [1 if action == positive_outcome_label else 0 for action in outcome_actions]

        if len(user_actions) < 2:
            continue
        if 1 not in outcome_actions:
            continue

        customer_sequences.append(tuple(user_actions))
        customer_outcomes.append(tuple(outcome_actions))

    def ngrams_and_bigrams(sequences, outcomes):
        ngrams = []
        bigrams = []
        bigrams_all = []

        for seq, out in zip(sequences, outcomes, strict=False):
            ngrams_seen = set()

            for n in range(2, len(seq) + 1):
                for i in range(len(seq) - n + 1):
                    ngram = seq[i : i + n]
                    ngram_outcomes = out[i : i + n]

                    if ngram_outcomes[-1] == 1:
                        if len(ngram) == 2:
                            bigrams_all.append(ngram)
                            bigrams.append(ngram)
                        else:
                            ngrams.append(ngram)
                            for j in range(len(ngram) - 1):
                                bigrams_all.append(ngram[j : j + 2])

                        if ngram not in ngrams_seen:
                            count_sequences[3][ngram] += 1
                            ngrams_seen.add(ngram)

        return ngrams, bigrams, bigrams_all

    ngrams, bigrams, bigrams_all = ngrams_and_bigrams(customer_sequences, customer_outcomes)

    for seq in ngrams:
        count_sequences[1][seq] += 1
    for bigram in bigrams_all:
        count_sequences[0][bigram] += 1
        count_actions[0][(bigram[0],)] += 1
        count_actions[1][(bigram[1],)] += 1
    for bigram in bigrams:
        count_sequences[2][bigram] += 1

    return customer_sequences, customer_outcomes, count_actions, count_sequences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ih_from_rows(rows):
    """Build an IH from a list of (customer_id, action, outcome) tuples.

    Required IH columns (InteractionID / Outcome / OutcomeTime) are
    populated with synthetic values; ``Action`` and ``CustomerID`` are
    carried as extra columns for use by ``get_sequences``.
    """
    from datetime import datetime, timedelta

    base = datetime(2024, 1, 1)
    expanded = [
        (cid, action, outcome, str(i), base + timedelta(seconds=i)) for i, (cid, action, outcome) in enumerate(rows)
    ]
    df = pl.DataFrame(
        expanded,
        schema={
            "CustomerID": pl.Utf8,
            "Action": pl.Utf8,
            "Outcome": pl.Utf8,
            "InteractionID": pl.Utf8,
            "OutcomeTime": pl.Datetime,
        },
        orient="row",
    )
    return IH(df.lazy())


def _assert_counts_equal(new_result, ref_result):
    new_seqs, new_outs, new_actions, new_sequences = new_result
    ref_seqs, ref_outs, ref_actions, ref_sequences = ref_result

    # customer_sequences / customer_outcomes are untouched by the rewrite; they
    # must match exactly (same order — both come from the same sorted groupby).
    assert new_seqs == ref_seqs
    assert new_outs == ref_outs

    assert len(new_actions) == len(ref_actions) == 2
    for i in range(2):
        assert dict(new_actions[i]) == dict(ref_actions[i]), f"count_actions[{i}] mismatch"

    assert len(new_sequences) == len(ref_sequences) == 4
    for i in range(4):
        assert dict(new_sequences[i]) == dict(ref_sequences[i]), f"count_sequences[{i}] mismatch"


def _run_both(ih):
    new = ih.get_sequences(
        positive_outcome_label="Conversion",
        level="Action",
        outcome_column="Outcome",
        customerid_column="CustomerID",
    )
    ref = _get_sequences_reference(
        ih,
        positive_outcome_label="Conversion",
        level="Action",
        outcome_column="Outcome",
        customerid_column="CustomerID",
    )
    return new, ref


# ---------------------------------------------------------------------------
# (a) Bit-exact equivalence — hand-crafted inputs
# ---------------------------------------------------------------------------


def test_equivalence_simple_bigram_ending_positive():
    rows = [
        ("c1", "A", "Impression"),
        ("c1", "B", "Conversion"),
    ]
    ih = _ih_from_rows(rows)
    new, ref = _run_both(ih)
    _assert_counts_equal(new, ref)


def test_equivalence_mixed_outcomes():
    rows = [
        ("c1", "A", "Impression"),
        ("c1", "B", "Impression"),
        ("c1", "C", "Conversion"),
        ("c1", "D", "Impression"),
        ("c1", "E", "Conversion"),
        ("c2", "X", "Impression"),
        ("c2", "Y", "Conversion"),
        ("c2", "Z", "Impression"),
    ]
    ih = _ih_from_rows(rows)
    new, ref = _run_both(ih)
    _assert_counts_equal(new, ref)


# ---------------------------------------------------------------------------
# (c) Hand-crafted edge cases with exact-value assertions
# ---------------------------------------------------------------------------


def test_single_customer_single_bigram_ending_positive():
    rows = [
        ("c1", "A", "Impression"),
        ("c1", "B", "Conversion"),
    ]
    ih = _ih_from_rows(rows)
    seqs, outs, ca, cs = ih.get_sequences("Conversion", "Action", "Outcome", "CustomerID")
    assert seqs == [("A", "B")]
    assert outs == [(0, 1)]
    # Exactly one bigram ("A", "B") ending on a positive outcome.
    assert dict(cs[0]) == {("A", "B"): 1}
    assert dict(cs[1]) == {}
    assert dict(cs[2]) == {("A", "B"): 1}
    assert dict(cs[3]) == {("A", "B"): 1}
    assert dict(ca[0]) == {("A",): 1}
    assert dict(ca[1]) == {("B",): 1}


def test_single_customer_no_positive_outcomes():
    rows = [
        ("c1", "A", "Impression"),
        ("c1", "B", "Impression"),
        ("c1", "C", "Impression"),
    ]
    ih = _ih_from_rows(rows)
    seqs, outs, ca, cs = ih.get_sequences("Conversion", "Action", "Outcome", "CustomerID")
    assert seqs == []
    assert outs == []
    for i in range(4):
        assert dict(cs[i]) == {}
    for i in range(2):
        assert dict(ca[i]) == {}


def test_single_customer_all_positive_outcomes():
    rows = [
        ("c1", "A", "Conversion"),
        ("c1", "B", "Conversion"),
        ("c1", "C", "Conversion"),
    ]
    ih = _ih_from_rows(rows)
    new, ref = _run_both(ih)
    _assert_counts_equal(new, ref)
    _, _, _, cs = new
    # Every position j >= 1 is a valid end.
    # j=1: bigram (A,B)
    # j=2: bigram (B,C), trigram (A,B,C)
    assert dict(cs[2]) == {("A", "B"): 1, ("B", "C"): 1}
    assert dict(cs[1]) == {("A", "B", "C"): 1}
    assert dict(cs[3]) == {
        ("A", "B"): 1,
        ("B", "C"): 1,
        ("A", "B", "C"): 1,
    }
    # bigrams_all: (A,B), (B,C), plus the two inside the trigram: (A,B), (B,C).
    assert dict(cs[0]) == {("A", "B"): 2, ("B", "C"): 2}


def test_single_customer_length_one_skipped():
    rows = [
        ("c1", "A", "Conversion"),
    ]
    ih = _ih_from_rows(rows)
    seqs, outs, ca, cs = ih.get_sequences("Conversion", "Action", "Outcome", "CustomerID")
    assert seqs == []
    assert outs == []
    for i in range(4):
        assert dict(cs[i]) == {}


def test_two_customers_one_with_positive_only():
    rows = [
        ("c1", "A", "Impression"),
        ("c1", "B", "Conversion"),
        ("c2", "X", "Impression"),
        ("c2", "Y", "Impression"),
    ]
    ih = _ih_from_rows(rows)
    new, ref = _run_both(ih)
    _assert_counts_equal(new, ref)
    _, _, ca, cs = new
    # Only c1 contributes — c2 has no positives so is skipped entirely.
    assert dict(cs[0]) == {("A", "B"): 1}
    assert dict(ca[0]) == {("A",): 1}
    assert dict(ca[1]) == {("B",): 1}


def test_repeated_bigram_across_customers_unique_per_customer():
    # Same bigram (A,B) on two customers → cs[3] counts once per customer = 2.
    rows = [
        ("c1", "A", "Impression"),
        ("c1", "B", "Conversion"),
        ("c2", "A", "Impression"),
        ("c2", "B", "Conversion"),
    ]
    ih = _ih_from_rows(rows)
    new, ref = _run_both(ih)
    _assert_counts_equal(new, ref)
    _, _, _, cs = new
    # cs[3] uniqueness is per-customer: c1 contributes (A,B) once, c2 once.
    assert dict(cs[3]) == {("A", "B"): 2}
    # cs[0] aggregates across customers: two (A,B) bigrams total.
    assert dict(cs[0]) == {("A", "B"): 2}


# ---------------------------------------------------------------------------
# (b) Property-based test — randomised inputs (hypothesis is not available,
#     so use random.Random with fixed seeds).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(20)))
def test_equivalence_random_inputs(seed):
    rng = random.Random(seed)
    actions = ["A", "B", "C", "D", "E"]
    rows = []
    n_customers = rng.randint(1, 6)
    for c in range(n_customers):
        length = rng.randint(1, 20)
        for _ in range(length):
            action = rng.choice(actions)
            outcome = "Conversion" if rng.random() < 0.25 else "Impression"
            rows.append((f"c{c}", action, outcome))
    if not rows:
        rows.append(("c0", "A", "Impression"))
    ih = _ih_from_rows(rows)
    new, ref = _run_both(ih)
    _assert_counts_equal(new, ref)


# ---------------------------------------------------------------------------
# (d) Integration test against mock data.
# ---------------------------------------------------------------------------


def test_equivalence_on_mock_data():
    # IH.from_mock_data doesn't emit a customer-id column natively, so we
    # build a moderately-sized randomised corpus via ``_ih_from_rows`` and
    # run the full get_sequences → calculate_pmi → pmi_overview pipeline
    # against both implementations. A fixed random seed makes this fully
    # deterministic.
    rng = random.Random(42)
    actions = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for c in range(200):
        length = rng.randint(2, 25)
        for _ in range(length):
            action = rng.choice(actions)
            outcome = "Conversion" if rng.random() < 0.05 else "Impression"
            rows.append((f"c{c}", action, outcome))
    ih = _ih_from_rows(rows)

    new, ref = _run_both(ih)
    _assert_counts_equal(new, ref)

    # Full pipeline: get_sequences -> calculate_pmi -> pmi_overview should
    # yield an identical DataFrame under either implementation.
    _, _, new_actions, new_sequences = new
    _, _, ref_actions, ref_sequences = ref

    new_pmi = IH.calculate_pmi(new_actions, new_sequences)
    ref_pmi = IH.calculate_pmi(ref_actions, ref_sequences)

    new_df = IH.pmi_overview(new_pmi, new_sequences, new[0], new[1])
    ref_df = IH.pmi_overview(ref_pmi, ref_sequences, ref[0], ref[1])

    from polars.testing import assert_frame_equal

    sort_cols = new_df.columns
    assert_frame_equal(
        new_df.sort(sort_cols),
        ref_df.sort(sort_cols),
        check_row_order=False,
    )


# ---------------------------------------------------------------------------
# Benchmark — manual, not run under pytest by default.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import time

    def _build_corpus(length_range, conv_rate, min_rows, seed=0):
        rng = random.Random(seed)
        actions = [f"A{i}" for i in range(10)]
        rows = []
        while len(rows) < min_rows:
            cid = f"c{rng.randint(0, 999)}"
            length = rng.randint(*length_range)
            for _ in range(length):
                action = rng.choice(actions)
                outcome = "Conversion" if rng.random() < conv_rate else "Impression"
                rows.append((cid, action, outcome))
        return rows

    configs = [
        ("5-30", (5, 30), 0.05, 17_544),
        ("20-80", (20, 80), 0.05, 25_264),
        ("50-150", (50, 150), 0.05, 30_063),
        ("50-150", (50, 150), 0.01, 30_063),
    ]

    print(f"{'L range':>8} {'conv':>6} {'rows':>7} {'ref(s)':>8} {'new(s)':>8} {'speedup':>9}")
    for label, length_range, conv, min_rows in configs:
        rows = _build_corpus(length_range, conv, min_rows)
        ih = _ih_from_rows(rows)

        t0 = time.perf_counter()
        _get_sequences_reference(ih, "Conversion", "Action", "Outcome", "CustomerID")
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        ih.get_sequences(
            positive_outcome_label="Conversion",
            level="Action",
            outcome_column="Outcome",
            customerid_column="CustomerID",
        )
        t_new = time.perf_counter() - t0

        print(f"{label:>8} {conv:>6.0%} {len(rows):>7d} {t_ref:>8.3f} {t_new:>8.3f} {t_ref / t_new:>8.2f}x")
