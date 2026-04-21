"""Regression and edge-case tests for the vectorised overlap helpers.

The tests in this module compare the vectorised polars implementations
in ``pdstools.utils.cdh_utils`` against verbatim copies of the original
Python-loop reference implementations to guarantee bit-exact behaviour.
"""

from __future__ import annotations

import random

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from pdstools.utils import cdh_utils


# ---------------------------------------------------------------------------
# Reference implementations (verbatim from the pre-vectorisation version).
# ---------------------------------------------------------------------------


def _overlap_matrix_reference(
    df: pl.DataFrame,
    list_col: str,
    by: str,
    show_fraction: bool = True,
) -> pl.DataFrame:
    list_col_series = df[list_col]
    nrows = list_col_series.len()
    result = []
    for i in range(nrows):
        set_i = set(list_col_series[i].to_list())
        if show_fraction:
            overlap_w_other_rows = [
                (
                    len(set_i & set(list_col_series[j].to_list())) / len(set(list_col_series[j].to_list()))
                    if i != j
                    else None
                )
                for j in range(nrows)
            ]
        else:
            overlap_w_other_rows = [len(set_i & set(list_col_series[j].to_list())) for j in range(nrows)]
        result.append(
            pl.Series(
                name=f"Overlap_{list_col_series.name}_{df[by][i]}",
                values=overlap_w_other_rows,
            ),
        )
    return pl.DataFrame(result).with_columns(pl.Series(df[by]))


def _overlap_lists_polars_reference(col: pl.Series) -> pl.Series:
    nrows = col.len()
    average_overlap = []
    for i in range(nrows):
        set_i = set(col[i].to_list())
        overlap_w_other_rows = [len(set_i & set(col[j].to_list())) for j in range(nrows) if i != j]
        if len(overlap_w_other_rows) > 0 and len(set_i) > 0:
            average_overlap += [
                sum(overlap_w_other_rows) / len(overlap_w_other_rows) / len(set_i),
            ]
        else:
            average_overlap += [0.0]
    return pl.Series(average_overlap)


# ---------------------------------------------------------------------------
# Hand-crafted exact-value tests for overlap_matrix.
# ---------------------------------------------------------------------------


def _basic_df():
    return pl.DataFrame(
        {
            "Channel": ["Mobile", "Web", "Email"],
            "Actions": [[1, 2, 3], [2, 3, 4, 6], [3, 5, 7, 8]],
        },
    )


def test_overlap_matrix_show_fraction_true_exact():
    out = cdh_utils.overlap_matrix(_basic_df(), "Actions", "Channel", show_fraction=True)
    expected = pl.DataFrame(
        {
            "Overlap_Actions_Mobile": [None, 0.5, 0.25],
            "Overlap_Actions_Web": [2.0 / 3, None, 0.25],
            "Overlap_Actions_Email": [1.0 / 3, 0.25, None],
            "Channel": ["Mobile", "Web", "Email"],
        },
    )
    assert_frame_equal(out, expected)


def test_overlap_matrix_show_fraction_false_exact():
    out = cdh_utils.overlap_matrix(_basic_df(), "Actions", "Channel", show_fraction=False)
    expected = pl.DataFrame(
        {
            "Overlap_Actions_Mobile": [3, 2, 1],
            "Overlap_Actions_Web": [2, 4, 1],
            "Overlap_Actions_Email": [1, 1, 4],
            "Channel": ["Mobile", "Web", "Email"],
        },
    )
    assert_frame_equal(out, expected)


def test_overlap_matrix_single_row():
    df = pl.DataFrame({"Channel": ["A"], "Actions": [[1, 2, 3]]})
    # show_fraction=True: only diagonal -> None
    out_true = cdh_utils.overlap_matrix(df, "Actions", "Channel", show_fraction=True)
    expected_true = pl.DataFrame(
        {"Overlap_Actions_A": [None], "Channel": ["A"]},
        schema={"Overlap_Actions_A": pl.Float64, "Channel": pl.Utf8},
    )
    assert_frame_equal(out_true, expected_true)

    out_false = cdh_utils.overlap_matrix(df, "Actions", "Channel", show_fraction=False)
    expected_false = pl.DataFrame({"Overlap_Actions_A": [3], "Channel": ["A"]})
    assert_frame_equal(out_false, expected_false)


def test_overlap_matrix_two_rows_disjoint():
    df = pl.DataFrame({"Channel": ["X", "Y"], "Actions": [[1, 2], [3, 4]]})
    out = cdh_utils.overlap_matrix(df, "Actions", "Channel", show_fraction=True)
    expected = pl.DataFrame(
        {
            "Overlap_Actions_X": [None, 0.0],
            "Overlap_Actions_Y": [0.0, None],
            "Channel": ["X", "Y"],
        },
    )
    assert_frame_equal(out, expected)


def test_overlap_matrix_all_rows_identical():
    df = pl.DataFrame(
        {"Channel": ["A", "B", "C"], "Actions": [[1, 2], [1, 2], [1, 2]]},
    )
    out = cdh_utils.overlap_matrix(df, "Actions", "Channel", show_fraction=True)
    expected = pl.DataFrame(
        {
            "Overlap_Actions_A": [None, 1.0, 1.0],
            "Overlap_Actions_B": [1.0, None, 1.0],
            "Overlap_Actions_C": [1.0, 1.0, None],
            "Channel": ["A", "B", "C"],
        },
    )
    assert_frame_equal(out, expected)


def test_overlap_matrix_duplicate_elements_within_list():
    # Set semantics: duplicates within a list are collapsed.
    df = pl.DataFrame(
        {"Channel": ["A", "B"], "Actions": [[1, 1, 2, 2, 3], [2, 3, 3]]},
    )
    out = cdh_utils.overlap_matrix(df, "Actions", "Channel", show_fraction=True)
    expected = pl.DataFrame(
        {
            "Overlap_Actions_A": [None, 1.0],  # |{1,2,3}∩{2,3}|/|{2,3}| = 2/2
            "Overlap_Actions_B": [2.0 / 3, None],
            "Channel": ["A", "B"],
        },
    )
    assert_frame_equal(out, expected)


# ---------------------------------------------------------------------------
# Hand-crafted exact-value tests for overlap_lists_polars.
# ---------------------------------------------------------------------------


def test_overlap_lists_basic():
    s = pl.Series([[1, 2, 3], [2, 3, 4, 6], [3, 5, 7, 8]])
    out = cdh_utils.overlap_lists_polars(s)
    assert out.to_list() == [0.5, 0.375, 0.25]
    assert out.dtype == pl.Float64


def test_overlap_lists_single_row():
    s = pl.Series([[1, 2, 3]])
    out = cdh_utils.overlap_lists_polars(s)
    assert out.to_list() == [0.0]
    assert out.dtype == pl.Float64


def test_overlap_lists_empty_list_element():
    # If one row has an empty list, its overlap is 0.0; others are unaffected.
    s = pl.Series([[1, 2], [], [1, 3]])
    out = cdh_utils.overlap_lists_polars(s)
    ref = _overlap_lists_polars_reference(s)
    assert_series_equal(out, ref)
    assert out.to_list() == [0.25, 0.0, 0.25]


def test_overlap_lists_all_disjoint():
    s = pl.Series([[1], [2], [3]])
    out = cdh_utils.overlap_lists_polars(s)
    assert out.to_list() == [0.0, 0.0, 0.0]


def test_overlap_lists_via_map_batches():
    df = pl.DataFrame(
        {"Channel": ["M", "W", "E"], "Actions": [[1, 2, 3], [2, 3, 4, 6], [3, 5, 7, 8]]},
    )
    out = df.with_columns(pl.col("Actions").map_batches(cdh_utils.overlap_lists_polars))
    assert out["Actions"].to_list() == [0.5, 0.375, 0.25]


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_overlap_matrix_empty_dataframe():
    df = pl.DataFrame(
        {"Channel": [], "Actions": []},
        schema={"Channel": pl.Utf8, "Actions": pl.List(pl.Int64)},
    )
    out = cdh_utils.overlap_matrix(df, "Actions", "Channel", show_fraction=True)
    assert out.height == 0
    assert "Channel" in out.columns


def test_overlap_lists_empty_series():
    s = pl.Series([], dtype=pl.List(pl.Int64))
    out = cdh_utils.overlap_lists_polars(s)
    assert out.to_list() == []
    assert out.dtype == pl.Float64


# ---------------------------------------------------------------------------
# Reference-equivalence (random fuzz) tests.
# ---------------------------------------------------------------------------


def _random_lists_df(rng: random.Random, n: int, max_list_len: int, vocab: int, min_list_len: int = 1):
    by_vals = [f"row_{i}" for i in range(n)]
    lists = []
    for _ in range(n):
        k = rng.randint(min_list_len, max_list_len)
        lists.append([rng.randrange(vocab) for _ in range(k)])
    return pl.DataFrame({"by": by_vals, "lst": lists})


@pytest.mark.parametrize(
    ("n", "max_len", "vocab", "seed"),
    [
        (2, 5, 10, 1),
        (5, 8, 6, 2),
        (10, 4, 5, 3),
        (20, 10, 15, 4),
        (50, 6, 8, 5),
    ],
)
def test_overlap_matrix_matches_reference_random(n, max_len, vocab, seed):
    rng = random.Random(seed)
    df = _random_lists_df(rng, n, max_len, vocab)

    for show_fraction in (True, False):
        new = cdh_utils.overlap_matrix(df, "lst", "by", show_fraction=show_fraction)
        ref = _overlap_matrix_reference(df, "lst", "by", show_fraction=show_fraction)
        assert_frame_equal(new, ref)


@pytest.mark.parametrize(
    ("n", "max_len", "vocab", "seed"),
    [
        (1, 5, 10, 1),
        (2, 5, 5, 2),
        (5, 6, 4, 3),
        (15, 8, 12, 4),
        (30, 4, 6, 5),
        (60, 10, 8, 6),
    ],
)
def test_overlap_lists_matches_reference_random(n, max_len, vocab, seed):
    rng = random.Random(seed)
    df = _random_lists_df(rng, n, max_len, vocab)
    new = cdh_utils.overlap_lists_polars(df["lst"])
    ref = _overlap_lists_polars_reference(df["lst"])
    assert_series_equal(new, ref)


# ---------------------------------------------------------------------------
# Benchmark.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    def bench(n: int, m: int) -> None:
        rng = random.Random(0)
        df = _random_lists_df(rng, n, m, vocab=max(m * 2, 10))

        t0 = time.perf_counter()
        _overlap_matrix_reference(df, "lst", "by", show_fraction=True)
        t_old_mat = time.perf_counter() - t0

        t0 = time.perf_counter()
        cdh_utils.overlap_matrix(df, "lst", "by", show_fraction=True)
        t_new_mat = time.perf_counter() - t0

        t0 = time.perf_counter()
        _overlap_lists_polars_reference(df["lst"])
        t_old_avg = time.perf_counter() - t0

        t0 = time.perf_counter()
        cdh_utils.overlap_lists_polars(df["lst"])
        t_new_avg = time.perf_counter() - t0

        print(
            f"N={n:5d} avg_list={m:3d} | "
            f"matrix old={t_old_mat:7.3f}s new={t_new_mat:7.3f}s "
            f"({t_old_mat / max(t_new_mat, 1e-9):6.1f}x) | "
            f"avg    old={t_old_avg:7.3f}s new={t_new_avg:7.3f}s "
            f"({t_old_avg / max(t_new_avg, 1e-9):6.1f}x)"
        )

    for n in (10, 50, 200, 1000):
        for m in (10, 50):
            bench(n, m)
