"""
Testing the functionality of the types module
"""

import polars as pl
from typing import Union
from pdstools.utils.types import ANY_FRAME, QUERY


def test_any_frame_type_alias():
    """Test that the ANY_FRAME type alias is defined correctly"""
    # Create instances of DataFrame and LazyFrame
    df = pl.DataFrame({"A": [1, 2, 3]})
    ldf = pl.DataFrame({"A": [1, 2, 3]}).lazy()

    # Check that the type alias is a Union of DataFrame and LazyFrame
    assert ANY_FRAME.__origin__ is Union
    assert pl.DataFrame in ANY_FRAME.__args__
    assert pl.LazyFrame in ANY_FRAME.__args__

    # Check that both instances are of the expected types
    assert isinstance(df, pl.DataFrame)
    assert isinstance(ldf, pl.LazyFrame)


def test_query_type_alias():
    """Test that the QUERY type alias is defined correctly"""
    # Create instances of the different query types
    expr = pl.col("A") > 5
    expr_list = [pl.col("A") > 5, pl.col("B").is_null()]
    dict_query = {"column": ["value1", "value2"]}

    # Check that the type alias is a Union of the expected types
    assert QUERY.__origin__ is Union
    assert pl.Expr in QUERY.__args__

    # Check that the instances are of the expected types
    assert isinstance(expr, pl.Expr)
    assert isinstance(expr_list, list)
    assert all(isinstance(e, pl.Expr) for e in expr_list)
    assert isinstance(dict_query, dict)
