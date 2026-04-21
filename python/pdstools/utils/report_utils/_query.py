"""Serialise/deserialise pdstools ``QUERY`` objects for embedding in reports."""

import io
import json

import polars as pl

from ..types import QUERY


def serialize_query(query: QUERY | None) -> dict | None:
    if query is None:
        return None

    if isinstance(query, pl.Expr):
        query = [query]

    if isinstance(query, (list, tuple)):
        serialized_exprs = {}
        for i, expr in enumerate(query):
            if not isinstance(expr, pl.Expr):
                raise ValueError("All items in query list must be Expressions")
            serialized_exprs[str(i)] = json.loads(expr.meta.serialize(format="json"))
        return {"type": "expr_list", "expressions": serialized_exprs}

    if isinstance(query, dict):
        return {"type": "dict", "data": query}

    raise ValueError(f"Unsupported query type: {type(query)}")


def deserialize_query(serialized_query: dict | None) -> QUERY | None:
    """Deserialize a query that was previously serialized with serialize_query.

    Parameters
    ----------
    serialized_query : dict | None
        A serialized query dictionary created by serialize_query

    Returns
    -------
    QUERY | None
        The deserialized query

    """
    if serialized_query is None:
        return None

    if serialized_query["type"] == "expr_list":
        expr_list = []
        for _, val in serialized_query["expressions"].items():
            json_str = json.dumps(val)
            str_io = io.StringIO(json_str)
            expr_list.append(pl.Expr.deserialize(str_io, format="json"))
        return expr_list

    if serialized_query["type"] == "dict":
        return serialized_query["data"]

    raise ValueError(f"Unknown query type: {serialized_query['type']}")
