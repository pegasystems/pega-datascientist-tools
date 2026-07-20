from __future__ import annotations

import csv
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from ..pega_io.File import read_data
from ..prediction.Prediction import Prediction
from ..utils import cdh_utils
from .ADMDatamart import ADMDatamart

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date, datetime

HealthCheckSource = str | os.PathLike[str] | BytesIO
ConfiguredDtype = str | pl.DataType | type[pl.DataType]

logger = logging.getLogger(__name__)

_DTYPE_ALIASES: dict[str, pl.DataType | type[pl.DataType]] = {
    "bool": pl.Boolean,
    "boolean": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime,
    "float": pl.Float64,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "int": pl.Int64,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "str": pl.String,
    "string": pl.String,
    "utf8": pl.String,
}

MODEL_CACHE_FILENAME = "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
PREDICTOR_CACHE_FILENAME = "PR_DATA_DM_ADMMART_PRED.parquet"
PREDICTION_CACHE_FILENAME = "PR_DATA_DM_SNAPSHOTS.parquet"
_MODEL_CONTEXT_COLUMNS = frozenset({"Channel", "Direction", "Issue", "Group"})


@dataclass(frozen=True)
class HealthCheckReadOptions:
    """Structured reader options for one Health Check input source.

    Parameters
    ----------
    delimiter : str, optional
        Field delimiter for CSV, TSV, or TXT input.
    quote_char : str, optional
        Quote character for CSV, TSV, or TXT input. Set to None to disable
        quote handling.
    encoding : str, default "utf8"
        Text encoding forwarded to Polars.
    has_header : bool, default True
        Whether the first input row contains column names.
    skip_rows : int, default 0
        Number of rows to skip before reading delimited text.
    infer_schema_length : int, optional
        Number of rows used for schema inference. Use zero to read all columns
        as strings before applying normalization overrides.
    ignore_errors : bool, default False
        Continue parsing rows that contain values incompatible with the inferred
        delimited-text schema.
    null_values : tuple[str, ...], optional
        Values interpreted as null for delimited text.
    schema_overrides : Mapping[str, ConfiguredDtype]
        Reader-level column type overrides.
    excel_sheet_name : str, optional
        Excel sheet name to read.
    excel_sheet_id : int, optional
        One-based Excel sheet index to read.
    excel_header_row : int, optional
        Zero-based Excel row containing column names.
    """

    delimiter: str | None = None
    quote_char: str | None = '"'
    encoding: str = "utf8"
    has_header: bool = True
    skip_rows: int = 0
    infer_schema_length: int | None = None
    ignore_errors: bool = False
    null_values: tuple[str, ...] | None = None
    schema_overrides: Mapping[str, ConfiguredDtype] = field(default_factory=dict)
    excel_sheet_name: str | None = None
    excel_sheet_id: int | None = None
    excel_header_row: int | None = None

    def __post_init__(self) -> None:
        if self.delimiter is not None and len(self.delimiter) != 1:
            raise ValueError("delimiter must contain exactly one character")
        if self.quote_char is not None and len(self.quote_char) != 1:
            raise ValueError("quote_char must contain exactly one character")
        if self.skip_rows < 0:
            raise ValueError("skip_rows cannot be negative")
        if self.infer_schema_length is not None and self.infer_schema_length < 0:
            raise ValueError("infer_schema_length cannot be negative")
        if self.excel_sheet_name is not None and self.excel_sheet_id is not None:
            raise ValueError("Configure either excel_sheet_name or excel_sheet_id, not both")
        if self.excel_sheet_id is not None and self.excel_sheet_id < 1:
            raise ValueError("excel_sheet_id must be one or greater")
        if self.excel_header_row is not None and self.excel_header_row < 0:
            raise ValueError("excel_header_row cannot be negative")


@dataclass(frozen=True)
class HealthCheckRowFilter:
    """Simple row predicate applied during Health Check source normalization.

    Parameters
    ----------
    column : str
        Column to evaluate, matched case-insensitively.
    operator : str
        Predicate operator. Supported values are ``==``, ``!=``, ``contains``,
        ``not_contains``, ``starts_with``, ``not_starts_with``, ``ends_with``,
        ``not_ends_with``, ``is_null``, and ``is_not_null``.
    value : str, optional
        Predicate value. Required for all operators except ``is_null`` and
        ``is_not_null``.
    """

    column: str
    operator: Literal[
        "==",
        "!=",
        "contains",
        "not_contains",
        "starts_with",
        "not_starts_with",
        "ends_with",
        "not_ends_with",
        "is_null",
        "is_not_null",
    ]
    value: str | None = None


@dataclass(frozen=True)
class SourceNormalizationOptions:
    """Declarative repairs applied to one input before pdstools validation.

    Parameters
    ----------
    rename_columns : Mapping[str, str]
        Existing columns to rename, matched case-insensitively.
    row_filters : tuple[HealthCheckRowFilter, ...]
        Simple row predicates to apply before column cleanup.
    text_replacements : Mapping[str, Mapping[str, str]]
        Literal text replacements by column, matched case-insensitively.
    fill_null_values : Mapping[str, object]
        Existing columns whose null values should be filled with a constant,
        matched case-insensitively.
    derived_columns : Mapping[str, str]
        Columns to create or replace from a source column (``target=source``) or
        numeric subtraction (``target=left-right``), matched case-insensitively.
        String concatenation is supported as
        ``target=concat(column, "literal", other)``.
    drop_columns : tuple[str, ...]
        Existing columns to remove, matched case-insensitively.
    type_overrides : Mapping[str, ConfiguredDtype]
        Non-strict casts to apply, matched case-insensitively.
    timestamp_column : str, optional
        Existing timestamp column to parse, or a new column to create when a
        fallback is configured.
    timestamp_format : str, optional
        Explicit ``strptime`` format for a string timestamp column.
    timestamp_fallback : datetime or date, optional
        Value used for missing or unparseable timestamps.
    constant_columns : Mapping[str, object]
        New columns and their constant values.
    """

    rename_columns: Mapping[str, str] = field(default_factory=dict)
    row_filters: tuple[HealthCheckRowFilter, ...] = ()
    text_replacements: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    fill_null_values: Mapping[str, object] = field(default_factory=dict)
    derived_columns: Mapping[str, str] = field(default_factory=dict)
    drop_columns: tuple[str, ...] = ()
    type_overrides: Mapping[str, ConfiguredDtype] = field(default_factory=dict)
    timestamp_column: str | None = None
    timestamp_format: str | None = None
    timestamp_fallback: datetime | date | None = None
    constant_columns: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceImportOptions:
    """Read and normalization options for one Health Check source."""

    read: HealthCheckReadOptions = field(default_factory=HealthCheckReadOptions)
    normalize: SourceNormalizationOptions = field(default_factory=SourceNormalizationOptions)


@dataclass(frozen=True)
class HealthCheckSourceMetadata:
    """Non-content metadata describing an imported source."""

    name: str
    extension: str


@dataclass(frozen=True)
class HealthCheckImportResult:
    """Imported objects and constructor-ready normalized source data."""

    datamart: ADMDatamart
    prediction: Prediction | None
    model_data: pl.LazyFrame
    predictor_data: pl.LazyFrame | None
    prediction_data: pl.LazyFrame | None
    sources: Mapping[str, HealthCheckSourceMetadata]
    warnings: tuple[str, ...]


def _configured_dtype(dtype: ConfiguredDtype) -> pl.DataType | type[pl.DataType]:
    if not isinstance(dtype, str):
        return dtype
    try:
        return _DTYPE_ALIASES[dtype.casefold()]
    except KeyError:
        supported = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"Unsupported data type '{dtype}'. Supported values: {supported}") from None


def _column_lookup(df: pl.LazyFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in df.collect_schema().names():
        folded = column.casefold()
        if folded in lookup:
            raise ValueError(f"Columns '{lookup[folded]}' and '{column}' differ only by case")
        lookup[folded] = column
    return lookup


def _resolve_column(df: pl.LazyFrame, configured: str, purpose: str) -> str:
    lookup = _column_lookup(df)
    try:
        return lookup[configured.casefold()]
    except KeyError:
        raise ValueError(f"{purpose} column '{configured}' was not found") from None


def _count_message(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _row_filter_expr(df: pl.LazyFrame, row_filter: HealthCheckRowFilter) -> pl.Expr:
    column = _resolve_column(df, row_filter.column, "Row filter")
    operator = row_filter.operator
    value = row_filter.value
    if operator in {"is_null", "is_not_null"}:
        if value not in (None, ""):
            raise ValueError(f"Row filter operator '{operator}' does not accept a value")
        expr = pl.col(column).is_null()
        return ~expr if operator == "is_not_null" else expr
    if value is None:
        raise ValueError(f"Row filter operator '{operator}' requires a value")

    if operator == "==":
        return pl.col(column).cast(pl.String) == value
    if operator == "!=":
        return pl.col(column).cast(pl.String) != value
    if operator == "contains":
        return pl.col(column).cast(pl.String).str.contains(re.escape(value))
    if operator == "not_contains":
        return ~pl.col(column).cast(pl.String).str.contains(re.escape(value)).fill_null(False)
    if operator == "starts_with":
        return pl.col(column).cast(pl.String).str.starts_with(value)
    if operator == "not_starts_with":
        return ~pl.col(column).cast(pl.String).str.starts_with(value).fill_null(False)
    if operator == "ends_with":
        return pl.col(column).cast(pl.String).str.ends_with(value)
    if operator == "not_ends_with":
        return ~pl.col(column).cast(pl.String).str.ends_with(value).fill_null(False)
    raise ValueError(f"Unsupported row filter operator '{operator}'")


def _derived_column_expr(df: pl.LazyFrame, target: str, expression: str) -> pl.Expr:
    expression = expression.strip()
    if expression.startswith("concat(") and expression.endswith(")"):
        arguments = [argument.strip() for argument in expression[7:-1].split(",")]
        if not arguments:
            raise ValueError(f"Derived column '{target}' has an empty concat expression")
        expressions: list[pl.Expr] = []
        for argument in arguments:
            if len(argument) >= 2 and argument[0] == argument[-1] and argument[0] in {'"', "'"}:
                expressions.append(pl.lit(argument[1:-1]))
            else:
                expressions.append(pl.col(_resolve_column(df, argument, "Derived")).cast(pl.String))
        return pl.concat_str(expressions).alias(target)
    if "-" in expression:
        left, right = (part.strip() for part in expression.split("-", 1))
        if not left or not right:
            raise ValueError(f"Derived column '{target}' has an invalid subtraction expression")
        left_column = _resolve_column(df, left, "Derived")
        right_column = _resolve_column(df, right, "Derived")
        return (
            pl.col(left_column).cast(pl.Float64, strict=False) - pl.col(right_column).cast(pl.Float64, strict=False)
        ).alias(target)
    source_column = _resolve_column(df, expression.strip(), "Derived")
    return pl.col(source_column).alias(target)


def normalize_health_check_data(
    df: pl.LazyFrame,
    options: SourceNormalizationOptions | None = None,
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """Apply safe, declarative repairs to one Health Check input.

    Parameters
    ----------
    df : pl.LazyFrame
        Raw input data.
    options : SourceNormalizationOptions, optional
        Repairs to apply. Column names are matched case-insensitively.

    Returns
    -------
    tuple[pl.LazyFrame, tuple[str, ...]]
        Normalized data and data-safe descriptions of applied repairs.

    Raises
    ------
    ValueError
        If configured columns do not exist, settings conflict, or a configured
        data type is unsupported.
    """
    options = options or SourceNormalizationOptions()
    warnings: list[str] = []

    if options.rename_columns:
        lookup = _column_lookup(df)
        renamed: dict[str, str] = {}
        targets = set(lookup)
        for configured, target in options.rename_columns.items():
            source = _resolve_column(df, configured, "Rename")
            folded_source = source.casefold()
            folded_target = target.casefold()
            if folded_target in targets and folded_target != folded_source:
                raise ValueError(f"Rename target column '{target}' already exists")
            renamed[source] = target
            targets.discard(folded_source)
            targets.add(folded_target)
        df = df.rename(renamed)
        warnings.append(f"Renamed {_count_message(len(renamed), 'configured column', 'configured columns')}.")

    if options.row_filters:
        for row_filter in options.row_filters:
            df = df.filter(_row_filter_expr(df, row_filter))
        warnings.append(f"Applied {_count_message(len(options.row_filters), 'row filter', 'row filters')}.")

    if options.text_replacements:
        expressions: list[pl.Expr] = []
        for configured, replacements in options.text_replacements.items():
            column = _resolve_column(df, configured, "Text replacement")
            expr = pl.col(column).cast(pl.String)
            for old, new in replacements.items():
                expr = expr.str.replace_all(old, new, literal=True)
            expressions.append(expr.alias(column))
        df = df.with_columns(expressions)
        warnings.append(
            f"Applied {_count_message(len(expressions), 'text replacement column', 'text replacement columns')}.",
        )

    dropped = {column.casefold() for column in options.drop_columns}
    constants = {column.casefold() for column in options.constant_columns}
    derived = {column.casefold() for column in options.derived_columns}
    conflict = dropped & (constants | derived)
    if conflict:
        raise ValueError("A column cannot be both dropped and created")

    if options.type_overrides:
        type_override_expressions: list[pl.Expr] = []
        for configured, dtype in options.type_overrides.items():
            column = _resolve_column(df, configured, "Type override")
            type_override_expressions.append(pl.col(column).cast(_configured_dtype(dtype), strict=False))
        df = df.with_columns(type_override_expressions)
        warnings.append(
            f"Applied {_count_message(len(type_override_expressions), 'non-strict type override', 'non-strict type overrides')}.",
        )

    if options.fill_null_values:
        fill_null_expressions = []
        for configured, value in options.fill_null_values.items():
            column = _resolve_column(df, configured, "Fill-null")
            fill_null_expressions.append(pl.col(column).fill_null(value).alias(column))
        df = df.with_columns(fill_null_expressions)
        warnings.append(
            f"Filled {_count_message(len(fill_null_expressions), 'column null value', 'column null values')}."
        )

    if options.derived_columns:
        expressions = []
        for target, expression in options.derived_columns.items():
            expressions.append(_derived_column_expr(df, target, expression))
        df = df.with_columns(expressions)
        warnings.append(f"Created {_count_message(len(expressions), 'derived column', 'derived columns')}.")

    if options.drop_columns:
        resolved = [_resolve_column(df, column, "Drop") for column in options.drop_columns]
        df = df.drop(resolved)
        warnings.append(f"Dropped {_count_message(len(resolved), 'configured column', 'configured columns')}.")

    if options.timestamp_column is not None:
        lookup = _column_lookup(df)
        timestamp_column = lookup.get(options.timestamp_column.casefold())
        if timestamp_column is None:
            if options.timestamp_fallback is None:
                raise ValueError(
                    f"Timestamp column '{options.timestamp_column}' was not found and no fallback was configured",
                )
            df = df.with_columns(
                pl.lit(options.timestamp_fallback).cast(pl.Datetime).alias(options.timestamp_column),
            )
            warnings.append(
                f"Added missing timestamp column '{options.timestamp_column}' from its configured fallback.",
            )
        else:
            timestamp_type = df.collect_schema()[timestamp_column]
            if timestamp_type.is_temporal():
                timestamp_expr = pl.col(timestamp_column).cast(pl.Datetime)
            else:
                timestamp_expr = cdh_utils.parse_pega_date_time_formats(
                    timestamp_column,
                    timestamp_fmt=options.timestamp_format,
                )
            if options.timestamp_fallback is not None:
                timestamp_expr = timestamp_expr.fill_null(pl.lit(options.timestamp_fallback).cast(pl.Datetime))
            df = df.with_columns(timestamp_expr.alias(timestamp_column))
            suffix = " with a configured fallback" if options.timestamp_fallback is not None else ""
            warnings.append(f"Parsed timestamp column '{timestamp_column}'{suffix}.")

    if options.constant_columns:
        existing = _column_lookup(df)
        for column in options.constant_columns:
            if column.casefold() in existing:
                raise ValueError(f"Constant column '{column}' already exists")
        df = df.with_columns([pl.lit(value).alias(column) for column, value in options.constant_columns.items()])
        warnings.append(
            f"Added {_count_message(len(options.constant_columns), 'constant column', 'constant columns')}.",
        )

    return df, tuple(warnings)


def _polars_read_options(
    options: HealthCheckReadOptions,
    extension: str,
    *,
    full_schema_inference: bool = False,
) -> dict:
    schema_overrides = {column: _configured_dtype(dtype) for column, dtype in options.schema_overrides.items()}
    extension = extension.casefold()
    if extension in {".csv", ".tsv", ".txt"}:
        read_options: dict = {
            "encoding": options.encoding,
            "has_header": options.has_header,
            "skip_rows": options.skip_rows,
            "ignore_errors": options.ignore_errors,
            "quote_char": options.quote_char,
        }
        if options.delimiter is not None:
            read_options["separator"] = options.delimiter
        if options.infer_schema_length is not None:
            read_options["infer_schema_length"] = options.infer_schema_length
        elif full_schema_inference:
            read_options["infer_schema_length"] = None
        if options.null_values is not None:
            read_options["null_values"] = list(options.null_values)
        if schema_overrides:
            read_options["schema_overrides"] = schema_overrides
        return read_options

    if extension in {".json", ".jsonl", ".ndjson", ".zip", ".gz"}:
        read_options = {}
        if options.infer_schema_length is not None:
            read_options["infer_schema_length"] = options.infer_schema_length
        elif full_schema_inference:
            read_options["infer_schema_length"] = None
        if schema_overrides:
            read_options["schema_overrides"] = schema_overrides
        return read_options

    if extension in {".xlsx", ".xls"}:
        read_options = {"has_header": options.has_header}
        if options.excel_sheet_name is not None:
            read_options["sheet_name"] = options.excel_sheet_name
        if options.excel_sheet_id is not None:
            read_options["sheet_id"] = options.excel_sheet_id
        if options.infer_schema_length is not None:
            read_options["infer_schema_length"] = options.infer_schema_length
        if schema_overrides:
            read_options["schema_overrides"] = schema_overrides
        if options.excel_header_row is not None:
            read_options["read_options"] = {"header_row": options.excel_header_row}
        return read_options

    return {}


def _source_metadata(source: HealthCheckSource) -> HealthCheckSourceMetadata:
    name = getattr(source, "name", "uploaded-data") if isinstance(source, BytesIO) else os.fspath(source)
    source_name = Path(name).name
    return HealthCheckSourceMetadata(
        name=source_name,
        extension=Path(source_name).suffix.casefold(),
    )


def _source_read_extension(source: HealthCheckSource) -> str:
    metadata = _source_metadata(source)
    if metadata.extension == ".zip":
        return Path(metadata.name.removesuffix(".zip")).suffix.casefold() or ".zip"
    if metadata.extension == ".gz":
        return Path(metadata.name.removesuffix(".gz")).suffix.casefold() or ".json"
    return metadata.extension


def _readable_source(source: HealthCheckSource) -> str | Path | BytesIO:
    return source if isinstance(source, (str, BytesIO)) else Path(source)


def _parse_delimited_header(header_line: str, options: HealthCheckReadOptions, extension: str) -> tuple[str, ...]:
    delimiter = options.delimiter if options.delimiter is not None else "\t" if extension in {".tsv", ".txt"} else ","
    if options.quote_char is None:
        return tuple(next(csv.reader([header_line], delimiter=delimiter, quoting=csv.QUOTE_NONE)))
    return tuple(next(csv.reader([header_line], delimiter=delimiter, quotechar=options.quote_char)))


def _preview_delimited_header(
    source: HealthCheckSource,
    options: HealthCheckReadOptions,
    extension: str,
) -> tuple[str, ...] | None:
    if extension not in {".csv", ".tsv", ".txt"} or not options.has_header:
        return None

    metadata = _source_metadata(source)
    if metadata.extension in {".gz", ".zip"}:
        return None

    if isinstance(source, BytesIO):
        original_position = source.tell()
        try:
            source.seek(0)
            for _ in range(options.skip_rows):
                source.readline()
            header = source.readline().decode(options.encoding, errors="replace")
        finally:
            source.seek(original_position)
    else:
        with Path(source).open(encoding=options.encoding, errors="replace") as file:
            for _ in range(options.skip_rows):
                file.readline()
            header = file.readline()

    return _parse_delimited_header(header, options, extension) if header else None


def _import_source(
    source: HealthCheckSource,
    options: SourceImportOptions | None,
    *,
    full_schema_inference: bool = False,
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    options = options or SourceImportOptions()
    extension = _source_read_extension(source)
    raw = read_data(
        _readable_source(source),
        read_options=_polars_read_options(
            options.read,
            extension,
            full_schema_inference=full_schema_inference,
        ),
    )
    return normalize_health_check_data(raw, options.normalize)


def _should_extract_pyname_keys(df: pl.LazyFrame, extract_pyname_keys: bool) -> bool:
    if not extract_pyname_keys:
        return False
    schema_names = set(cdh_utils._polars_capitalize(df).collect_schema().names())
    return "Name" in schema_names and not _MODEL_CONTEXT_COLUMNS.issubset(schema_names)


def preview_health_check_columns(
    source: HealthCheckSource,
    options: HealthCheckReadOptions | None = None,
) -> tuple[str, ...]:
    """Return column names read from a Health Check source.

    Parameters
    ----------
    source : str, path-like, or BytesIO
        Input source to inspect.
    options : HealthCheckReadOptions, optional
        Reader options to apply while inspecting the source.

    Returns
    -------
    tuple[str, ...]
        Column names found in the source.
    """
    options = options or HealthCheckReadOptions()
    extension = _source_read_extension(source)
    if (columns := _preview_delimited_header(source, options, extension)) is not None:
        return columns
    return tuple(
        read_data(
            _readable_source(source),
            read_options=_polars_read_options(options, extension),
        )
        .collect_schema()
        .names()
    )


def import_health_check_data(
    model_source: HealthCheckSource,
    predictor_source: HealthCheckSource | None = None,
    prediction_source: HealthCheckSource | None = None,
    *,
    model_options: SourceImportOptions | None = None,
    predictor_options: SourceImportOptions | None = None,
    prediction_options: SourceImportOptions | None = None,
    extract_pyname_keys: bool = True,
    predictor_categorization: Mapping[str, str | list[str]] | None = None,
    predictor_categorization_uses_regex: bool = False,
) -> HealthCheckImportResult:
    """Import model, optional predictor, and optional prediction data.

    The returned source frames are normalized but remain constructor-ready.
    This allows canonical parquet files to be loaded later by the normal
    :class:`ADMDatamart` and :class:`Prediction` validation paths.

    Parameters
    ----------
    model_source : HealthCheckSource
        Required model snapshot input.
    predictor_source : HealthCheckSource, optional
        Predictor-binning snapshot input.
    prediction_source : HealthCheckSource, optional
        Prediction snapshot input.
    model_options, predictor_options, prediction_options : SourceImportOptions, optional
        Independent read and normalization options for each source.
    extract_pyname_keys : bool, default True
        Whether to extract context keys embedded in the model Name field.
    predictor_categorization : Mapping[str, str or list[str]], optional
        Category names mapped to PredictorName substring patterns.
    predictor_categorization_uses_regex : bool, default False
        Interpret predictor categorization patterns as regular expressions.

    Returns
    -------
    HealthCheckImportResult
        Validated analysis objects, constructor-ready source data, metadata,
        and data-safe descriptions of applied repairs.
    """
    if model_source is None:
        raise ValueError("model_source is required")

    model_metadata = _source_metadata(model_source)
    logger.info("Importing Health Check model source '%s'.", model_metadata.name)
    model_data, model_warnings = _import_source(model_source, model_options)
    predictor_data: pl.LazyFrame | None = None
    prediction_data: pl.LazyFrame | None = None
    warnings = [f"Model: {warning}" for warning in model_warnings]
    sources: dict[str, HealthCheckSourceMetadata] = {"model": model_metadata}

    if predictor_source is not None:
        predictor_metadata = _source_metadata(predictor_source)
        logger.info("Importing Health Check predictor source '%s'.", predictor_metadata.name)
        predictor_data, predictor_warnings = _import_source(predictor_source, predictor_options)
        warnings.extend(f"Predictor: {warning}" for warning in predictor_warnings)
        sources["predictor"] = predictor_metadata

    if prediction_source is not None:
        prediction_metadata = _source_metadata(prediction_source)
        logger.info("Importing Health Check prediction source '%s'.", prediction_metadata.name)
        prediction_data, prediction_warnings = _import_source(
            prediction_source,
            prediction_options,
            full_schema_inference=True,
        )
        warnings.extend(f"Prediction: {warning}" for warning in prediction_warnings)
        sources["prediction"] = prediction_metadata

    logger.info("Validating Health Check datamart.")
    datamart_extract_pyname_keys = _should_extract_pyname_keys(model_data, extract_pyname_keys)
    if extract_pyname_keys and not datamart_extract_pyname_keys:
        logger.info("Skipping pyName key extraction; model context columns are already present.")
    datamart = ADMDatamart(
        model_df=model_data,
        predictor_df=predictor_data,
        extract_pyname_keys=datamart_extract_pyname_keys,
    )

    if predictor_data is not None and predictor_categorization:
        logger.info("Applying Health Check predictor categorization.")
        predictor_data = cdh_utils._polars_capitalize(predictor_data)
        predictor_data = datamart.apply_predictor_categorization(
            categorization=dict(predictor_categorization),
            use_regexp=predictor_categorization_uses_regex,
            df=predictor_data,
        )
        datamart = ADMDatamart(
            model_df=model_data,
            predictor_df=predictor_data,
            extract_pyname_keys=datamart_extract_pyname_keys,
        )

    prediction = Prediction(prediction_data) if prediction_data is not None else None
    logger.info("Health Check import complete.")

    return HealthCheckImportResult(
        datamart=datamart,
        prediction=prediction,
        model_data=model_data,
        predictor_data=predictor_data,
        prediction_data=prediction_data,
        sources=sources,
        warnings=tuple(warnings),
    )


def resolve_health_check_output_dir(
    model_source: HealthCheckSource | None = None,
    predictor_source: HealthCheckSource | None = None,
    prediction_source: HealthCheckSource | None = None,
    *,
    output_parent: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the ``HC`` output directory for processed parquet files.

    An explicit output parent takes precedence. Otherwise, the first filesystem
    source is selected in model, predictor, prediction order. In-memory-only
    imports fall back to the current working directory.

    Parameters
    ----------
    model_source, predictor_source, prediction_source : HealthCheckSource, optional
        Input sources used to select a filesystem parent.
    output_parent : str or path-like, optional
        Explicit directory under which the ``HC`` directory is created.

    Returns
    -------
    pathlib.Path
        Resolved output directory ending in ``HC``.
    """
    if output_parent is not None:
        return Path(output_parent).expanduser().resolve() / "HC"

    for source in (model_source, predictor_source, prediction_source):
        if source is not None and not isinstance(source, BytesIO):
            return Path(source).expanduser().resolve().parent / "HC"

    return Path.cwd().resolve() / "HC"


def save_health_check_parquet(
    result: HealthCheckImportResult,
    output_parent: str | os.PathLike[str],
) -> dict[str, Path]:
    """Atomically persist canonical Health Check cache parquet files.

    All requested frames are written to temporary files before any canonical
    file is replaced. Optional canonical files from an earlier import are
    removed after a successful write when the corresponding source is absent.

    Parameters
    ----------
    result : HealthCheckImportResult
        Imported source frames and validated analysis objects.
    output_parent : str or path-like
        Parent directory under which an ``HC`` directory is created.

    Returns
    -------
    dict[str, pathlib.Path]
        Canonical paths keyed by ``model``, ``predictor``, and ``prediction``.
    """
    output_dir = resolve_health_check_output_dir(output_parent=output_parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_data = result.datamart.model_data
    if model_data is None:
        raise ValueError("Health Check import result does not include model data")

    frames: dict[str, tuple[pl.LazyFrame, str]] = {
        "model": (model_data, MODEL_CACHE_FILENAME),
    }
    if result.datamart.predictor_data is not None:
        frames["predictor"] = (result.datamart.predictor_data, PREDICTOR_CACHE_FILENAME)
    if result.prediction_data is not None:
        frames["prediction"] = (result.prediction_data, PREDICTION_CACHE_FILENAME)

    temporary_paths: dict[str, Path] = {}
    canonical_paths = {role: output_dir / filename for role, (_, filename) in frames.items()}
    try:
        for role, (frame, filename) in frames.items():
            temporary = output_dir / f".{filename}.{uuid.uuid4().hex}.tmp.parquet"
            temporary_paths[role] = temporary
            frame.sink_parquet(temporary)

        for role, temporary in temporary_paths.items():
            os.replace(temporary, canonical_paths[role])

        optional_paths = {
            "predictor": output_dir / PREDICTOR_CACHE_FILENAME,
            "prediction": output_dir / PREDICTION_CACHE_FILENAME,
        }
        for role, path in optional_paths.items():
            if role not in frames:
                path.unlink(missing_ok=True)
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)

    return canonical_paths
