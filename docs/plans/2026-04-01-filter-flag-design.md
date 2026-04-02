# Decision Analyzer: `--filter` Flag Design

## Problem

Single decision analysis needs a way to select specific interactions or users
from potentially very large files (gigabytes). Currently the only pre-ingestion
option is `--sample`, which selects a random subset. Users who know specific IDs
(from a support ticket, log, etc.) need a way to extract just those rows.

## Design

### CLI Interface

New `--filter` flag, repeatable, mutually exclusive with `--sample`:

```
pdstools da --data-path big_export.zip --filter "Interaction ID=ABC-123,DEF-456"
pdstools da --data-path big_export.zip --filter "Subject ID=customer-42"
pdstools da --data-path big_export.zip --filter "Interaction ID=ABC" --filter "Channel=Web"
```

- **Syntax:** `"Column Name=value1,value2,..."` -- exact-match on any listed value.
- **Multiple filters:** Combined with AND.
- **Column names:** User-friendly display names from `column_schema.py`
  (e.g. "Interaction ID", "Subject ID", "Channel"). Resolution is case-insensitive.
- **Filter values:** Case-sensitive exact match.
- **Env var:** `PDSTOOLS_FILTER` -- JSON-encoded list of specs,
  e.g. `'["Interaction ID=ABC,DEF", "Channel=Web"]'`.
- **Combinable with --sample:** Filter is applied first, then sampling runs on the
  filtered result.

### Filter Parsing & Resolution

New `parse_filter_specs()` function in `decision_analyzer/utils.py`:

```python
def parse_filter_specs(
    filter_specs: list[str],
    available_columns: set[str],
    table_definition: dict | None = None,
) -> pl.Expr:
```

1. Parses each spec `"Column Name=val1,val2"` into (column_name, [values]).
2. Resolves the display name to the actual raw column present in the data,
   using the `display_name` and `aliases` mappings from both
   `DecisionAnalyzer` and `ExplainabilityExtract` schemas.
   Resolution order: display names, then aliases, then raw keys as fallback.
3. Builds `pl.col(resolved_name).is_in([values])` per spec.
4. ANDs all expressions together.
5. Returns the combined Polars expression.

**Extensibility:** The parser splits on the first `=`. Future operators
(`>=`, `!=`, etc.) can be detected before the `=` fallback without breaking
existing syntax.

### Pipeline Integration

**CLI layer (`cli.py`):**
- `--filter` argument with `action="append"`.
- Serialize to `PDSTOOLS_FILTER` env var as JSON list.
- Can be combined with `--sample` (filter applied first).

**App layer (`Home.py`):**
- Read `PDSTOOLS_FILTER` env var (only for `--data-path`, not file uploads).
- Call `parse_filter_specs()` with the raw LazyFrame's columns and schema.
- Apply `raw_data = raw_data.filter(expr)` -- Polars pushes predicates down
  into parquet row-group scanning.
- Pass through `prepare_and_save()` in caching mode to persist as
  `decision_analyzer_filter_<hash>.parquet`.
- On subsequent runs with the same filters + source, detect the cached file
  and skip re-processing.
- Show info banner describing the applied filters.

**Parquet metadata:**
- `pdstools:source_file` -- original source path.
- `pdstools:filter_specs` -- JSON-encoded list of filter specs applied.

**Programmatic API:** `parse_filter_specs()` is standalone and can be used
directly without the Streamlit app.

### Error Handling

- **Unknown column:** Clear error listing all filterable columns by display name.
- **No matching rows:** Warning, no parquet written.
- **Malformed spec:** `ValueError` with syntax hint:
  `"Expected format: 'Column Name=value1,value2'"`.
- **Commas in values:** Not supported (Pega IDs don't contain commas).
- **Column name resolution:** Case-insensitive. Filter values: case-sensitive.
