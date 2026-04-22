# DA garbage upload crashes the Home page

**Priority:** P2
**Files touched:** `python/pdstools/app/decision_analyzer/_home_page.py`

## Problem

When the user drops a file into the Decision Analysis Home uploader
that has a whitelisted extension (``.parquet``, ``.csv``, ``.json``,
``.zip``, etc.) but contains garbage / corrupt bytes, the page
crashes with an unhandled ``polars.exceptions.ComputeError`` instead
of surfacing a clear "this file is not a valid Decision Analyzer
export" message.

Reproduces with:

```python
uploaders[0].upload("nonsense.parquet", b"PAR-not-actually-parquet\\x00\\x00", "application/octet-stream")
```

The exception fires from
``raw_data.explain(optimized=False)`` at ``_home_page.py:302``
(``parquet: File out of specification: The file must end with PAR1``),
which is *outside* the ``try/except`` block that catches
``ValueError`` / capacity errors / missing-column errors at lines
321–342.

## Impact

A user who fat-fingers an upload (or drops a non-Pega file with a
plausible extension) sees the Streamlit "Uncaught app exception" red
banner and a Python stack trace, with no recovery path. The
autoloaded ``decision_data`` is also gone at this point — clobbered
by the ``handle_file_upload`` dedup-and-clear path before the parser
threw.

## Proposed approach

1. Move the ``raw_data.explain(...)`` and the ``data_id``
   computation inside the existing ``try/except BaseException`` block
   (or a dedicated outer try) so polars ``ComputeError`` is caught.
2. Add ``ComputeError`` to the list of "looks like a parser
   problem" exception classes that surface the
   "Cannot parse this file" ``st.error`` message.
3. Once fixed, drop the ``xfail`` marker on
   ``test_da_garbage_parquet_upload_handled_gracefully`` in
   ``python/tests/streamlit_apps/decision_analyzer/test_da_wrong_format_upload.py``
   and ``git rm`` this plan file.
