# In-memory ADM ZIP upload fails schema inference

**Priority:** P2
**Files touched:** `python/pdstools/adm/ADMDatamart.py`,
`python/pdstools/pega_io/File.py` (whichever owns the in-memory ZIP
ingest path)

## Problem

Loading a bundled ADM Datamart sample via ``ADMDatamart.from_ds_export``
works fine when given a filesystem path, but blows up with a polars
``ComputeError`` when the same bytes are passed as in-memory file-like
objects (e.g. Streamlit's ``UploadedFile`` or a ``BytesIO``):

```
ComputeError: cannot parse '31.542568' (f64) as Int64
ComputeError: got non-null value for NULL-typed column: 144.115952
```

Reproduces against both bundled samples shipped in ``data/``:

- ``Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip``
- ``Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip``

Same bytes load fine when written to disk first and passed by path.
This means the Health Check **Direct file upload** flow is broken for
any export that triggers schema inference on a partial sample
(default ``infer_schema_length`` is 10 000, which works for short rows
on disk but apparently scans differently for in-memory readers).

## Impact

The Streamlit "Direct file upload" branch in HC silently produces
``dm = None`` on a real export — the user sees no error banner because
``cached_datamart`` swallows the exception into ``st.error`` text but
the page already moved on. This is exactly the "uploader works"
half of the regression covered by the new tests
(``test_direct_upload_populates_dm.py``), which is why that test is
currently ``pytest.xfail``ed pointing at this plan file.

## Proposed approach

1. Reproduce in a unit test against ``ADMDatamart.from_ds_export`` with
   a ``BytesIO`` + ``io.BufferedReader`` wrapped over the bundled zip.
2. Likely fix is to bump the default ``infer_schema_length`` for
   in-memory readers, or pass an explicit schema for the model /
   predictor frames (we know the canonical column types).
3. Once fixed, drop the ``xfail`` marker on
   ``test_direct_upload_populates_dm`` and ``git rm`` this plan file
   in the same PR.
