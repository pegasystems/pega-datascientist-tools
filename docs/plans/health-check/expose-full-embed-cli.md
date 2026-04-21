# Expose `full_embed` as CLI option and UI advanced setting

**Priority:** P3
**Touches:** `python/pdstools/cli.py`, `python/pdstools/app/health_check/`, `python/pdstools/adm/Reports.py`

The `full_embed` flag (default `False`) controls whether JavaScript libraries are bundled into the HTML or loaded from CDN. Users in air-gapped environments need standalone output. Currently the Streamlit app hardcodes `full_embed=True`.

## Approach

- Add `--full-embed` flag to the CLI report command.
- Add an advanced toggle in the Streamlit Reports page so users can opt into standalone output.
- Document the trade-off (standalone = larger file / no esbuild CVE risk vs. CDN = smaller file / requires internet).
