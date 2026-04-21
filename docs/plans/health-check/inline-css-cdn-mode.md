# Inline CSS in CDN mode reports

**Priority:** P2
**Touches:** `python/pdstools/adm/Reports.py`, `python/pdstools/adm/report_utils.py`

Related: [#620](https://github.com/pegasystems/pega-datascientist-tools/issues/620)

With `embed-resources: false` (CDN mode), Quarto leaves CSS as external `<link>` tags. When `Reports.py` copies only the HTML to the output directory, these CSS references break. Result: plots render fine (Plotly loads from CDN) but text styling is completely lost — no Bootstrap theme, no Pega overrides.

**Root cause:** `run_quarto` renders into `temp_dir`. Quarto creates a support directory (e.g. `HealthCheck_files/libs/`) with CSS files. The HTML references them via relative `<link>` tags. Copying the HTML without the support directory breaks these links.

**Five CSS files affected (from actual CDN-mode output):**
1. `HealthCheck_files/libs/bootstrap/bootstrap-*.min.css` — flatly theme
2. `HealthCheck_files/libs/bootstrap/bootstrap-icons.css` — icon font
3. `HealthCheck_files/libs/quarto-html/quarto-syntax-highlighting-*.css`
4. `HealthCheck_files/libs/quarto-html/tippy.css` — tooltip styling
5. `assets/pega-report-overrides.css` — Pega logo, typography, colors

## Approach

Add `_inline_css(html_path, base_dir)` in `run_quarto` after Quarto completes (only when `full_embed=False` and `output_type="html"`). Use a regex to find `<link rel="stylesheet" href="...">` tags, resolve relative paths against `temp_dir`, read the CSS content, and replace each `<link>` with an inline `<style>` block. Skip absolute URLs (CDN resources). Total CSS is ~100 KB — negligible compared to the ~7 MB HTML.

**Why not just set `embed-resources: true`?** That triggers Quarto's esbuild pipeline to bundle JavaScript. DJS Docker images have esbuild removed due to CVE issues (see #620).
