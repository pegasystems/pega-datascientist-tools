"""Widget-interaction test: "Generate Health Check" button must trigger generation.

``2_Reports.py`` shows a "Generate Health Check" button.  When clicked,
the page calls ``st.session_state["dm"].generate.health_check(...)`` and
stores the result in session state, then renders a "Download Health Check"
download button.  The actual report generation requires Quarto (tested
separately in ``test_Reports.py``); this test monkeypatches
``health_check`` to return a pre-created temp HTML file and asserts that
the click → store → download-button flow completes without error.

If the button's session-state bookkeeping is broken — e.g. ``runID`` is
not incremented, the run dict is not updated, or the download button is
conditionally hidden — the button click silently does nothing and the
download button never appears.  Existing smoke tests stay green because
the page renders before the button is clicked.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def test_generate_button_shows_download_button(
    hc_app_dir,
    seeded_admdatamart,
    tmp_path,
) -> None:
    """Clicking 'Generate Health Check' renders a download button.

    Steps:
    1. Patch ``dm.generate.health_check`` to return a pre-created temp file.
    2. Seed ``session_state["dm"]`` with the patched dm.
    3. Run the page — button present, download button absent.
    4. Click the button and re-run — download button must appear and
       ``session_state["run"]`` must record the generated file.
    """
    mock_output = tmp_path / "mock_healthcheck.html"
    mock_output.write_text("<html><body>Mock Health Check</body></html>")
    captured_kwargs = {}

    # Patch health_check on the generate namespace instance.
    # AppTest runs in the same process, so instance-attribute shadowing works.
    def fake_health_check(**kwargs):
        captured_kwargs.update(kwargs)
        return str(mock_output)

    seeded_admdatamart.generate.health_check = fake_health_check

    page = hc_app_dir / "pages" / "2_Reports.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.session_state["_hc_output_dir"] = str(tmp_path / "HC")
    at.run()
    assert not at.exception, f"Page raised on initial run: {at.exception}"

    # Before clicking, no "Download Health Check" download button.
    assert not any("Health Check" in getattr(b, "label", "") for b in at.get("download_button")), (
        "Download button should not appear before generation is triggered."
    )

    # Locate the generate button.
    gen_button = next(
        (b for b in at.button if b.label == "Generate Health Check"),
        None,
    )
    assert gen_button is not None, "Expected a 'Generate Health Check' button on the page."

    gen_button.click().run()
    assert not at.exception, f"Page raised after button click: {at.exception}"

    # The download button must now be present.
    assert any("Health Check" in getattr(b, "label", "") for b in at.get("download_button")), (
        "Expected a 'Download Health Check' download button after generation; "
        f"got: {[getattr(b, 'label', '?') for b in at.get('download_button')]}"
    )

    # Session state must record the completed run.
    assert "run" in at.session_state
    run_id = at.session_state["runID"]
    assert run_id > 0, f"runID should have incremented from 0, got {run_id}"
    assert "file" in at.session_state["run"].get(run_id, {}), (
        f"session_state['run'][{run_id}] should contain a 'file' key after generation."
    )
    assert captured_kwargs["output_dir"] == tmp_path / "HC"


def test_generate_button_shows_generated_health_check_path(
    hc_app_dir,
    seeded_admdatamart,
    tmp_path,
) -> None:
    mock_output = tmp_path / "mock_healthcheck.html"
    mock_output.write_text("<html><body>Mock Health Check</body></html>")
    seeded_admdatamart.generate.health_check = lambda **kwargs: str(mock_output)

    written_path = tmp_path / "HC" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    page = hc_app_dir / "pages" / "2_Reports.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.session_state["_hc_output_dir"] = str(tmp_path / "HC")
    at.session_state["_hc_written_paths"] = (str(written_path),)
    at.run()
    assert not at.exception

    gen_button = next(button for button in at.button if button.label == "Generate Health Check")
    gen_button.click().run()

    assert not at.exception
    info_text = "\n".join(info.value for info in at.info)
    assert f"Generated health check file: {mock_output}" in info_text
    assert "Processed parquet destination" not in info_text
    assert str(written_path) not in info_text


def test_create_tables_shows_generated_excel_path(
    hc_app_dir,
    seeded_admdatamart,
    tmp_path,
) -> None:
    table_output = tmp_path / "HealthCheckExport.xlsx"
    table_output.write_bytes(b"mock xlsx")
    seeded_admdatamart.generate.excel_report = lambda *args, **kwargs: (table_output, [])

    written_path = tmp_path / "HC" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    page = hc_app_dir / "pages" / "2_Reports.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.session_state["_hc_output_dir"] = str(tmp_path / "HC")
    at.session_state["_hc_written_paths"] = (str(written_path),)
    at.run()
    assert not at.exception

    create_tables_button = next(button for button in at.button if button.label == "Create Tables")
    create_tables_button.click().run()

    assert not at.exception
    info_text = "\n".join(info.value for info in at.info)
    assert f"Generated Excel tables file: {tmp_path / 'HC' / 'HealthCheckExport.xlsx'}" in info_text
    assert "Processed parquet destination" not in info_text
    assert str(written_path) not in info_text


def test_report_full_embed_option_is_in_normal_options(
    hc_app_dir,
    seeded_admdatamart,
) -> None:
    """The full-embed checkbox belongs with normal report options."""
    page = hc_app_dir / "pages" / "2_Reports.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()

    assert not at.exception
    assert "Health Check options" in [expander.label for expander in at.expander]
    assert "Advanced" not in [expander.label for expander in at.expander]
    working_dir_input = next(widget for widget in at.text_input if widget.label == "Change working directory")
    assert working_dir_input.value == "healthCheckDir"
    labels = [checkbox.label for checkbox in at.checkbox]
    assert "Embed JavaScript/CSS into a single document" in labels
    assert "Embed JS/CSS for offline viewing (slower, larger file)" not in labels


def test_report_output_defaults_to_import_hc_folder(
    hc_app_dir,
    seeded_admdatamart,
    tmp_path,
) -> None:
    """Report generation defaults to the HC folder selected during import."""
    page = hc_app_dir / "pages" / "2_Reports.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.session_state["_hc_output_dir"] = str(tmp_path / "HC")
    at.run()

    assert not at.exception
    working_dir_input = next(widget for widget in at.text_input if widget.label == "Change working directory")
    assert working_dir_input.value == str(tmp_path / "HC")


def test_generate_button_increments_run_id_on_successive_clicks(
    hc_app_dir,
    seeded_admdatamart,
    tmp_path,
) -> None:
    """Each button click increments ``runID`` so runs are independently tracked."""
    mock_output = tmp_path / "mock_hc.html"
    mock_output.write_text("<html></html>")
    seeded_admdatamart.generate.health_check = lambda **kwargs: str(mock_output)

    page = hc_app_dir / "pages" / "2_Reports.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception

    first_click_button = next((b for b in at.button if b.label == "Generate Health Check"), None)
    assert first_click_button is not None
    first_click_button.click().run()
    assert not at.exception
    assert at.session_state["runID"] == 1

    second_click_button = next((b for b in at.button if b.label == "Generate Health Check"), None)
    assert second_click_button is not None
    second_click_button.click().run()
    assert not at.exception
    assert at.session_state["runID"] == 2, f"Second click should set runID to 2, got {at.session_state['runID']}"
    assert 2 in at.session_state["run"], "session_state['run'] should contain an entry for the second run."
