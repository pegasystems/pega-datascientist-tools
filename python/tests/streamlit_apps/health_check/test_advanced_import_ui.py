from __future__ import annotations

import json
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING

from streamlit.testing.v1 import AppTest

from pdstools.adm.HealthCheckImport import MODEL_CACHE_FILENAME

if TYPE_CHECKING:
    from pathlib import Path


def _write_semicolon_model(path: Path) -> None:
    path.write_text(
        "pyModelID;pyConfigurationName;pySnapshotTime;pyPositives;pyNegatives;"
        "pyResponseCount;pyPerformance;pyChannel;pyDirection;pyIssue;pyGroup;pyName\n"
        "model-1;Config;20260716T103000.000 GMT;10;90;100;0.7;Web;Inbound;Issue;Group;Action\n"
    )


def _write_single_quoted_model_with_bad_timestamp(path: Path) -> None:
    path.write_text(
        "pyModelID;pyConfigurationName;pySnapshotTime;pyPositives;pyNegatives;"
        "pyResponseCount;pyPerformance;pyChannel;pyDirection;pyIssue;pyGroup;pyName\n"
        "model-1;Config;not-a-date;10;90;100;0.7;Web;Inbound;Issue;Group;'Action;Quoted'\n"
    )


def _write_model_missing_repairable_required_fields(path: Path) -> None:
    path.write_text(
        "pyModelID,pyConfigurationName,pySnapshotTime,pyPositives,"
        "pyResponseCount,pyPerformance,pyIssue,pyGroup,pyName\n"
        "model-1,Config,20260716T103000.000 GMT,10,100,0.7,Issue,Group,Action\n"
    )


def _write_predictor_with_external_model_name(path: Path) -> None:
    path.write_text(
        "pyModelID,pyPredictorName,pyPerformance,pyBinIndex,pyBinPositives,"
        "pyBinNegatives,pyBinResponseCount,pySnapshotTime,pyEntryType\n"
        "model-1,Customer.Score,0.62,1,10,90,100,20260716T103000.000 GMT,Active\n"
        "model-1,Classifier,0.70,1,30,70,100,20260716T103000.000 GMT,Classifier\n"
    )


def _write_prediction_zip(path: Path) -> None:
    rows = [
        {
            "pyModelType": "PREDICTION",
            "pySnapShotTime": "20260716T103000.000 GMT",
            "pyModelId": "DATA-DECISION-REQUEST!PREDICTWEBPROPENSITY",
            "pySnapshotType": "Daily",
            "pyDataUsage": data_usage,
            "pyPositives": positives,
            "pyNegatives": negatives,
            "pyCount": positives + negatives,
            "pyValue": 0.71,
        }
        for data_usage, positives, negatives in (
            ("Control", 10.0, 90.0),
            ("Test", 20.0, 80.0),
            ("NBA", 30.0, 70.0),
        )
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.json", "\n".join(json.dumps(row) for row in rows))


def test_advanced_delimiter_and_cache_opt_in(hc_app_dir: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.csv"
    _write_semicolon_model(model_path)

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60).run()
    assert not at.exception

    model_input = next(widget for widget in at.text_input if widget.label == "Model Snapshot path")
    model_input.set_value(str(model_path)).run()

    text_area_labels = {widget.label for widget in at.text_area}
    assert "Model column renames" not in text_area_labels
    assert "Model row filters" not in text_area_labels
    assert "Model text replacements" not in text_area_labels
    assert "Reader type overrides" not in text_area_labels
    assert "Non-strict type overrides" not in text_area_labels
    assert "Fill null values" not in text_area_labels
    assert "Derived columns" not in text_area_labels
    assert "Constant columns" not in text_area_labels
    assert not any(widget.label == "Columns to drop" for widget in at.text_input)
    assert any("Settings for the Model Snapshot Data" in markdown.value for markdown in at.markdown)

    delimiter_input = next(widget for widget in at.text_input if widget.key == "_hc_model_delimiter")
    delimiter_input.set_value(";").run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    assert not at.exception
    assert "dm" in at.session_state
    assert at.session_state["dm"].model_data.select("ModelID").collect().item() == "model-1"
    assert not (tmp_path / "HC").exists(), "Processed parquet must be opt-in"

    keep_parquet = next(checkbox for checkbox in at.checkbox if checkbox.label == "Keep processed parquet files")
    keep_parquet.set_value(True).run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    canonical_model = tmp_path / "HC" / MODEL_CACHE_FILENAME
    assert not at.exception
    assert canonical_model.is_file()
    assert at.session_state["_hc_output_dir"] == str(tmp_path / "HC")
    assert at.session_state["_hc_written_paths"] == (str(canonical_model),)


def test_upload_cache_uses_configured_output_folder(hc_app_dir: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.csv"
    _write_semicolon_model(model_path)
    output_parent = tmp_path / "processed"

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60).run()
    assert not at.exception

    model_uploader = next(uploader for uploader in at.get("file_uploader") if "Model Snapshot" in uploader.label)
    model_uploader.upload(model_path.name, model_path.read_bytes(), "text/csv")
    at.run()

    delimiter_input = next(widget for widget in at.text_input if widget.key == "_hc_model_delimiter")
    delimiter_input.set_value(";").run()

    keep_parquet = next(checkbox for checkbox in at.checkbox if checkbox.label == "Keep processed parquet files")
    keep_parquet.set_value(True).run()

    output_input = next(widget for widget in at.text_input if widget.label == "Output folder")
    output_input.set_value(str(output_parent)).run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    canonical_model = output_parent / "HC" / MODEL_CACHE_FILENAME
    assert not at.exception
    assert canonical_model.is_file()
    assert not (tmp_path / "HC").exists()
    assert at.session_state["_hc_output_dir"] == str(output_parent / "HC")
    assert at.session_state["_hc_written_paths"] == (str(canonical_model),)


def test_single_quoted_model_path_imports(hc_app_dir: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.csv"
    _write_model_missing_repairable_required_fields(model_path)

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60).run()
    assert not at.exception

    model_input = next(widget for widget in at.text_input if widget.label == "Model Snapshot path")
    model_input.set_value(f"'{model_path}'").run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    assert not at.exception
    assert "dm" in at.session_state
    model_data = at.session_state["dm"].model_data.select("ModelID").collect()
    assert model_data["ModelID"].to_list() == ["model-1"]


def test_custom_quote_character_and_timestamp_fallback(hc_app_dir: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.csv"
    _write_single_quoted_model_with_bad_timestamp(model_path)

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60).run()
    assert not at.exception

    model_input = next(widget for widget in at.text_input if widget.label == "Model Snapshot path")
    model_input.set_value(str(model_path)).run()

    delimiter_input = next(widget for widget in at.text_input if widget.key == "_hc_model_delimiter")
    delimiter_input.set_value(";").run()

    quote_input = next(widget for widget in at.text_input if widget.key == "_hc_model_quote_char")
    quote_input.set_value("'").run()

    timestamp_column = next(widget for widget in at.text_input if widget.key == "_hc_model_timestamp_column")
    timestamp_column.set_value("pySnapshotTime").run()

    timestamp_format = next(widget for widget in at.text_input if widget.key == "_hc_model_timestamp_format")
    timestamp_format.set_value("%Y-%m-%d %H:%M:%S").run()

    timestamp_fallback = next(widget for widget in at.text_input if widget.key == "_hc_model_timestamp_fallback")
    timestamp_fallback.set_value("2026-01-01T00:00:00").run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    assert not at.exception
    assert "dm" in at.session_state
    model_data = at.session_state["dm"].model_data.select("Name", "SnapshotTime").collect()
    assert model_data.to_dict(as_series=False) == {
        "Name": ["Action;Quoted"],
        "SnapshotTime": [datetime(2026, 1, 1)],
    }


def test_missing_required_field_repairs_are_auto_populated(hc_app_dir: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.csv"
    _write_model_missing_repairable_required_fields(model_path)

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60).run()
    assert not at.exception

    model_input = next(widget for widget in at.text_input if widget.label == "Model Snapshot path")
    model_input.set_value(str(model_path)).run()

    import_tips = " ".join(info.value for info in at.info)
    assert "process AI exports" in import_tips
    assert "Channel, Direction, Issue, or Group" in import_tips

    field_repairs = next(widget for widget in at.text_area if widget.key == "_hc_model_field_repairs")
    assert field_repairs.value.splitlines() == [
        "pyNegatives=pyResponseCount-pyPositives",
        "pyChannel=NA",
        "pyDirection=NA",
    ]

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    assert not at.exception
    model_data = (
        at.session_state["dm"]
        .model_data.select(
            "Negatives",
            "Channel",
            "Direction",
        )
        .collect()
    )
    assert model_data.to_dict(as_series=False) == {
        "Negatives": [90.0],
        "Channel": ["NA"],
        "Direction": ["NA"],
    }


def test_predictor_categorization_defaults_are_real_values(hc_app_dir: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.csv"
    predictor_path = tmp_path / "predictor.csv"
    _write_semicolon_model(model_path)
    _write_predictor_with_external_model_name(predictor_path)

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60).run()
    assert not at.exception

    model_input = next(widget for widget in at.text_input if widget.label == "Model Snapshot path")
    model_input.set_value(str(model_path)).run()
    predictor_input = next(
        widget for widget in at.text_input if widget.label == "Predictor Binning snapshot path (optional)"
    )
    predictor_input.set_value(str(predictor_path)).run()

    category_mappings = next(widget for widget in at.text_area if widget.key == "_hc_predictor_categories")
    assert category_mappings.value.startswith("External Model=")
    assert "Propensity" in category_mappings.value
    assert "Score" in category_mappings.value
    assert "Class" in category_mappings.value

    delimiter_input = next(widget for widget in at.text_input if widget.key == "_hc_model_delimiter")
    delimiter_input.set_value(";").run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    assert not at.exception
    predictor_data = (
        at.session_state["dm"]
        .predictor_data.select("PredictorName", "PredictorCategory")
        .collect()
        .sort("PredictorName")
    )
    assert predictor_data.to_dict(as_series=False) == {
        "PredictorName": ["Classifier", "Customer.Score"],
        "PredictorCategory": [None, "External Model"],
    }


def test_prediction_import_settings_hide_schema_inference(hc_app_dir: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.csv"
    prediction_path = tmp_path / "prediction.zip"
    _write_semicolon_model(model_path)
    _write_prediction_zip(prediction_path)

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60).run()
    assert not at.exception

    model_input = next(widget for widget in at.text_input if widget.label == "Model Snapshot path")
    model_input.set_value(str(model_path)).run()
    prediction_input = next(widget for widget in at.text_input if widget.label == "Prediction Table path (optional)")
    prediction_input.set_value(str(prediction_path)).run()

    assert any(widget.key == "_hc_model_infer_schema_length" for widget in at.number_input)
    assert not any(widget.key == "_hc_prediction_infer_schema_length" for widget in at.number_input)
