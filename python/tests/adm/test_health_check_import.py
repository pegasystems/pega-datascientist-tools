from datetime import datetime
import gzip
from io import BytesIO
import json
import zipfile
from unittest.mock import patch

import polars as pl
import pytest

from pdstools.adm.HealthCheckImport import (
    HealthCheckReadOptions,
    HealthCheckRowFilter,
    MODEL_CACHE_FILENAME,
    PREDICTION_CACHE_FILENAME,
    PREDICTOR_CACHE_FILENAME,
    SourceImportOptions,
    SourceNormalizationOptions,
    import_health_check_data,
    normalize_health_check_data,
    preview_health_check_columns,
    resolve_health_check_output_dir,
    save_health_check_parquet,
)


MINIMAL_MODEL_HEADER = (
    "pyModelID,pyConfigurationName,pySnapshotTime,pyPositives,pyNegatives,"
    "pyResponseCount,pyPerformance,pyChannel,pyDirection,pyIssue,pyGroup,pyName\n"
)
MINIMAL_MODEL_ROW = "model-1,Config,20260716T103000.000 GMT,10,90,100,0.7,Web,Inbound,Issue,Group,Action\n"


def _uploaded_csv(contents: str, name: str = "model.csv") -> BytesIO:
    uploaded = BytesIO(contents.encode())
    uploaded.name = name
    return uploaded


def _minimal_model_csv() -> str:
    return MINIMAL_MODEL_HEADER + MINIMAL_MODEL_ROW


def _uploaded_zip_ndjson(records: list[dict], name: str = "prediction.zip") -> BytesIO:
    uploaded = BytesIO()
    with zipfile.ZipFile(uploaded, "w") as archive:
        archive.writestr("data.json", "\n".join(json.dumps(record) for record in records))
    uploaded.seek(0)
    uploaded.name = name
    return uploaded


def _uploaded_model_workbook() -> BytesIO:
    import xlsxwriter

    uploaded = BytesIO()
    workbook = xlsxwriter.Workbook(uploaded)
    worksheet = workbook.add_worksheet("Data")
    worksheet.write_row(0, 0, ["export metadata row"])
    worksheet.write_row(1, 0, MINIMAL_MODEL_HEADER.rstrip("\n").split(","))
    worksheet.write_row(
        2,
        0,
        [
            "model-1",
            "Config",
            "20260716T103000.000 GMT",
            10,
            90,
            100,
            0.7,
            "Web",
            "Inbound",
            "Issue",
            "Group",
            "Action",
        ],
    )
    workbook.close()
    uploaded.seek(0)
    uploaded.name = "model.xlsx"
    return uploaded


def test_normalize_health_check_data_applies_declarative_repairs():
    source = pl.LazyFrame(
        {
            "WHEN": ["16/07/2026 10:30", "invalid"],
            "AMOUNT": ["1.5", ""],
            "DROP_ME": ["large", "large"],
        },
    )
    options = SourceNormalizationOptions(
        drop_columns=("drop_me",),
        type_overrides={"amount": "float64"},
        timestamp_column="when",
        timestamp_format="%d/%m/%Y %H:%M",
        timestamp_fallback=datetime(2026, 1, 1),
        constant_columns={"pyDirection": "Inbound"},
    )

    actual, warnings = normalize_health_check_data(source, options)

    assert actual.collect().to_dict(as_series=False) == {
        "WHEN": [datetime(2026, 7, 16, 10, 30), datetime(2026, 1, 1)],
        "AMOUNT": [1.5, None],
        "pyDirection": ["Inbound", "Inbound"],
    }
    assert warnings == (
        "Applied 1 non-strict type override.",
        "Dropped 1 configured column.",
        "Parsed timestamp column 'WHEN' with a configured fallback.",
        "Added 1 constant column.",
    )


def test_normalize_health_check_data_applies_rmd_style_repairs():
    source = pl.LazyFrame(
        {
            "CONFIG": ["Keep", "Drop_AllCustomers"],
            "RAW_NAME": ['Action ""A""', 'Action ""B""'],
            "RESPONSES": ["100", "200"],
            "POSITIVES": ["10", "20"],
            "segment": [None, "Known"],
        },
    )

    actual, warnings = normalize_health_check_data(
        source,
        SourceNormalizationOptions(
            rename_columns={"config": "pyConfigurationName"},
            row_filters=(
                HealthCheckRowFilter(
                    column="pyConfigurationName",
                    operator="not_ends_with",
                    value="_AllCustomers",
                ),
            ),
            text_replacements={"raw_name": {'""': '"'}},
            type_overrides={"responses": "int64", "positives": "int64"},
            fill_null_values={"segment": "Unknown"},
            derived_columns={"pyNegatives": "RESPONSES-POSITIVES"},
            drop_columns=("POSITIVES",),
            constant_columns={"pyChannel": "NA"},
        ),
    )

    assert actual.collect().to_dict(as_series=False) == {
        "pyConfigurationName": ["Keep"],
        "RAW_NAME": ['Action "A"'],
        "RESPONSES": [100],
        "segment": ["Unknown"],
        "pyNegatives": [90.0],
        "pyChannel": ["NA"],
    }
    assert warnings == (
        "Renamed 1 configured column.",
        "Applied 1 row filter.",
        "Applied 1 text replacement column.",
        "Applied 2 non-strict type overrides.",
        "Filled 1 column null value.",
        "Created 1 derived column.",
        "Dropped 1 configured column.",
        "Added 1 constant column.",
    )


def test_normalize_health_check_data_creates_concat_derived_column():
    source = pl.LazyFrame(
        {
            "Channel": ["Email", "Web"],
            "Treatment": ["Offer", "Banner"],
        },
    )

    actual, warnings = normalize_health_check_data(
        source,
        SourceNormalizationOptions(
            derived_columns={"ChannelTreatment": 'concat(Channel,"_",Treatment)'},
        ),
    )

    assert actual.collect().to_dict(as_series=False) == {
        "Channel": ["Email", "Web"],
        "Treatment": ["Offer", "Banner"],
        "ChannelTreatment": ["Email_Offer", "Web_Banner"],
    }
    assert warnings == ("Created 1 derived column.",)


def test_normalize_health_check_data_creates_column_alias():
    source = pl.LazyFrame({"Original": ["value"]})

    actual, warnings = normalize_health_check_data(
        source,
        SourceNormalizationOptions(derived_columns={"Copied": "Original"}),
    )

    assert actual.collect().to_dict(as_series=False) == {
        "Original": ["value"],
        "Copied": ["value"],
    }
    assert warnings == ("Created 1 derived column.",)


def test_normalize_health_check_data_replaces_existing_derived_column():
    source = pl.LazyFrame(
        {
            "Channel": ["Email", "Web"],
            "Treatment": ["Offer", "Banner"],
        },
    )

    actual, warnings = normalize_health_check_data(
        source,
        SourceNormalizationOptions(
            derived_columns={"Channel": 'concat(Channel,"_",Treatment)'},
        ),
    )

    assert actual.collect().to_dict(as_series=False) == {
        "Channel": ["Email_Offer", "Web_Banner"],
        "Treatment": ["Offer", "Banner"],
    }
    assert warnings == ("Created 1 derived column.",)


def test_normalize_health_check_data_adds_missing_timestamp_from_fallback():
    source = pl.LazyFrame({"ModelID": ["model-1"]})

    actual, warnings = normalize_health_check_data(
        source,
        SourceNormalizationOptions(
            timestamp_column="SnapshotTime",
            timestamp_fallback=datetime(2026, 1, 1),
        ),
    )

    assert actual.collect().to_dict(as_series=False) == {
        "ModelID": ["model-1"],
        "SnapshotTime": [datetime(2026, 1, 1)],
    }
    assert warnings == ("Added missing timestamp column 'SnapshotTime' from its configured fallback.",)


def test_health_check_read_options_validate_invalid_settings():
    invalid_options = [
        {"delimiter": "::"},
        {"quote_char": "''"},
        {"skip_rows": -1},
        {"infer_schema_length": -1},
        {"excel_sheet_name": "Data", "excel_sheet_id": 1},
        {"excel_sheet_id": 0},
        {"excel_header_row": -1},
    ]

    for kwargs in invalid_options:
        with pytest.raises(ValueError):
            HealthCheckReadOptions(**kwargs)


def test_normalize_health_check_data_applies_all_row_filter_operators():
    source = pl.LazyFrame(
        {
            "name": ["Alpha", "Beta", "Gamma", None],
            "segment": ["keep", "drop", None, "keep"],
        },
    )

    cases = [
        (HealthCheckRowFilter("name", "==", "Alpha"), ["Alpha"]),
        (HealthCheckRowFilter("name", "!=", "Alpha"), ["Beta", "Gamma"]),
        (HealthCheckRowFilter("name", "contains", "mm"), ["Gamma"]),
        (HealthCheckRowFilter("name", "not_contains", "a"), [None]),
        (HealthCheckRowFilter("name", "starts_with", "Al"), ["Alpha"]),
        (HealthCheckRowFilter("name", "not_starts_with", "A"), ["Beta", "Gamma", None]),
        (HealthCheckRowFilter("name", "ends_with", "a"), ["Alpha", "Beta", "Gamma"]),
        (HealthCheckRowFilter("name", "not_ends_with", "a"), [None]),
        (HealthCheckRowFilter("segment", "is_null"), ["Gamma"]),
        (HealthCheckRowFilter("segment", "is_not_null"), ["Alpha", "Beta", None]),
    ]

    for row_filter, expected_names in cases:
        actual, _ = normalize_health_check_data(
            source,
            SourceNormalizationOptions(row_filters=(row_filter,)),
        )
        assert actual.collect().get_column("name").to_list() == expected_names


def test_normalize_health_check_data_rejects_invalid_row_filters_and_derived_expressions():
    source = pl.LazyFrame({"name": ["Alpha"], "left": [1], "right": [2]})

    invalid_filters = [
        HealthCheckRowFilter("name", "is_null", "Alpha"),
        HealthCheckRowFilter("name", "contains"),
        HealthCheckRowFilter("name", "unsupported", "Alpha"),
    ]
    for row_filter in invalid_filters:
        with pytest.raises(ValueError):
            normalize_health_check_data(source, SourceNormalizationOptions(row_filters=(row_filter,)))

    invalid_derived = [
        {"bad": "left-"},
        {"missing": "not_present"},
    ]
    for derived_columns in invalid_derived:
        with pytest.raises(ValueError):
            normalize_health_check_data(source, SourceNormalizationOptions(derived_columns=derived_columns))


def test_normalize_health_check_data_rejects_conflicting_and_duplicate_columns():
    with pytest.raises(ValueError, match="differ only by case"):
        normalize_health_check_data(
            pl.LazyFrame({"Name": ["A"], "name": ["B"]}),
            SourceNormalizationOptions(drop_columns=("Name",)),
        )

    with pytest.raises(ValueError, match="already exists"):
        normalize_health_check_data(
            pl.LazyFrame({"old": ["A"], "new": ["B"]}),
            SourceNormalizationOptions(rename_columns={"old": "new"}),
        )

    with pytest.raises(ValueError, match="both dropped and created"):
        normalize_health_check_data(
            pl.LazyFrame({"old": ["A"]}),
            SourceNormalizationOptions(drop_columns=("new",), constant_columns={"new": "B"}),
        )

    with pytest.raises(ValueError, match="Unsupported data type"):
        normalize_health_check_data(
            pl.LazyFrame({"value": ["1"]}),
            SourceNormalizationOptions(type_overrides={"value": "definitely-not-a-type"}),
        )

    with pytest.raises(ValueError, match="timestamp_format is required"):
        normalize_health_check_data(
            pl.LazyFrame({"SnapshotTime": ["20260716T103000.000 GMT"]}),
            SourceNormalizationOptions(timestamp_column="SnapshotTime"),
        )

    with pytest.raises(ValueError, match="no fallback was configured"):
        normalize_health_check_data(
            pl.LazyFrame({"ModelID": ["model-1"]}),
            SourceNormalizationOptions(timestamp_column="SnapshotTime"),
        )

    with pytest.raises(ValueError, match="already exists"):
        normalize_health_check_data(
            pl.LazyFrame({"Channel": ["Web"]}),
            SourceNormalizationOptions(constant_columns={"channel": "Email"}),
        )


def test_normalize_health_check_data_casts_existing_temporal_timestamp():
    source = pl.LazyFrame({"SnapshotTime": [datetime(2026, 7, 16, 10, 30)]})

    actual, warnings = normalize_health_check_data(
        source,
        SourceNormalizationOptions(timestamp_column="SnapshotTime"),
    )

    assert actual.collect().get_column("SnapshotTime").to_list() == [datetime(2026, 7, 16, 10, 30)]
    assert warnings == ("Parsed timestamp column 'SnapshotTime'.",)


def test_normalize_health_check_data_rejects_unknown_drop_column():
    with pytest.raises(ValueError, match="Drop column 'missing' was not found"):
        normalize_health_check_data(
            pl.LazyFrame({"present": [1]}),
            SourceNormalizationOptions(drop_columns=("missing",)),
        )


def test_import_health_check_data_reads_and_validates_nonstandard_csv():
    model = _uploaded_csv(
        "PYMODELID;PYCONFIGURATIONNAME;PYSNAPSHOTTIME;PYPOSITIVES;"
        "PYNEGATIVES;PYRESPONSECOUNT;PYPERFORMANCE;PYCHANNEL;PYDIRECTION;"
        "PYISSUE;PYGROUP;PYNAME\n"
        "model-1;Config;16/07/2026 10:30;10;90;100;70;Web;Inbound;"
        "Issue;Group;Action\n",
    )
    options = SourceImportOptions(
        read=HealthCheckReadOptions(
            delimiter=";",
            infer_schema_length=0,
        ),
        normalize=SourceNormalizationOptions(
            type_overrides={
                "pypositives": "int64",
                "pynegatives": "int64",
                "pyresponsecount": "int64",
                "pyperformance": "float64",
            },
            timestamp_column="pysnapshottime",
            timestamp_format="%d/%m/%Y %H:%M",
        ),
    )

    result = import_health_check_data(
        model,
        model_options=options,
        extract_pyname_keys=False,
    )

    assert result.predictor_data is None
    assert result.prediction is None
    assert result.sources["model"].extension == ".csv"
    model_data = result.datamart.model_data.collect()
    assert model_data.select(
        "ModelID",
        "SnapshotTime",
        "Positives",
        "ResponseCount",
        "Performance",
        "SuccessRate",
    ).to_dict(as_series=False) == {
        "ModelID": ["model-1"],
        "SnapshotTime": [datetime(2026, 7, 16, 10, 30)],
        "Positives": [10],
        "ResponseCount": [100],
        "Performance": [0.7],
        "SuccessRate": [0.1],
    }
    assert result.model_data.collect().get_column("PYSNAPSHOTTIME").dtype == pl.Datetime


def test_import_health_check_data_reads_tab_delimited_path_with_null_values(tmp_path):
    model_path = tmp_path / "quoted-model.txt"
    model_path.write_text(
        "pyModelID\tpyConfigurationName\tpySnapshotTime\tpyPositives\tpyNegatives\t"
        "pyResponseCount\tpyPerformance\tpyChannel\tpyDirection\tpyIssue\tpyGroup\tpyName\n"
        'model-1\tConfig\t20260716T103000.000 GMT\t10\t90\t100\t0.7\tWeb\tInbound\tIssue\tGroup\t"Action\tQuoted"\n',
    )

    result = import_health_check_data(
        model_path,
        model_options=SourceImportOptions(
            read=HealthCheckReadOptions(
                delimiter="\t",
                null_values=("", "NA"),
                schema_overrides={"pyPositives": "float64"},
            ),
        ),
        extract_pyname_keys=False,
    )

    assert result.sources["model"].name == "quoted-model.txt"
    assert result.sources["model"].extension == ".txt"
    assert result.datamart.model_data.select("Name", "Positives").collect().to_dict(as_series=False) == {
        "Name": ["Action\tQuoted"],
        "Positives": [10.0],
    }


def test_import_health_check_data_applies_read_options_to_gzip_inner_csv(tmp_path):
    model_path = tmp_path / "model.csv.gz"
    with gzip.open(model_path, "wb") as gzip_file:
        gzip_file.write(
            b"PYMODELID;PYCONFIGURATIONNAME;PYSNAPSHOTTIME;PYPOSITIVES;"
            b"PYNEGATIVES;PYRESPONSECOUNT;PYPERFORMANCE;PYCHANNEL;PYDIRECTION;"
            b"PYISSUE;PYGROUP;PYNAME\n"
            b"model-1;Config;16/07/2026 10:30;10;90;100;70;Web;Inbound;"
            b"Issue;Group;Action\n",
        )

    result = import_health_check_data(
        model_path,
        model_options=SourceImportOptions(
            read=HealthCheckReadOptions(
                delimiter=";",
                infer_schema_length=0,
            ),
            normalize=SourceNormalizationOptions(
                timestamp_column="PYSNAPSHOTTIME",
                timestamp_format="%d/%m/%Y %H:%M",
                type_overrides={
                    "PYPOSITIVES": "float64",
                    "PYNEGATIVES": "float64",
                    "PYRESPONSECOUNT": "float64",
                    "PYPERFORMANCE": "float64",
                },
            ),
        ),
        extract_pyname_keys=False,
    )

    assert result.sources["model"].extension == ".gz"
    assert result.datamart.model_data.select("ModelID", "Performance").collect().to_dict(as_series=False) == {
        "ModelID": ["model-1"],
        "Performance": [0.7],
    }


def test_preview_and_import_health_check_data_read_excel_header_row():
    columns = preview_health_check_columns(
        _uploaded_model_workbook(),
        HealthCheckReadOptions(excel_sheet_id=1, excel_header_row=1),
    )
    assert columns == tuple(MINIMAL_MODEL_HEADER.rstrip("\n").split(","))

    result = import_health_check_data(
        _uploaded_model_workbook(),
        model_options=SourceImportOptions(
            read=HealthCheckReadOptions(
                excel_sheet_name="Data",
                excel_header_row=1,
                schema_overrides={"pyPositives": pl.Float64},
            ),
        ),
        extract_pyname_keys=False,
    )

    assert result.sources["model"].extension == ".xlsx"
    assert result.datamart.model_data.select("ModelID", "Positives").collect().to_dict(as_series=False) == {
        "ModelID": ["model-1"],
        "Positives": [10.0],
    }


def test_import_health_check_data_rejects_missing_model_source():
    with pytest.raises(ValueError, match="model_source is required"):
        import_health_check_data(None)  # type: ignore[arg-type]


def test_import_health_check_data_applies_predictor_categorization():
    model = _uploaded_csv(
        "pyModelID,pyConfigurationName,pySnapshotTime,pyPositives,pyNegatives,"
        "pyResponseCount,pyPerformance,pyChannel,pyDirection,pyIssue,pyGroup,pyName\n"
        "model-1,Config,20260716T103000.000 GMT,10,90,100,0.7,Web,Inbound,Issue,Group,Action\n",
    )
    predictor = _uploaded_csv(
        "pyModelID,pyPredictorName,pyPerformance,pyBinIndex,pyBinPositives,"
        "pyBinNegatives,pyBinResponseCount,pySnapshotTime,pyEntryType\n"
        "model-1,Customer.Score,0.62,1,10,90,100,20260716T103000.000 GMT,Active\n"
        "model-1,Classifier,0.70,1,30,70,100,20260716T103000.000 GMT,Classifier\n",
        name="predictor.csv",
    )

    result = import_health_check_data(
        model,
        predictor,
        extract_pyname_keys=False,
        predictor_categorization={"External Model": ["Score"]},
    )

    assert result.predictor_data is not None
    assert result.predictor_data.select("PredictorName", "PredictorCategory").collect().sort("PredictorName").to_dict(
        as_series=False
    ) == {
        "PredictorName": ["Classifier", "Customer.Score"],
        "PredictorCategory": [None, "External Model"],
    }


def test_import_health_check_data_applies_regex_predictor_categorization():
    model = _uploaded_csv(_minimal_model_csv())
    predictor = _uploaded_csv(
        "pyModelID,pyPredictorName,pyPerformance,pyBinIndex,pyBinPositives,"
        "pyBinNegatives,pyBinResponseCount,pySnapshotTime,pyEntryType\n"
        "model-1,Customer.ClassProbability,0.62,1,10,90,100,20260716T103000.000 GMT,Active\n"
        "model-1,Age,0.70,1,30,70,100,20260716T103000.000 GMT,Active\n",
        name="predictor.csv",
    )

    result = import_health_check_data(
        model,
        predictor,
        extract_pyname_keys=False,
        predictor_categorization={"External Model": r"Class|Score"},
        predictor_categorization_uses_regex=True,
    )

    assert result.predictor_data is not None
    assert result.predictor_data.select("PredictorName", "PredictorCategory").collect().sort("PredictorName").to_dict(
        as_series=False
    ) == {
        "PredictorName": ["Age", "Customer.ClassProbability"],
        "PredictorCategory": [None, "External Model"],
    }


def test_import_health_check_data_infers_sparse_prediction_export_schema():
    model = _uploaded_csv(
        "pyModelID,pyConfigurationName,pySnapshotTime,pyPositives,pyNegatives,"
        "pyResponseCount,pyPerformance,pyChannel,pyDirection,pyIssue,pyGroup,pyName\n"
        "model-1,Config,20260716T103000.000 GMT,10,90,100,0.7,Web,Inbound,Issue,Group,Action\n",
    )
    leading_rows = [
        {
            "pyFieldName": f"Predictor.{index}",
            "pyModelType": "ADAPTIVE",
            "pySnapShotTime": "20260716T103000.000 GMT",
            "pyValue": index,
        }
        for index in range(150)
    ]
    prediction = _uploaded_zip_ndjson(
        leading_rows
        + [
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
        ],
    )

    result = import_health_check_data(
        model,
        prediction_source=prediction,
        extract_pyname_keys=False,
    )

    assert result.prediction is not None
    assert result.warnings == ()
    prediction_data = result.prediction.predictions.collect().sort("Positives")
    assert prediction_data["Performance"].to_list() == pytest.approx([0.71] * 3)
    assert prediction_data.select(
        "ModelName",
        "ResponseCount",
        "Positives",
        "Positives_Test",
        "Positives_Control",
    ).to_dict(as_series=False) == {
        "ModelName": ["PREDICTWEBPROPENSITY"] * 3,
        "ResponseCount": [100.0] * 3,
        "Positives": [10.0, 20.0, 30.0],
        "Positives_Test": [20.0] * 3,
        "Positives_Control": [10.0] * 3,
    }


def _minimal_import_result():
    model = _uploaded_csv(_minimal_model_csv())
    return import_health_check_data(model, extract_pyname_keys=False)


def test_resolve_health_check_output_dir_uses_source_path_priority(tmp_path):
    model_parent = tmp_path / "model"
    predictor_parent = tmp_path / "predictor"
    prediction_parent = tmp_path / "prediction"

    assert (
        resolve_health_check_output_dir(
            model_parent / "model.csv",
            predictor_parent / "predictor.csv",
            prediction_parent / "prediction.csv",
        )
        == model_parent / "HC"
    )
    assert (
        resolve_health_check_output_dir(
            _uploaded_csv("x"),
            predictor_parent / "predictor.csv",
            prediction_parent / "prediction.csv",
        )
        == predictor_parent / "HC"
    )
    assert (
        resolve_health_check_output_dir(
            _uploaded_csv("x"),
            output_parent=tmp_path / "chosen",
        )
        == tmp_path / "chosen" / "HC"
    )


def test_resolve_health_check_output_dir_defaults_to_current_directory_for_uploads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_health_check_output_dir(_uploaded_csv("x")) == tmp_path / "HC"


def test_save_health_check_parquet_uses_canonical_names_and_round_trips(tmp_path):
    result = _minimal_import_result()

    paths = save_health_check_parquet(result, tmp_path)

    assert paths == {"model": tmp_path / "HC" / MODEL_CACHE_FILENAME}
    assert paths["model"].is_file()
    saved = pl.read_parquet(paths["model"])
    assert saved.columns == [
        "ModelID",
        "Configuration",
        "SnapshotTime",
        "Positives",
        "Negatives",
        "ResponseCount",
        "Performance",
        "Channel",
        "Direction",
        "Issue",
        "Group",
        "Name",
        "ModelTechnique",
        "SuccessRate",
        "IsUpdated",
        "LastUpdate",
    ]
    assert saved.schema["ModelTechnique"] == pl.String
    reloaded = import_health_check_data(
        paths["model"],
        extract_pyname_keys=False,
    )
    expected = saved.select(
        "ModelID",
        "SnapshotTime",
        "ResponseCount",
        "Performance",
    )
    actual = reloaded.datamart.model_data.select(
        "ModelID",
        "SnapshotTime",
        "ResponseCount",
        "Performance",
    ).collect()
    assert actual.equals(expected)


def test_save_health_check_parquet_removes_stale_optional_files(tmp_path):
    output_dir = tmp_path / "HC"
    output_dir.mkdir()
    stale_predictor = output_dir / PREDICTOR_CACHE_FILENAME
    stale_prediction = output_dir / PREDICTION_CACHE_FILENAME
    stale_predictor.write_bytes(b"stale")
    stale_prediction.write_bytes(b"stale")

    save_health_check_parquet(_minimal_import_result(), tmp_path)

    assert not stale_predictor.exists()
    assert not stale_prediction.exists()


def test_save_health_check_parquet_writes_optional_predictor_and_prediction_files(tmp_path):
    model = _uploaded_csv(_minimal_model_csv())
    predictor = _uploaded_csv(
        "pyModelID,pyPredictorName,pyPerformance,pyBinIndex,pyBinPositives,"
        "pyBinNegatives,pyBinResponseCount,pySnapshotTime,pyEntryType\n"
        "model-1,Customer.Score,0.62,1,10,90,100,20260716T103000.000 GMT,Active\n",
        name="predictor.csv",
    )
    prediction = _uploaded_zip_ndjson(
        [
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
        ],
    )

    result = import_health_check_data(
        model,
        predictor_source=predictor,
        prediction_source=prediction,
        extract_pyname_keys=False,
    )

    paths = save_health_check_parquet(result, tmp_path)

    assert set(paths) == {"model", "predictor", "prediction"}
    assert (tmp_path / "HC" / PREDICTOR_CACHE_FILENAME).is_file()
    assert (tmp_path / "HC" / PREDICTION_CACHE_FILENAME).is_file()
    assert pl.read_parquet(paths["predictor"]).get_column("PredictorName").to_list() == ["Customer.Score"]
    assert pl.read_parquet(paths["prediction"]).get_column("pyDataUsage").to_list() == ["Control", "Test", "NBA"]


def test_save_health_check_parquet_does_not_replace_files_after_write_failure(tmp_path):
    model = _uploaded_csv(
        "pyModelID,pyConfigurationName,pySnapshotTime,pyPositives,pyNegatives,"
        "pyResponseCount,pyPerformance,pyChannel,pyDirection,pyIssue,pyGroup,pyName\n"
        "model-1,Config,20260716T103000.000 GMT,10,90,100,0.7,Web,Inbound,Issue,Group,Action\n",
    )
    predictor = _uploaded_csv(
        "pyModelID,pyPredictorName,pyPerformance,pyBinIndex,pyBinPositives,"
        "pyBinNegatives,pyBinResponseCount,pySnapshotTime,pyEntryType\n"
        "model-1,Customer.Score,0.62,1,10,90,100,20260716T103000.000 GMT,Active\n",
        name="predictor.csv",
    )
    result = import_health_check_data(model, predictor, extract_pyname_keys=False)
    output_dir = tmp_path / "HC"
    output_dir.mkdir()
    existing_model = output_dir / MODEL_CACHE_FILENAME
    existing_model.write_bytes(b"existing")

    with patch.object(
        pl.LazyFrame,
        "sink_parquet",
        side_effect=[None, RuntimeError("write failed")],
    ):
        with pytest.raises(RuntimeError, match="write failed"):
            save_health_check_parquet(result, tmp_path)

    assert existing_model.read_bytes() == b"existing"
    assert list(output_dir.glob(".*.tmp.parquet")) == []
