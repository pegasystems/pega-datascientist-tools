"""Targeted tests for under-covered branches in pdstools.pega_io.File."""

import gzip
import json
import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest

from pdstools.pega_io import File as F


# ---------------------------------------------------------------------------
# _is_artifact / _clean_artifacts
# ---------------------------------------------------------------------------


class TestArtifactHelpers:
    def test_is_artifact_macosx(self):
        assert F._is_artifact("__MACOSX/file")
        assert F._is_artifact("._something")
        assert F._is_artifact(".DS_Store")
        assert F._is_artifact("Thumbs.db")
        assert F._is_artifact("desktop.ini")

    def test_is_not_artifact(self):
        assert not F._is_artifact("data.json")
        assert not F._is_artifact("normal_file.parquet")

    def test_clean_artifacts_removes_macosx_dir(self, tmp_path):
        macos = tmp_path / "__MACOSX"
        macos.mkdir()
        (macos / "junk").write_text("x")
        (tmp_path / "data.json").write_text("{}")
        F._clean_artifacts(str(tmp_path))
        assert not macos.exists()
        assert (tmp_path / "data.json").exists()

    def test_clean_artifacts_removes_dotunder_files(self, tmp_path):
        (tmp_path / "._foo").write_text("x")
        (tmp_path / ".DS_Store").write_text("x")
        (tmp_path / "real.csv").write_text("a,b\n1,2\n")
        F._clean_artifacts(str(tmp_path))
        assert not (tmp_path / "._foo").exists()
        assert not (tmp_path / ".DS_Store").exists()
        assert (tmp_path / "real.csv").exists()

    def test_clean_artifacts_removes_thumbs_and_desktop_ini(self, tmp_path):
        (tmp_path / "Thumbs.db").write_text("x")
        (tmp_path / "desktop.ini").write_text("x")
        F._clean_artifacts(str(tmp_path))
        assert not (tmp_path / "Thumbs.db").exists()
        assert not (tmp_path / "desktop.ini").exists()


# ---------------------------------------------------------------------------
# _read_from_bytesio
# ---------------------------------------------------------------------------


def _ndjson_bytes(records):
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode()


class TestReadFromBytesIO:
    def test_parquet(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        buf = BytesIO()
        df.write_parquet(buf)
        out = F._read_from_bytesio(buf, ".parquet").collect()
        assert out["a"].to_list() == [1, 2, 3]

    def test_csv(self):
        buf = BytesIO(b"a,b\n1,2\n3,4\n")
        out = F._read_from_bytesio(buf, ".csv").collect()
        assert out.shape == (2, 2)

    def test_arrow(self):
        df = pl.DataFrame({"a": [1, 2]})
        buf = BytesIO()
        df.write_ipc(buf)
        out = F._read_from_bytesio(buf, ".arrow").collect()
        assert out["a"].to_list() == [1, 2]

    def test_json(self):
        buf = BytesIO(_ndjson_bytes([{"a": 1}, {"a": 2}]))
        out = F._read_from_bytesio(buf, ".json").collect()
        assert out["a"].to_list() == [1, 2]

    def test_gz_with_name_attribute(self):
        raw = _ndjson_bytes([{"x": 1}])
        compressed = gzip.compress(raw)
        buf = BytesIO(compressed)
        buf.name = "mydata.json.gz"
        out = F._read_from_bytesio(buf, ".gz").collect()
        assert out["x"].to_list() == [1]

    def test_gz_without_name_defaults_to_json(self):
        raw = _ndjson_bytes([{"y": 5}])
        buf = BytesIO(gzip.compress(raw))
        out = F._read_from_bytesio(buf, ".gz").collect()
        assert out["y"].to_list() == [5]

    def test_zip_with_top_level_data_json(self):
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("data.json", _ndjson_bytes([{"a": 1}, {"a": 2}]).decode())
        zip_buf.seek(0)
        out = F._read_from_bytesio(zip_buf, ".zip").collect()
        assert out["a"].to_list() == [1, 2]

    def test_zip_with_nested_data_json(self):
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("sub/data.json", _ndjson_bytes([{"a": 7}]).decode())
        zip_buf.seek(0)
        out = F._read_from_bytesio(zip_buf, ".zip").collect()
        assert out["a"].to_list() == [7]

    def test_zip_without_data_json_reads_supported_file(self):
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("model.csv", "a,b\n1,2\n3,4\n")
        zip_buf.seek(0)
        out = F._read_from_bytesio(zip_buf, ".zip").collect()
        assert out.to_dict(as_series=False) == {"a": [1, 3], "b": [2, 4]}

    def test_zip_with_bad_directory_raises_actionable_message(self):
        zip_buf = BytesIO(b"PK\x03\x04not-a-complete-zip")

        with pytest.raises(ValueError, match="not a complete ZIP archive"):
            F._read_from_bytesio(zip_buf, ".zip")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            F._read_from_bytesio(BytesIO(b"x"), ".unsupported")


# ---------------------------------------------------------------------------
# _extract_zip / _extract_tar
# ---------------------------------------------------------------------------


class TestExtractArchives:
    def test_extract_zip(self, tmp_path):
        archive = tmp_path / "x.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("inside.txt", "hi")
        extracted = F._extract_zip(archive)
        assert (Path(extracted) / "inside.txt").read_text() == "hi"

    def test_extract_tar(self, tmp_path):
        import tarfile

        # Build an inner file then archive it
        inner = tmp_path / "inner.txt"
        inner.write_text("hello")
        archive = tmp_path / "a.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(inner, arcname="inner.txt")
        extracted = F._extract_tar(archive)
        assert (Path(extracted) / "inner.txt").read_text() == "hello"


# ---------------------------------------------------------------------------
# read_multi_zip
# ---------------------------------------------------------------------------


class TestReadMultiZip:
    def _make_gz(self, path: Path, records):
        with gzip.open(path, "wb") as gz:
            gz.write(_ndjson_bytes(records))

    def test_reads_and_concatenates(self, tmp_path):
        f1 = tmp_path / "a.json.gz"
        f2 = tmp_path / "b.json.gz"
        self._make_gz(f1, [{"a": 1}])
        self._make_gz(f2, [{"a": 2}])
        out = F.read_multi_zip([str(f1), str(f2)], verbose=False).collect()
        assert sorted(out["a"].to_list()) == [1, 2]

    def test_unsupported_zip_type_removed(self):
        # zip_type kwarg removed in v5; only gzip was ever supported.
        with pytest.raises(TypeError):
            F.read_multi_zip([], zip_type="bzip2")  # type: ignore[call-arg]

    def test_verbose_prints(self, tmp_path, capsys):
        f1 = tmp_path / "a.json.gz"
        self._make_gz(f1, [{"a": 1}])
        F.read_multi_zip([str(f1)], verbose=True).collect()
        captured = capsys.readouterr()
        assert "Combining completed" in captured.out


# ---------------------------------------------------------------------------
# get_latest_file edge cases
# ---------------------------------------------------------------------------


class TestGetLatestFile:
    def test_unknown_target_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown target"):
            F.get_latest_file(str(tmp_path), target="something_else")


def test_read_ds_export_prefers_local_repo_data_for_raw_github_samples(monkeypatch):
    import requests

    def _fail_network(*args, **kwargs):
        raise AssertionError("network should not be used for checkout-local sample data")

    monkeypatch.setattr(requests, "get", _fail_network)

    df = F.read_ds_export(
        "sample_explainability_extract.parquet",
        path="https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data",
    )

    assert df.collect().shape == (7297, 17)


# ---------------------------------------------------------------------------
# Excel support in read_data
# ---------------------------------------------------------------------------


class TestReadDataExcel:
    FIXTURE = Path(__file__).parent.parent / "data" / "ia" / "ImpactAnalyzerExport_minimal.xlsx"

    def test_read_data_xlsx_returns_expected_row_count(self):
        result = F.read_data(self.FIXTURE)
        assert result.collect().height == 41

    def test_read_data_xlsx_known_columns_and_value(self):
        df = F.read_data(self.FIXTURE).collect()
        # The fixture is a Pega Infinity IA Excel export (Data sheet schema).
        assert "Experiment Name" in df.columns
        assert "Channel" in df.columns
        assert df.height == 41
        # Round-trip parity with a direct pl.read_excel call on the same fixture.
        expected = pl.read_excel(self.FIXTURE)
        assert df.equals(expected)

    def test_read_data_xlsx_bytesio_matches_path(self):
        uploaded = BytesIO(self.FIXTURE.read_bytes())
        uploaded.name = self.FIXTURE.name

        actual = F.read_data(uploaded).collect()
        expected = F.read_data(self.FIXTURE).collect()

        assert actual.equals(expected)

    def test_read_data_xlsx_forwards_read_options_for_bytesio(self):
        import xlsxwriter

        uploaded = BytesIO()
        workbook = xlsxwriter.Workbook(uploaded)
        worksheet = workbook.add_worksheet("Data")
        worksheet.write_row(0, 0, ["ignored metadata"])
        worksheet.write_row(1, 0, ["name", "value"])
        worksheet.write_row(2, 0, ["first", 1])
        workbook.close()
        uploaded.seek(0)
        uploaded.name = "data.xlsx"

        actual = F.read_data(
            uploaded,
            read_options={"sheet_name": "Data", "read_options": {"header_row": 1}},
        ).collect()

        assert actual.to_dict(as_series=False) == {
            "name": ["first"],
            "value": [1],
        }

    def test_read_data_xlsx_missing_backend_raises_friendly_error(self, monkeypatch):
        def _raise_missing_backend(path, **kwargs):
            raise ModuleNotFoundError("fastexcel")

        monkeypatch.setattr(pl, "read_excel", _raise_missing_backend)

        with pytest.raises(Exception, match="fastexcel"):
            F.read_data(self.FIXTURE).collect()


class TestReadDataDelimitedText:
    @pytest.mark.parametrize("extension", [".tsv", ".txt"])
    def test_tab_separated_path(self, tmp_path, extension):
        source = tmp_path / f"data{extension}"
        source.write_text("name\tvalue\nfirst\t1\nsecond\t2\n")

        actual = F.read_data(source).collect()

        assert actual.to_dict(as_series=False) == {
            "name": ["first", "second"],
            "value": [1, 2],
        }

    def test_custom_separator_bytesio(self):
        uploaded = BytesIO(b"name;value\nfirst;1\nsecond;2\n")
        uploaded.name = "data.txt"

        actual = F.read_data(
            uploaded,
            read_options={"separator": ";"},
        ).collect()

        assert actual.to_dict(as_series=False) == {
            "name": ["first", "second"],
            "value": [1, 2],
        }

    def test_custom_separator_glob(self, tmp_path):
        (tmp_path / "part-1.csv").write_text("name;value\nfirst;1\n")
        (tmp_path / "part-2.csv").write_text("name;value\nsecond;2\n")

        actual = F.read_data(
            str(tmp_path / "part-*.csv"),
            read_options={"separator": ";"},
        ).collect()

        assert actual.sort("value").to_dict(as_series=False) == {
            "name": ["first", "second"],
            "value": [1, 2],
        }

    def test_recursive_directory_reads_tab_separated_text(self, tmp_path):
        partition = tmp_path / "pxDecisionTime_day=20260716"
        partition.mkdir()
        (partition / "data.txt").write_text("name\tvalue\nfirst\t1\nsecond\t2\n")

        actual = F.read_data(tmp_path).collect()

        assert actual.to_dict(as_series=False) == {
            "name": ["first", "second"],
            "value": [1, 2],
        }

    def test_gzip_csv_path_forwards_read_options(self, tmp_path):
        source = tmp_path / "data.csv.gz"
        with gzip.open(source, "wb") as gz_file:
            gz_file.write(b"name;value\nfirst;1\nsecond;2\n")

        actual = F.read_data(source, read_options={"separator": ";"}).collect()

        assert actual.to_dict(as_series=False) == {
            "name": ["first", "second"],
            "value": [1, 2],
        }
