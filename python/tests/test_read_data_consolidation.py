"""Tests for read_data consolidation to pega_io.File."""

from io import BytesIO

import polars as pl


class TestReadDataConsolidation:
    """Verify read_data consolidation works correctly."""

    def test_import_from_pega_io_module(self):
        """read_data should be importable from pega_io module."""
        from pdstools.pega_io import read_data

        assert callable(read_data)

    def test_import_from_pega_io_file(self):
        """read_data should be importable from pega_io.File directly."""
        from pdstools.pega_io.File import read_data

        assert callable(read_data)

    def test_read_data_basic_functionality(self, tmp_path):
        """read_data should work with basic file types."""
        from pdstools.pega_io import read_data

        # Test with parquet
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        result = read_data(path)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_read_data_with_bytesio(self):
        """read_data should handle BytesIO objects."""
        from pdstools.pega_io import read_data

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        bio = BytesIO()
        df.write_parquet(bio)
        bio.seek(0)
        bio.name = "test.parquet"

        result = read_data(bio)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_read_ds_export_delegates_to_read_data_for_bytesio(self):
        """read_ds_export should delegate to read_data for BytesIO objects."""
        from pdstools.pega_io import read_ds_export

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        bio = BytesIO()
        df.write_parquet(bio)
        bio.seek(0)
        bio.name = "test.parquet"

        result = read_ds_export(bio)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_read_ds_export_still_works_for_files(self, tmp_path):
        """read_ds_export should still work for file paths."""
        from pdstools.pega_io import read_ds_export

        df = pl.DataFrame({"col1": [10, 20], "col2": ["foo", "bar"]})
        path = tmp_path / "data.csv"
        df.write_csv(path)

        result = read_ds_export(str(path))
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (2, 2)

    def test_helper_functions_accessible(self):
        """Helper functions should be accessible from pega_io.File."""
        from pdstools.pega_io.File import _clean_artifacts, _is_artifact

        assert callable(_clean_artifacts)
        assert callable(_is_artifact)

        # Test _is_artifact logic
        assert _is_artifact("__MACOSX") is True
        assert _is_artifact("._file") is True
        assert _is_artifact(".DS_Store") is True
        assert _is_artifact("normal_file.txt") is False

    def test_decision_analyzer_specific_functions_remain(self):
        """Decision Analyzer-specific functions should remain in data_read_utils."""
        from pdstools.decision_analyzer.data_read_utils import (
            read_gzipped_data,
            read_nested_zip_files,
            validate_columns,
        )

        assert callable(read_gzipped_data)
        assert callable(read_nested_zip_files)
        assert callable(validate_columns)

    def test_backward_compatibility_all_formats(self, tmp_path):
        """Verify all supported formats work after consolidation."""
        from pdstools.pega_io import read_data

        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

        # Test each format
        formats = [
            ("test.parquet", lambda p: df.write_parquet(p)),
            ("test.csv", lambda p: df.write_csv(p)),
            ("test.arrow", lambda p: df.write_ipc(p)),
            ("test.json", lambda p: df.write_ndjson(p)),
        ]

        for filename, writer in formats:
            path = tmp_path / filename
            writer(path)
            result = read_data(path)
            assert isinstance(result, pl.LazyFrame), f"Failed for {filename}"
            assert result.collect().shape[0] == 3, f"Wrong shape for {filename}"

    def test_streamlit_utils_imports_correctly(self):
        """Streamlit utils should import from the new locations."""
        from pdstools.app.decision_analyzer.da_streamlit_utils import (
            _clean_artifacts,
            read_data,
            read_nested_zip_files,
        )

        assert callable(_clean_artifacts)
        assert callable(read_data)
        assert callable(read_nested_zip_files)

    def test_tar_archive_support(self, tmp_path):
        """read_data should handle TAR archives."""
        import tarfile

        from pdstools.pega_io import read_data

        # Create a parquet file
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        parquet_path = tmp_path / "data.parquet"
        df.write_parquet(parquet_path)

        # Create a tar archive
        tar_path = tmp_path / "archive.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(parquet_path, arcname="data.parquet")

        result = read_data(tar_path)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert collected.shape == (3, 2)

    def test_gzip_decompression(self, tmp_path):
        """read_data should handle gzipped files."""
        import gzip

        from pdstools.pega_io import read_data

        # Create a gzipped CSV
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        csv_content = df.write_csv()

        gz_path = tmp_path / "data.csv.gz"
        with gzip.open(gz_path, "wb") as f:
            f.write(csv_content.encode())

        result = read_data(gz_path)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert collected.shape == (3, 2)

    def test_clean_artifacts_removes_os_junk(self, tmp_path):
        """_clean_artifacts should remove __MACOSX, .DS_Store, and ._* files."""
        from pdstools.pega_io.File import _clean_artifacts

        # Create various artifact files and directories
        (tmp_path / "__MACOSX").mkdir()
        (tmp_path / "__MACOSX" / "junk.txt").write_text("junk")
        (tmp_path / ".DS_Store").write_text("junk")
        (tmp_path / "._hidden").write_text("junk")
        (tmp_path / "Thumbs.db").write_text("junk")
        (tmp_path / "desktop.ini").write_text("junk")
        (tmp_path / "real_file.txt").write_text("real data")

        _clean_artifacts(str(tmp_path))

        # Artifacts should be gone
        assert not (tmp_path / "__MACOSX").exists()
        assert not (tmp_path / ".DS_Store").exists()
        assert not (tmp_path / "._hidden").exists()
        assert not (tmp_path / "Thumbs.db").exists()
        assert not (tmp_path / "desktop.ini").exists()
        # Real file should remain
        assert (tmp_path / "real_file.txt").exists()

    def test_bytesio_csv(self):
        """read_data should handle BytesIO with CSV data."""
        from pdstools.pega_io import read_data

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        bio = BytesIO()
        df.write_csv(bio)
        bio.seek(0)
        bio.name = "test.csv"

        result = read_data(bio)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_bytesio_json(self):
        """read_data should handle BytesIO with JSON data."""
        from pdstools.pega_io import read_data

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        bio = BytesIO()
        df.write_ndjson(bio)
        bio.seek(0)
        bio.name = "test.json"

        result = read_data(bio)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_bytesio_arrow(self):
        """read_data should handle BytesIO with Arrow/IPC data."""
        from pdstools.pega_io import read_data

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        bio = BytesIO()
        df.write_ipc(bio)
        bio.seek(0)
        bio.name = "test.arrow"

        result = read_data(bio)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_bytesio_feather(self):
        """read_data should handle BytesIO with feather extension."""
        from pdstools.pega_io import read_data

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        bio = BytesIO()
        df.write_ipc(bio)
        bio.seek(0)
        bio.name = "test.feather"

        result = read_data(bio)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_tar_gz_archive(self, tmp_path):
        """read_data should handle .tar.gz archives."""
        import tarfile

        from pdstools.pega_io import read_data

        # Create a parquet file
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        parquet_path = tmp_path / "data.parquet"
        df.write_parquet(parquet_path)

        # Create a tar.gz archive
        tar_path = tmp_path / "archive.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(parquet_path, arcname="data.parquet")

        result = read_data(tar_path)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_tgz_archive(self, tmp_path):
        """read_data should handle .tgz archives."""
        import tarfile

        from pdstools.pega_io import read_data

        # Create a parquet file
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        parquet_path = tmp_path / "data.parquet"
        df.write_parquet(parquet_path)

        # Create a .tgz archive
        tar_path = tmp_path / "archive.tgz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(parquet_path, arcname="data.parquet")

        result = read_data(tar_path)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_pega_dataset_export_zip(self, tmp_path):
        """read_data should handle Pega Dataset Export format (zip with data.json)."""
        import zipfile

        from pdstools.pega_io import read_data

        # Create a Pega Dataset Export format zip
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        json_data = df.write_ndjson()

        zip_path = tmp_path / "pega_export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.json", json_data)
            zf.writestr("META-INF/MANIFEST.mf", "Manifest-Version: 1.0")

        result = read_data(zip_path)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)

    def test_zip_with_nested_data_json(self, tmp_path):
        """read_data should handle zip with nested data.json."""
        import zipfile

        from pdstools.pega_io import read_data

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        json_data = df.write_ndjson()

        zip_path = tmp_path / "nested.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("some_folder/data.json", json_data)

        result = read_data(zip_path)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().shape == (3, 2)
