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
