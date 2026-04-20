"""Tests for pdstools.pega_io.S3 — async S3Data class with mocked aioboto3."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdstools.pega_io.S3 import S3Data


class _AsyncIter:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        async def gen():
            for item in self.items:
                yield item

        return gen()


def _make_bucket_with_keys(keys):
    """Build a mock bucket whose objects.filter(Prefix=...) yields S3 objects."""
    bucket = MagicMock()
    objects = MagicMock()
    bucket.objects = objects

    def filter_(Prefix):
        # Only return keys that start with the requested prefix
        matched = [MagicMock(key=k) for k in keys if k.startswith(Prefix)]
        return _AsyncIter(matched)

    objects.filter = filter_
    return bucket


def _patch_aioboto3(monkeypatch, keys):
    """Inject a fake aioboto3 module into sys.modules and return the
    download_file mock used by tasks."""
    download_file = AsyncMock(return_value=None)

    s3_resource = MagicMock()
    s3_resource.meta.client.download_file = download_file
    bucket = _make_bucket_with_keys(keys)
    s3_resource.Bucket = AsyncMock(return_value=bucket)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=s3_resource)
    cm.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.resource = MagicMock(return_value=cm)

    fake_aioboto3 = MagicMock()
    fake_aioboto3.Session = MagicMock(return_value=session)
    monkeypatch.setitem(sys.modules, "aioboto3", fake_aioboto3)
    return download_file


class TestConstructor:
    def test_defaults(self):
        s3 = S3Data(bucketName="my-bucket")
        assert s3.bucketName == "my-bucket"
        assert s3.temp_dir == "./s3_download"

    def test_custom_temp_dir(self):
        s3 = S3Data(bucketName="b", temp_dir="/tmp/x")
        assert s3.temp_dir == "/tmp/x"


class TestMissingAioboto3:
    def test_raises_missing_dependencies_exception(self, monkeypatch):
        # Force "import aioboto3" inside getS3Files to fail
        monkeypatch.setitem(sys.modules, "aioboto3", None)
        from pdstools.utils.namespaces import MissingDependenciesException

        s3 = S3Data(bucketName="b", temp_dir="/tmp/no")
        import asyncio

        with pytest.raises(MissingDependenciesException):
            asyncio.run(s3.getS3Files(prefix="x"))


class TestGetS3Files:
    def test_downloads_new_files(self, monkeypatch, tmp_path):
        keys = ["data/file1.json", "data/file2.json"]
        download_file = _patch_aioboto3(monkeypatch, keys)

        s3 = S3Data(bucketName="b", temp_dir=str(tmp_path / "dl"))
        import asyncio

        result = asyncio.run(s3.getS3Files(prefix="data/", verbose=False))

        assert download_file.await_count == 2
        # All returned paths should be inside temp_dir
        assert all(str(tmp_path / "dl") in r for r in result)
        assert len(result) == 2

    def test_skips_files_already_on_disk(self, monkeypatch, tmp_path):
        keys = ["data/file1.json", "data/file2.json"]
        download_file = _patch_aioboto3(monkeypatch, keys)

        temp = tmp_path / "dl"
        temp.mkdir()
        # Pre-create the local files so they are detected as already present.
        # Mirror the path layout the implementation uses: temp_dir/<key>
        sub = temp / "data"
        sub.mkdir()
        (sub / "file1.json").write_text("x")
        (sub / "file2.json").write_text("x")

        s3 = S3Data(bucketName="b", temp_dir=str(temp))
        import asyncio

        result = asyncio.run(s3.getS3Files(prefix="data/", verbose=False))
        # No downloads since both already on disk
        assert download_file.await_count == 0
        assert len(result) == 2

    def test_use_meta_files(self, monkeypatch, tmp_path):
        # With use_meta_files=True, the implementation lists '.files' prefix
        # and only picks keys ending with .meta, mapping them back to the real file.
        keys = [
            "data/.files_001.json.meta",
            "data/.files_002.json.meta",
            # A .meta-less file under the dotted prefix should be skipped
            "data/.files_README",
        ]
        download_file = _patch_aioboto3(monkeypatch, keys)

        s3 = S3Data(bucketName="b", temp_dir=str(tmp_path / "dl"))
        import asyncio

        result = asyncio.run(
            s3.getS3Files(prefix="data/files", use_meta_files=True, verbose=False),
        )
        # Two .meta files → two real files downloaded
        assert download_file.await_count == 2
        # Resulting paths should reference the real (non-.meta) filenames
        joined = " ".join(result)
        assert "files_001.json" in joined
        assert "files_002.json" in joined
        assert ".meta" not in joined

    def test_verbose_prints_summary(self, monkeypatch, tmp_path, capsys):
        _patch_aioboto3(monkeypatch, ["d/a.json"])
        s3 = S3Data(bucketName="b", temp_dir=str(tmp_path / "dl"))
        import asyncio

        asyncio.run(s3.getS3Files(prefix="d/", verbose=True))
        captured = capsys.readouterr()
        assert "Completed d/" in captured.out
        assert "Imported 1 files" in captured.out

    def test_tqdm_missing_falls_back(self, monkeypatch, tmp_path):
        download_file = _patch_aioboto3(monkeypatch, ["d/a.json"])
        # Hide tqdm.asyncio so the ImportError fallback path runs
        monkeypatch.setitem(sys.modules, "tqdm.asyncio", None)

        s3 = S3Data(bucketName="b", temp_dir=str(tmp_path / "dl"))
        import asyncio

        asyncio.run(s3.getS3Files(prefix="d/", verbose=False))
        assert download_file.await_count == 1


class TestGetDatamartData:
    def test_routes_modelSnapshot_to_correct_prefix(self, monkeypatch, tmp_path):
        with patch.object(S3Data, "getS3Files", new=AsyncMock(return_value=["x"])) as mock_files:
            s3 = S3Data(bucketName="b", temp_dir=str(tmp_path))
            import asyncio

            result = asyncio.run(s3.getDatamartData("modelSnapshot", verbose=False))
        assert result == ["x"]
        kwargs = mock_files.await_args.kwargs
        assert kwargs["prefix"] == "datamart/Data-Decision-ADM-ModelSnapshot_pzModelSnapshots"

    def test_custom_datamart_folder(self, monkeypatch, tmp_path):
        with patch.object(S3Data, "getS3Files", new=AsyncMock(return_value=[])) as mock_files:
            s3 = S3Data(bucketName="b", temp_dir=str(tmp_path))
            import asyncio

            asyncio.run(
                s3.getDatamartData("histogram", datamart_folder="dm2", verbose=False),
            )
        assert mock_files.await_args.kwargs["prefix"] == "dm2/Data-DM-Histogram"

    def test_unknown_table_raises(self, tmp_path):
        s3 = S3Data(bucketName="b", temp_dir=str(tmp_path))
        import asyncio

        with pytest.raises(KeyError):
            asyncio.run(s3.getDatamartData("nonexistent_table"))
