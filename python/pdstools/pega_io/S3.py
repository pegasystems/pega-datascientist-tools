"""Async S3 helper for downloading Pega dataset exports."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from .File import read_multi_zip

if TYPE_CHECKING:
    from ..adm.ADMDatamart import ADMDatamart

logger = logging.getLogger(__name__)


# Datamart table → S3 key prefix.  Public so users can extend.
DATAMART_TABLE_PREFIXES: dict[str, str] = {
    "modelSnapshot": "Data-Decision-ADM-ModelSnapshot_pzModelSnapshots",
    "predictorSnapshot": "Data-Decision-ADM-PredictorBinningSnapshot_pzADMPredictorSnapshots",
    "binaryDistribution": "Data-DM-BinaryDistribution",
    "contingencyTable": "Data-DM-ContingencyTable",
    "histogram": "Data-DM-Histogram",
    "snapshot": "Data-DM-Snapshot",
    "notification": "Data-DM-Notification",
}


class S3Data:
    """Asynchronous helper for downloading Pega datasets from S3.

    Use this when Prediction Studio is configured to export monitoring
    tables to an S3 bucket: it downloads the partitioned ``.json.gz``
    files into a local directory and (optionally) hands them off to
    :class:`pdstools.adm.ADMDatamart`.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket containing the dataset folder.
    temp_dir : str, default="./s3_download"
        Directory where downloaded files are placed.  Should be a folder
        you don't mind being filled with cached exports.
    """

    def __init__(self, bucket_name: str, temp_dir: str = "./s3_download"):
        self.bucket_name = bucket_name
        self.temp_dir = temp_dir

    async def get_files(
        self,
        prefix: str,
        *,
        use_meta_files: bool = False,
        verbose: bool = True,
    ) -> list[str]:
        """Download files from the bucket whose key starts with ``prefix``.

        Pega data exports are split into many small files.  This method
        fetches them concurrently into :attr:`temp_dir`, skipping any
        file that already exists locally.

        When ``use_meta_files`` is True, each real export file ``X`` is
        accompanied by a ``.X.meta`` sentinel file that signals the
        export has finished. We list keys under the dotted prefix
        (``path/to/.files``), keep entries ending in ``.meta``, and map
        them back to the underlying file (``path/to/files_001.json``).
        ``.meta`` files themselves are never copied locally.

        When ``use_meta_files`` is False, every key under ``prefix`` is
        downloaded.

        Parameters
        ----------
        prefix : str
            S3 key prefix (see boto3 ``Bucket.objects.filter(Prefix=...)``).
        use_meta_files : bool, keyword-only, default=False
            Whether to use companion ``.meta`` files to gate downloads.
        verbose : bool, keyword-only, default=True
            Show a tqdm progress bar (if installed) and print a summary.

        Returns
        -------
        list[str]
            Local paths of all files that match ``prefix`` (newly
            downloaded *and* already cached).
        """
        try:
            import aioboto3
        except ImportError:
            from ..utils.namespaces import MissingDependenciesException

            raise MissingDependenciesException(
                ["aioboto3"],
                namespace="S3Data",
                deps_group="pega_io",
            )

        def ensure_path(path: str) -> None:
            if not os.path.exists(path):
                os.mkdir(path)

        def local_path(file: str) -> str:
            return f"{self.temp_dir}/{file}"

        def split_new_files(files: list[str]) -> tuple[list[str], list[str]]:
            new_files, already_on_disk = [], []
            for file in files:
                (already_on_disk if os.path.exists(local_path(file)) else new_files).append(file)
            return new_files, already_on_disk

        def make_download_task(file: str):
            filename = f"{self.temp_dir}/{file}"
            ensure_path(f"{self.temp_dir}/{file.rsplit('/')[:-1][0]}")
            return asyncio.create_task(
                s3.meta.client.download_file(self.bucket_name, file, filename),
            )

        async def discover_files(bucket, prefix: str, use_meta_files: bool):
            if use_meta_files:
                to_import: list[str] = []
                dotted_prefix = "/.".join(prefix.rsplit("/", 1))
                async for s3_object in bucket.objects.filter(Prefix=dotted_prefix):
                    key = s3_object.key
                    if str(key).endswith(".meta"):
                        to_import.append(
                            "/".join(key.rsplit("/.", 1)).rsplit(".meta", 1)[0],
                        )
            else:
                to_import = [s3_object.key async for s3_object in bucket.objects.filter(Prefix=prefix)]
            return split_new_files(to_import)

        ensure_path(self.temp_dir)

        session = aioboto3.Session()
        async with session.resource("s3") as s3:
            bucket = await s3.Bucket(self.bucket_name)
            files, already_on_disk = await discover_files(bucket, prefix, use_meta_files)

            tasks = [make_download_task(f) for f in files]

            try:
                from tqdm.asyncio import tqdm

                iterable = tqdm.as_completed(
                    tasks,
                    total=len(tasks),
                    desc="Downloading files...",
                    disable=not verbose,
                )
            except ImportError:
                iterable = tasks

            for task in iterable:
                await task

        if verbose:
            print(
                f"Completed {prefix}. Imported {len(files)} files, skipped {len(already_on_disk)} files.",
            )
        return [local_path(f) for f in (*files, *already_on_disk)]

    async def get_datamart_data(
        self,
        table: str,
        *,
        datamart_folder: str = "datamart",
        verbose: bool = True,
    ) -> list[str]:
        """Download a single datamart table from S3.

        Parameters
        ----------
        table : str
            Datamart table name. One of the keys in
            :data:`DATAMART_TABLE_PREFIXES`: ``"modelSnapshot"``,
            ``"predictorSnapshot"``, ``"binaryDistribution"``,
            ``"contingencyTable"``, ``"histogram"``, ``"snapshot"``,
            ``"notification"``.
        datamart_folder : str, keyword-only, default="datamart"
            Top-level folder inside the bucket that contains the
            datamart export.
        verbose : bool, keyword-only, default=True
            Show download progress.

        Returns
        -------
        list[str]
            Local paths of the downloaded files.
        """
        prefix = f"{datamart_folder}/{DATAMART_TABLE_PREFIXES[table]}"
        return await self.get_files(prefix=prefix, verbose=verbose)

    async def get_adm_datamart(
        self,
        *,
        datamart_folder: str = "datamart",
        verbose: bool = True,
    ) -> ADMDatamart:
        """Construct an :class:`ADMDatamart` directly from S3.

        Convenience wrapper that downloads the model and predictor
        snapshot exports and feeds them into :class:`ADMDatamart`.
        Because this is an async function, it must be awaited.

        Parameters
        ----------
        datamart_folder : str, keyword-only, default="datamart"
            Top-level folder inside the bucket that contains the
            datamart export.
        verbose : bool, keyword-only, default=True
            Show download progress.

        Returns
        -------
        ADMDatamart
            A datamart populated with the freshly downloaded files.

        Examples
        --------
        >>> dm = await S3Data(bucket_name="testbucket").get_adm_datamart()
        """
        from ..adm.ADMDatamart import ADMDatamart

        model_files = await self.get_datamart_data("modelSnapshot", datamart_folder=datamart_folder, verbose=verbose)
        predictor_files = await self.get_datamart_data(
            "predictorSnapshot", datamart_folder=datamart_folder, verbose=verbose
        )
        return ADMDatamart(
            model_df=read_multi_zip(model_files, verbose=verbose),
            predictor_df=read_multi_zip(predictor_files, verbose=verbose),
        )
