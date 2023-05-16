import asyncio
import aioboto3
from tqdm.asyncio import tqdm
from . import File


class S3Data:
    def __init__(
        self,
        bucketName: str,
        temp_dir="./s3_download",
    ):
        """A class to interact with datasets exported to an S3 repository.

        Any files that are downloaded are copied into the `temp_dir` folder.
        Recommended to configure this to point to an empty folder on your computer.

        Parameters
        ----------
        bucketName: str
            The name of the bucket the datamart folder is in
        temp_dir: str, default = './s3_download'
            The directory to download the s3 files to before reading them
        """

        self.bucketName = bucketName
        self.temp_dir = temp_dir

    async def getS3Files(self, prefix, use_meta_files=False, verbose=True):
        """OOTB file exports can be written in many very small files.

        This method asyncronously retrieves these files, and puts them in
        a temporary directory.

        The logic, if `use_meta_files` is True, is:

        1. Take the prefix, add a `.` in front of it
        (`'path/to/files'` becomes (`'path/to/.files'`)

        * rsplit on `/` (`['path/to', 'files']`)

        * take the last element (`'files'`)

        * add `.` in front of it (`'.files'`)

        * concat back to a filepath  (`'path/to/.files'`)

        3. fetch all files in the repo that adhere to the prefix (`'path/to/.files*'`)

        4. For each file, if the file ends with `.meta`:

        * rsplit on '/' (`['path/to', '.files_001.json.meta']`)

        * for the last element (just the filename), strip the period and the .meta (`['path/to', 'files_001.json']`)

        * concat back to a filepath (`'path/to/files_001.json'`)

        5. Import all files in the list

        If `use_meta_files` is False, the logic is as simple as:

        1. Import all files starting with the prefix 
        (`'path/to/files'` gives 
        `['path/to/files_001.json', 'path/to/files_002.json', etc]`, 
        irrespective of whether a `.meta` file exists).

        Parameters
        ----------
        prefix: str
            The prefix, pointing to the s3 files. See boto3 docs for filter.
        use_meta_files: bool, default=False
            Whether to use the meta files to check for eligible files

        Notes
        -----
        We don't import/copy over the .meta files at all.
        There is an internal function, getNewFiles(), that checks if the filename
        exists in the local file system. Since the meta files are not really useful for
        local processing, there's no sense in copying them over. This logic also still
        works with the use_meta_files - we first check which files are 'eligible' in S3
        because they have a meta file, then we check if the 'real' files exist on disk.
        If the file is already on disk, we don't copy it over.

        """
        import os

        def createPathIfNotExists(path):
            if not os.path.exists(path):
                os.mkdir(path)

        def localFile(file):
            return f"{self.temp_dir}/{file}"

        def getNewFiles(files):
            newFiles, alreadyOnDisk = [], []
            for file in files:
                localFileName = localFile(file)
                if os.path.exists(localFileName):
                    alreadyOnDisk.append(file)
                else:
                    newFiles.append(file)
            return newFiles, alreadyOnDisk

        def createTask(file):
            filename = f"{self.temp_dir}/{file}"
            createPathIfNotExists(f"{self.temp_dir}/{file.rsplit('/')[:-1][0]}")
            return asyncio.create_task(
                s3.meta.client.download_file(self.bucketName, file, filename)
            )

        async def getfilesToImport(bucket, prefix, use_meta_files=False):
            if use_meta_files:
                to_import = []
                prefix2 = "/.".join(prefix.rsplit("/", 1))
                async for s3_object in bucket.objects.filter(Prefix=prefix2):
                    f = s3_object.key
                    if str(f).endswith(".meta"):
                        to_import.append(
                            "/".join(f.rsplit("/.", 1)).rsplit(".meta", 1)[0]
                        )
            else:
                to_import = [
                    s3_object.key
                    async for s3_object in bucket.objects.filter(Prefix=prefix)
                ]
            return getNewFiles(to_import)

        createPathIfNotExists(self.temp_dir)

        session = aioboto3.Session()
        async with session.resource("s3") as s3:
            bucket = await s3.Bucket(self.bucketName)
            files, alreadyOnDisk = await getfilesToImport(
                bucket, prefix, use_meta_files
            )

            tasks = [createTask(f) for f in files]

            _ = [
                await task_
                for task_ in tqdm.as_completed(
                    tasks,
                    total=len(tasks),
                    desc="Downloading files...",
                    disable=not verbose,
                )
            ]

        if verbose:
            print(
                f"Completed {prefix}. Imported {len(files)} files, skipped {len(alreadyOnDisk)} files."
            )
        return list(map(localFile, [*files, *alreadyOnDisk]))

    async def getDatamartData(
        self, table, datamart_folder: str = "datamart", verbose: bool = True
    ):
        """Wrapper method to import one of the tables in the datamart.

        Parameters
        ----------
        table: str
            One of the datamart tables. See notes for the full list.
        datamart_folder: str, default='datamart'
            The path to the 'datamart' folder within the s3 bucket.
            Typically, this is the top-level folder in the bucket.
        verbose: bool, default = True
            Whether to print out the progress of the import

        Note
        ----
        Supports the following tables:
        {
            - "modelSnapshot": "Data-Decision-ADM-ModelSnapshot_pzModelSnapshots",
            - "predictorSnapshot": "Data-Decision-ADM-PredictorBinningSnapshot_pzADMPredictorSnapshots",
            - "binaryDistribution": "Data-DM-BinaryDistribution",
            - "contingencyTable": "Data-DM-ContingencyTable",
            - "histogram": "Data-DM-Histogram",
            - "snapshot": "Data-DM-Snapshot",
            - "notification": "Data-DM-Notification",
        }
        """
        tables = {
            "modelSnapshot": "Data-Decision-ADM-ModelSnapshot_pzModelSnapshots",
            "predictorSnapshot": "Data-Decision-ADM-PredictorBinningSnapshot_pzADMPredictorSnapshots",
            "binaryDistribution": "Data-DM-BinaryDistribution",
            "contingencyTable": "Data-DM-ContingencyTable",
            "histogram": "Data-DM-Histogram",
            "snapshot": "Data-DM-Snapshot",
            "notification": "Data-DM-Notification",
        }
        prefix = f"{datamart_folder}/{tables[table]}"
        importedFiles = await self.getS3Files(prefix=prefix, verbose=verbose)
        return importedFiles

    async def get_ADMDatamart(
        self, datamart_folder: str = "datamart", verbose: bool = True
    ):
        """Get the ADMDatamart class directly from files in S3

        In the Prediction Studio settings, you can configure an automatic
        export of the monitoring tables to a chosen repository. This method
        interacts with that repository to retrieve files.

        Because this is an async function, you need to await it.
        See `Examples` for an example on how to use this (in a jupyter notebook).

        It checks for files that are already on your local device, but it always
        concatenates the raw zipped files together when calling the function, which can
        potentially make it slow. If you don't always need the latest data, just use
        :meth:`pdstools.adm.ADMDatamart.save_data()` to save the data to more easily
        digestible files.

        Parameters
        ----------
        verbose:
            Whether to print out the progress of the imports
        datamart_folder: str, default='datamart'
            The path to the 'datamart' folder within the s3 bucket.
            Typically, this is the top-level folder in the bucket.
        Examples
        --------
        >>> dm = await S3Datamart(bucketName='testbucket').get_ADMDatamart()
        """
        from pdstools import ADMDatamart

        modelData = await self.getDatamartData(
            "modelSnapshot", datamart_folder, verbose
        )
        predictorData = await self.getDatamartData(
            "predictorSnapshot", datamart_folder, verbose
        )
        return ADMDatamart(
            model_df=File.readMultiZip(modelData, verbose=verbose),
            predictor_df=File.readMultiZip(predictorData, verbose=verbose),
        )
