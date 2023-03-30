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

    async def getS3Files(self, prefix, verbose=True):
        """OOTB file exports can be written in many very small files.

        This method asyncronously retrieves these files, and puts them in
        a temporary directory.

        parameters
        ----------
        prefix: str
            The prefix, pointing to the s3 files. See boto3 docs for filter.

        """
        import os

        def createPathIfNotExists(path):
            if not os.path.exists(path):
                os.mkdir(path)

        createPathIfNotExists(self.temp_dir)
        session = aioboto3.Session()
        async with session.resource("s3") as s3:
            tasks = []
            files = []
            alreadyOnDisk = []
            bucket = await s3.Bucket(self.bucketName)
            async for s3_object in bucket.objects.filter(Prefix=prefix):
                filename = f"{self.temp_dir}/{s3_object.key}"
                files.append(filename)
                if os.path.exists(filename):
                    alreadyOnDisk.append(filename)
                    continue

                createPathIfNotExists(
                    f"{self.temp_dir}/{s3_object.key.rsplit('/')[:-1][0]}"
                )

                tasks.append(
                    asyncio.create_task(
                        s3.meta.client.download_file(
                            self.bucketName, s3_object.key, filename
                        )
                    )
                )
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
        return files

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
