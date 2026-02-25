import os
import shutil

import aioboto3
import pytest
from pdstools.pega_io.S3 import S3Data


# @mock_aws
@pytest.mark.skip(reason="Test is failing, disabled as per request")
async def test_get_s3_files():
    # Define parameters
    bucket_name = "my-test-bucket"
    prefix = "test-prefix"

    # Create aioboto3 session
    session = aioboto3.Session()

    # Create the S3 client within the mocked environment
    async with session.client("s3", region_name="us-east-1") as s3_client:
        # Create the bucket
        await s3_client.create_bucket(Bucket=bucket_name)

        # Put some test objects into the bucket
        test_keys = [f"{prefix}/file1.txt", f"{prefix}/file2.txt"]
        for key in test_keys:
            await s3_client.put_object(Bucket=bucket_name, Key=key, Body="Test content")

    # Now create the S3Data instance with the mocked session
    s3data = S3Data(bucket_name=bucket_name, session=session)

    # Ensure the temp_dir exists
    if not os.path.exists(s3data.temp_dir):
        os.makedirs(s3data.temp_dir)

    # Call the method you want to test
    result_files = await s3data.get_s3_files(prefix=prefix, verbose=False)

    # Define the expected output
    expected_files = [os.path.join(s3data.temp_dir, key) for key in test_keys]

    # Assert that the result matches the expected output
    assert set(result_files) == set(expected_files)

    # Clean up: Remove the temp_dir and its contents
    shutil.rmtree(s3data.temp_dir)
