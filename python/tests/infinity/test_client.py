from pdstools.infinity import Infinity


def test_init_without_version():
    client = Infinity.from_client_id_and_secret("TEST_URL", "NA", "NA")
    assert not client.version
    assert not hasattr(client, "prediction_studio")
