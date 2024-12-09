from os import PathLike


def _read_client_credential_file(credential_file: PathLike):  # pragma: no cover
    outputdict = {}
    with open(credential_file) as f:
        for idx, line in enumerate(f.readlines()):
            if (idx % 2) == 0:
                key = line.rstrip("\n")
            else:
                outputdict[key] = line.rstrip("\n")
        return outputdict


def get_url(credential_file: PathLike):  # pragma: no cover
    """Returns the URL of the Infinity instance in the credential file"""
    url = _read_client_credential_file(credential_file)["Authorization endpoint"]
    return url.rsplit("/prweb")[0]


def get_token(credential_file: PathLike, verify: bool = True):  # pragma: no cover
    """Get API credentials to a Pega Platform instance.

    After setting up OAuth2 authentication in Dev Studio, you should
    be able to download a credential file. Simply point this method to that file,
    and it'll read the relevant properties and give you your access token.

    Parameters
    ----------
    credentialFile: str
        The credential file downloaded after setting up OAuth in a Pega system
    verify: bool, default = True
        Whether to only allow safe SSL requests.
        In case you're connecting to an unsecured API endpoint, you need to
        explicitly set verify to False, otherwise Python will yell at you.

    """
    import requests

    creds = _read_client_credential_file(credential_file)
    response = requests.post(
        url=creds["Access token endpoint"],
        data={"grant_type": "client_credentials"},
        auth=(creds["Client ID"], creds["Client Secret"]),
        verify=verify,
    ).json()
    if "errors" in response:
        raise ConnectionRefusedError(f"Error when connecting to infinity: {response}")
    return response["access_token"]
