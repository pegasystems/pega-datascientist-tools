import requests

def readClientCredentialFile(credentialFile):  # pragma: no cover
    outputdict = {}
    with open(credentialFile) as f:
        for idx, line in enumerate(f.readlines()):
            if (idx % 2) == 0:
                key = line.rstrip("\n")
            else:
                outputdict[key] = line.rstrip("\n")
        return outputdict


def getToken(credentialFile, verify=True, **kwargs):  # pragma: no cover
    creds = readClientCredentialFile(credentialFile)
    return requests.post(
        url=kwargs.get("URL", creds["Access token endpoint"]),
        data={"grant_type": "client_credentials"},
        auth=(creds["Client ID"], creds["Client Secret"]),
        verify=verify,
    ).json()["access_token"]
