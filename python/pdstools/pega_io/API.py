import requests
from typing import Literal


def _readClientCredentialFile(credentialFile):  # pragma: no cover
    outputdict = {}
    with open(credentialFile) as f:
        for idx, line in enumerate(f.readlines()):
            if (idx % 2) == 0:
                key = line.rstrip("\n")
            else:
                outputdict[key] = line.rstrip("\n")
        return outputdict


def getToken(credentialFile: str, verify: bool = True, **kwargs):  # pragma: no cover
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

    Keyword arguments
    -----------------
    url: str
        An optional override of the URL to connect to.
        This is also extracted out of the credential file, but you may want
        to customize this (to a different port, etc).
    """
    creds = _readClientCredentialFile(credentialFile)
    return requests.post(
        url=kwargs.get("URL", creds["Access token endpoint"]),
        data={"grant_type": "client_credentials"},
        auth=(creds["Client ID"], creds["Client Secret"]),
        verify=verify,
    ).json()["access_token"]


def setupAzureOpenAI(
    api_base: str = "https://aze-openai-01.openai.azure.com/",
    api_version: Literal["2023-03-15-preview", "2022-12-01"] = "2023-03-15-preview",
):
    """Convenience function to automagically setup Azure AD-based authentication
    for the Azure OpenAI service. Mostly meant as an internal tool within Pega,
    but can of course also be used beyond.

    Prerequisites (you should only need to do this once!):
    - Download Azure CLI (https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
    - Once installed, run 'az login' in your terminal
    - Additional dependencies: `(pip install)` azure-identity and `(pip install)` openai

    Running this function automatically sets, among others:
    - `openai.api_key`
    - `os.environ["OPENAI_API_KEY"]`

    This should ensure that you don't need to pass tokens and/or api_keys around.
    The key that's set has a lifetime, typically of one hour. Therefore, if you
    get an error message like 'invalid token', you may need to run this method again
    to refresh the token for another hour.

    Parameters
    ----------
    api_base : str
        The url of the Azure service name you'd like to connect to
        If you have access to the Azure OpenAI playground
        (https://oai.azure.com/portal), you can easily find this url by clicking
        'view code' in one of the playgrounds. If you have access to the Azure portal
        directly (https://portal.azure.com), this will be found under 'endpoint'.
        Else, ask your system administrator for the correct url.
    api_version : Literal["2023-03-15-preview", "2022-12-01"]:
        The version of the api to use

    Usage
    -----
    >>> from pdstools import setupAzureOpenAI
    >>> setupAzureOpenAI()

    """
    try:
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            EnvironmentCredential,
        )
    except ImportError:
        raise ImportError(
            "Can't find azure identity. Install through `pip install azure-identity`."
        )
    try:
        import openai
    except ImportError:
        raise ImportError(
            "Can't find openai. Install through `pip install --upgrade openai."
        )
    import os

    # Define strategy which potential authentication methods should be tried to gain an access token
    credential = ChainedTokenCredential(
        ManagedIdentityCredential(), EnvironmentCredential(), AzureCliCredential()
    )
    try:
        access_token = credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )
    except Exception as e:
        raise Exception(
            f"Exception: {e}. \nAre you sure you've installed Azure CLI & ran `az login`?"
        )
    openai.api_key = access_token.token
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_type = "azure_ad"

    os.environ["OPENAI_API_KEY"] = access_token.token
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_API_VERSION"] = api_version
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
