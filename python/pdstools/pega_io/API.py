"""Pega Infinity OAuth2 helpers (credential-file based)."""

from __future__ import annotations

from os import PathLike


def _read_client_credential_file(credential_file: PathLike) -> dict[str, str]:  # pragma: no cover
    """Parse a Pega OAuth credential file into a key/value dict.

    The file is a simple key-then-value, line-by-line format exported
    by Dev Studio.
    """
    outputdict: dict[str, str] = {}
    key = ""
    with open(credential_file) as f:
        for idx, line in enumerate(f.readlines()):
            if (idx % 2) == 0:
                key = line.rstrip("\n")
            else:
                outputdict[key] = line.rstrip("\n")
    return outputdict


def get_url(credential_file: PathLike) -> str:  # pragma: no cover
    """Return the base URL of the Infinity instance in the credential file."""
    url = _read_client_credential_file(credential_file)["Authorization endpoint"]
    return url.rsplit("/prweb")[0]


def get_token(credential_file: PathLike, verify: bool = True) -> str:  # pragma: no cover
    """Fetch an OAuth2 access token for a Pega Platform instance.

    After configuring OAuth2 in Dev Studio, download the credential
    file and point this helper at it.

    Parameters
    ----------
    credential_file : PathLike
        Path to the credential file downloaded from Pega.
    verify : bool, default=True
        Whether to verify TLS certificates.  Set to ``False`` only for
        unsecured test endpoints.

    Returns
    -------
    str
        The bearer access token.
    """
    import requests  # type: ignore[import-untyped]  # requests has no PEP 561 stubs

    creds = _read_client_credential_file(credential_file)
    response = requests.post(
        url=creds["Access token endpoint"],
        data={"grant_type": "client_credentials"},
        auth=(creds["Client ID"], creds["Client Secret"]),
        verify=verify,
    ).json()
    if "errors" in response:
        raise ConnectionRefusedError(f"Error when connecting to Infinity: {response}")
    return response["access_token"]
