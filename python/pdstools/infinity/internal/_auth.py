from __future__ import annotations

import logging
import time

import httpx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class PegaOAuth(httpx.Auth):
    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        verify: bool = False,
    ):
        self.base_url = base_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.verify = verify

    @property
    def token(self):
        if token := self.__dict__.get("_token"):
            if self._token_expiry - time.time() > 5:  # 5 seconds for some safety margin
                logger.debug("Reusing valid token.")
                return token

        logger.debug("Requesting new token.")
        response = httpx.post(
            url=self.base_url + "/prweb/PRRestService/oauth2/v1/token",
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            verify=self.verify,
        )
        if response.status_code != 200:
            try:
                body = response.json()
                # Only surface the safe OAuth error fields; never leak the full body.
                error_detail = body.get("error_description") or body.get("error") or "no error detail returned"
            except Exception:
                error_detail = "non-JSON response from token endpoint"
            raise ConnectionError(
                f"Unable to obtain OAuth token (HTTP {response.status_code}): {error_detail}. "
                "Check that client_id and client_secret are correct."
            )
        try:
            new_token = response.json()
        except Exception:
            raise ConnectionError(
                "OAuth token endpoint returned a non-JSON response."
            ) from None
        self._token_expiry = time.time() + new_token.get("expires_in")
        self._token = new_token.get("access_token")
        return self._token

    @property
    def _auth_header(self):
        return f"Bearer {self.token}"

    def auth_flow(
        self,
        request: httpx.Request,
    ) -> Generator[httpx.Request, httpx.Response, None]:  # pragma: no cover
        request.headers["Authorization"] = self._auth_header
        yield request


def _read_client_credential_file(credential_file):  # pragma: no cover
    outputdict = {}
    with open(credential_file) as f:
        for idx, line in enumerate(f.readlines()):
            if (idx % 2) == 0:
                key = line.strip("\n")
            else:
                outputdict[key] = line.strip("\n")
        return outputdict
