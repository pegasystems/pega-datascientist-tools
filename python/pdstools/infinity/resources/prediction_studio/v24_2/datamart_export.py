from ....internal._resource import api_method
from ..base import AsyncDataMartExport as AsyncPreviousDatamartExport
from ..base import DataMartExport as PreviousDatamartExport


class _DatamartExportV24_2Mixin:
    """v24.2 DatamartExport business logic — defined once."""

    def __init__(self, client, referenceId: str, location: str, repositoryName: str):
        """Initialize the DataMartExport class.

        Parameters
        ----------
        client : Client
            The client used to interact with the API.
        reference_id : str
            The reference ID for the data mart export.
        location : str
            The location of the data mart export.
        repository_name : str
            The name of the repository for the data mart export.

        """
        super().__init__(client=client)  # type: ignore[call-arg]
        self.reference_id = referenceId
        self.location = location
        self.repository_name = repositoryName

    @api_method
    async def get_export_status(self):
        """Fetches the current export status of a datamart.

        This method queries the export status of a datamart by its reference ID.

        Returns
        -------
        dict
            The response from the server containing the export status of the datamart.

        """
        endpoint = f"/prweb/api/PredictionStudio/v1/datamart/export/{self.reference_id}"
        response = await self._a_get(endpoint)
        if response.get("status") == "New":
            return {
                "status": response["status"],
                "last_message": "",
                "last_update_time": response["updateTimeStamp"],
            }
        return {
            "status": response["status"],
            "last_message": response["lastMessage"],
            "last_update_time": response["updateTimeStamp"],
        }


class DatamartExport(_DatamartExportV24_2Mixin, PreviousDatamartExport):
    pass


class AsyncDatamartExport(_DatamartExportV24_2Mixin, AsyncPreviousDatamartExport):
    pass
