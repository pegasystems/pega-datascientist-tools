import pytest
import sys
import polars as pl

import pathlib

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools.utils.hds_utils import Config, DataAnonymization


@pytest.fixture
def sampleInput():
    """Fixture to serve as input dataframe"""
    from io import StringIO

    input = StringIO(
        """{"Context_Name":"FirstMortgage30yr","Customer_MaritalStatus":"Married","Customer_CLV":"1460","Customer_City":"Port Raoul","IH_Web_Inbound_Accepted_pxLastGroupID":"Account","Decision_Outcome":"Rejected"}
            {"Context_Name":"FirstMortgage30yr","Customer_MaritalStatus":"Unknown","Customer_CLV":"669","Customer_City":"Laurianneshire","IH_Web_Inbound_Accepted_pxLastGroupID":"AutoLoans","Decision_Outcome":"Accepted"}
            {"Context_Name":"MoneyMarketSavingsAccount","Customer_MaritalStatus":"No Resp+","Customer_CLV":"1174","Customer_City":"Jacobshaven","IH_Web_Inbound_Accepted_pxLastGroupID":"Account","Decision_Outcome":"Rejected"}
            {"Context_Name":"BasicChecking","Customer_MaritalStatus":"Unknown","Customer_CLV":"1476","Customer_City":"Lindton","IH_Web_Inbound_Accepted_pxLastGroupID":"Account","Decision_Outcome":"Rejected"}
            {"Context_Name":"BasicChecking","Customer_MaritalStatus":"Married","Customer_CLV":"","Customer_City":"South Jimmieshire","IH_Web_Inbound_Accepted_pxLastGroupID":"DepositAccounts","Decision_Outcome":"Accepted"}
            {"Context_Name":"UPlusFinPersonal","Customer_MaritalStatus":"No Resp+","Customer_CLV":"533","Customer_City":"Bergeville","IH_Web_Inbound_Accepted_pxLastGroupID":null,"Decision_Outcome":"Rejected"}
            {"Context_Name":"BasicChecking","Customer_MaritalStatus":"No Resp+","Customer_CLV":"555","Customer_City":"Willyville","IH_Web_Inbound_Accepted_pxLastGroupID":"DepositAccounts","Decision_Outcome":"Rejected"}
            """
    )
    return pl.read_ndjson(input)


def testDefault(sampleInput):
    processed = DataAnonymization(df=sampleInput).process()
    cols = [
        "PREDICTOR_0",
        "PREDICTOR_1",
        "PREDICTOR_2",
        "Context_Name",
        "IH_PREDICTOR_0",
        "Decision_Outcome",
    ]
    assert processed.columns == cols
    processed = processed.select(cols)
    assert processed.select(pl.col(pl.Float64).round(8)).to_series().to_list() == [
        None if x is None else round(x, 8)
        for x in [
            0.9830328738069989,
            0.14422057264050903,
            0.679745493107105,
            1.0,
            None,
            0.0,
            0.02332979851537646,
        ]
    ]
    assert processed[5, 4] is None
    assert processed.get_column("Decision_Outcome").to_list() == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]