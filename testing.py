import pdstools
from pdstools import ADMDatamart

import polars as pl


datamart = ADMDatamart(
    "/Users/uyany/Library/CloudStorage/OneDrive-SharedLibraries-PegasystemsInc/AI Chapter Data Sets - Documents/Customers/T-Mobile US/20231215/datamart",
    extract_keys=True,
)


datamart.predictorData.collect()


datamart.combinedData.collect().filter(pl.col("Name") == "HINTHomeInternet").filter(
    pl.col("Channel") == "CallCenter"
).sort("GroupIndex")
