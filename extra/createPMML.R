# Example preprocessing to create off-line model reports for
# CDH Sample, for CreditCards offers in Outbound channels

library(cdhtools)
library(data.table)
library(XML)

# Read the latest ADM Model export file from the Downloads folder
modelz <- readADMDatamartModelExport("~/Downloads")

# Subset to only the models of interest. Note that the field names have
# been stripped from the "py"/"px" prefixes.
modelz <- modelz[ ConfigurationName == "OmniAdaptiveModel" & Group == "CreditCards" & Direction == "Outbound"]

# Read the most recent ADM Predictor export file from the Downloads folder
predz <- readADMDatamartPredictorExport("~/Downloads", latestOnly = F, noBinning = F)

# Generate the PMML file
adm2pmml(dmModels = modelz, dmPredictors = predz, tmpDir = "~/tmp", verbose = T)
