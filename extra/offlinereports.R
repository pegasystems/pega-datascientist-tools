# Example preprocessing to create off-line model reports for
# CDH Sample, for CreditCards offers in Outbound channels

library(cdhtools)
library(data.table)

# Read the latest ADM Model export file from the Downloads folder
modelz <- readADMDatamartModelExport("~/Downloads")

# Subset to only the models of interest. Note that the field names have
# been stripped from the "py"/"px" prefixes.
modelz <- modelz[ ConfigurationName == "OmniAdaptiveModel" & Group == "CreditCards" & Direction == "Outbound"]

# Read the latest ADM Predictor export file from the Downloads folder
#predz <- readADMDatamartPredictorExport("~/Downloads")
predz <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots", "~/Downloads")
applyUniformPegaFieldCasing(predz)

# Write back a CSV with only the model data of interest
write.csv(modelz, "models.csv", row.names = F)

# Write back a CSV with only the predictor data for the models of interest
write.csv(predz[ModelID %in% modelz$ModelID], "predictors.csv", row.names = F)





