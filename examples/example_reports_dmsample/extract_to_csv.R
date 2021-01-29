library(cdhtools)
library(data.table)

dataFolder <- "../extra/example_reports_dmsample"

# Read data set export, filter on some of the Sales models and export to CSV

models <- readDSExport("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots", dataFolder)[pyConfigurationName=="SalesModel" & pyResponseCount > 1000]
write.csv(models, file = file.path(dataFolder, "models.csv"), row.names = F)

# Read data set export for the predictor binning details and export to CSV

predbinning <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots", dataFolder)[pyModelID %in% models$pyModelID]
write.csv(predbinning, file = file.path(dataFolder, "predbinning.csv"), row.names = F)
