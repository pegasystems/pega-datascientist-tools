# Export ADM Datamart models to PMML from a dataset export

library(XML)
library(pdstools)

models <- readDSExport("Data-Decision-ADM-ModelSnapshot", srcFolder="~/Downloads", tmpFolder="tmp")
predictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot", srcFolder="~/Downloads", tmpFolder="tmp")

adm2pmml(models, predictors, verbose=T, destDir="~/Downloads", ruleNameFilter = "^(?!VerySimple).*$")
