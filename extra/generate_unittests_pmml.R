# Generate ADM to PMML unittests

# Steps to generate unittest files by pulling data from both the ADM datamart and the internal JSON representation of the models.
#
# 1) First create the models:
#    In PRPC, create the ADM models by configuring the ADM models & strategy as desired and run CreateModelsForPMMLTesting. 
# 2) Then create the test data:
#    When the models are created (check the landing page for no outstanding responses) update the source data set as 
#    desired (for the inputs) and run CreatePMMLExportUnitTestData. This runs the same strategy and copies the SRs to an
#    embedded property. The input data is stored in a dataset and the resulting strategy results (with ADM propensities)
#    in another dataset. These two will be joined in this R script to form a single input dataset with expected propensity
#    for each of the different SRs.
# 3) Export this data:
#    Export both of the data sets PMMLUnitTestInputs (containing the input data) and 
#    PMMLUnitTestingResults (containing the ADM propensities for all SRs) & download the exported CSV files
# 4) Update the code in this script to pull in the right models 
# 5) Make sure the models are up to date in the datamart (press Refresh if necessary)
# 6) Then run this script to create (temp) files with all the results. Move/rename this as appropriate, run the newly
#    created unit tests (in the run script) and put in GIT when happy.

# The name of the test to generate:
data_appliesto <- "Exam-PMDSM-Data-Customer"
results_appliesto <- "Exam-PMDSM-Data-Customer-WithSR"
admmodelname <- "BigModel"
testname <- admmodelname
datasetexportfolder <- "~/Downloads"

library(tidyverse)
library(data.table)
library(RJDBC)

# get directory of current script, see https://stackoverflow.com/questions/1815606/rscript-determine-path-of-the-executing-script
this.file <- sys.frame(tail(grep('source',sys.calls()),n=1))$ofile
this.dir <- dirname(this.file)

source(paste(this.dir,"..","utils","pmdsm_utils.R",sep="/"))
source(paste(this.dir,"exportUtilsDatamart.R",sep="/"))
source(paste(this.dir,"exportUtilsJSON.R",sep="/"))

dest.dir <- paste(this.dir, "unittests", sep="/")
tmp.dir <- "tmp"
if (!dir.exists(tmp.dir)) { dir.create(tmp.dir) }

drv <- JDBC("org.postgresql.Driver", "~/Documents/tools/jdbc-drivers/postgresql-9.2-1002.jdbc4.jar")
pg_host <- "localhost:5432"
pg_db <- "pega731"
pg_user <- "pega"
pg_pwd <- "pega"

conn <- dbConnect(drv, paste("jdbc:postgresql://", pg_host,  "/", pg_db, sep=""), pg_user, pg_pwd)
dmmodels <- getModelsForClassFromDatamart(conn, data_appliesto, admmodelname)
dmpredictors <- getPredictorsForModelsFromDatamart(conn, dmmodels)
jsonmodels <- getModelsFromJSONTable(conn, data_appliesto, admmodelname)
dbDisconnect(conn)

inputs <- readDSExport(paste(data_appliesto, "PMMLUnitTestInputs", sep="_"), datasetexportfolder, tmpFolder=tmp.dir) # input fields
outputs <- readDSExport(paste(results_appliesto, "PMMLUnitTestingResults", sep="_"), datasetexportfolder, tmpFolder=tmp.dir) # all strategy results
setnames(inputs, tolower(names(inputs)))
setnames(outputs, tolower(names(outputs)))

stdKeys <- c("pyIssue","pyGroup","pyName","pyChannel","pyDirection","pyTreatment")
inputset <- merge( outputs[, intersect(names(outputs), c("pysubjectid", tolower(stdKeys), "pypropensity", "evidence")), with=F], 
                   inputs[, setdiff(names(inputs), c("pxobjclass")), with=F],
                   by.x = "pysubjectid", by.y = "customerid")
inputset[["Expected.Propensity"]] <- inputset$pypropensity
inputset[["Expected.Evidence"]] <- inputset$evidence

inputfileName <- paste(dest.dir, paste(testname, "input.csv", sep="_"), sep="/")
write.csv(inputset, inputfileName, row.names = F, quote = F)

modelfileName <- paste(dest.dir, paste(testname, "modeldata.csv", sep="_"), sep="/")
predictorfileName <- paste(dest.dir, paste(testname, "predictordata.csv", sep="_"), sep="/")
write.csv(dmmodels, modelfileName, row.names = F)
write.csv(dmpredictors, predictorfileName, row.names = F)

jsonfolder <- paste(dest.dir, paste(testname, "json", sep="."), sep="/")
if(!dir.exists(jsonfolder)) {
  dir.create(jsonfolder)
}
for(n in seq(nrow(jsonmodels))) {
  jsonDumpFileName <- paste(jsonfolder, paste(testname, "json", getJSONModelContextAsString(jsonmodels$pymodelpartition[n]), "json", sep="."), sep="/")
  sink(file = jsonDumpFileName, append = F)
  cat(jsonmodels$pyfactory[n])
  sink()
}

cat("Input file", inputfileName, "size", nrow(inputset), "x", ncol(inputset), fill=T)
cat("Model file", modelfileName, "size", nrow(dmmodels), fill=T)
cat("Predictors file", predictorfileName, "size", nrow(dmpredictors), fill=T)
cat("JSON factory folder", jsonfolder, "# models", nrow(jsonmodels), fill=T)
