# Generate unittests for ADM to PMML conversion from a Pega instance

library(tidyverse)
library(data.table)
library(RJDBC)
library(cdhtools)
library(jsonlite)

# To create ADM2PMML unittests:
#  - make sure there are trained ADM models in some Pega instance
#  - then, in that instance, run the models over an input dataset with "store results for later response matching" on; this
#    puts all results and payload in pxDecisionResults
#  - export and download pxDecisionResults
#  - make sure connection details are set up in this script and run this script to generate
#    1) file with all inputs and expected results (propensity)
#    2) files for the ADM data mart corresponding to this model
#    3) files for the ADM model from the (internal) factory table
#
# Then... try them out!
#  - copy the files (created under "unittests" here) to tests/testthat/data
#  - include the test in the "test_pmml.R" file
#  - run the tests (Shift-Cmd-T)
#  - debug the test
#    - the test framework will leave files in the "tmp" folder with PMML, inputs etc
#    - there are files with ".dm" and with ".json"
#    - invalid PMML errors will usually give a big stacktrace from JPMML
#    - if there's an issue, isolate it by stripping down the files to just that model/those inputs
#  - until fixed, then check in

# The ADM2PMML ruleset contains a DMSample compatible dataflow to run the SalesModel. This can easily be
# changed and used in other applications.
# See
#    "TrainModelsForPMMLTests" to train the models
#    "ScoreModelsForPMMLTests" to run the models over a small subset of the inputs



# ProductOffers strategy uses the SalesModel and can be executed over all Customers

appliesTo <- "DMOrg-DMSample-Data-Customer"
modelName <- "ExampleModelForADM2PMMLUnitTesting"

datasetExportFolder <- "~/Downloads"
dataset <- "Data-Decision-Results_pxDecisionResults"
tmpFolder <- "tmp"

#testname <- paste("DMSample",modelName,sep="-")
testname <- "testfw"

# Below is typical for Pegalabs sytems

drv <- JDBC("org.postgresql.Driver", "~/Documents/tools/jdbc-drivers/postgresql-9.2-1002.jdbc4.jar")
pg_host <- "10.60.215.90:5432"
pg_db <- "pega"
pg_user <- "pega"
pg_pwd <- "pega"

# Get directory of current script, see https://stackoverflow.com/questions/1815606/rscript-determine-path-of-the-executing-script
# NB this does not work in interactive mode

this.file <- sys.frame(tail(grep('source',sys.calls()),n=1))$ofile
this.dir <- dirname(this.file)
dest.dir <- paste(this.dir, "unittests", sep="/")

if(file.exists(dest.dir)) {
  unlink(dest.dir, recursive = T)
}
dir.create(dest.dir)
inputfileName <- paste(dest.dir, paste(testname, "input.csv", sep="_"), sep="/")
modelfileName <- paste(dest.dir, paste(testname, "modeldata.csv", sep="_"), sep="/")
predictorfileName <- paste(dest.dir, paste(testname, "predictordata.csv", sep="_"), sep="/")

# Read the pxDR export - with embedded structures for ADM inputs and decision results
srs <- readDSExport(dataset, srcFolder = datasetExportFolder, excludeComplexTypes=F)[, -"pxUpdateDateTime"]
setnames(srs, tolower(names(srs)))

# Peel off the onion for ADM inputs - currently only considering the common inputs - the others are iffy and format might change
for (i in seq(nrow(srs))) {
  commonInputs <- sapply(jsonlite::fromJSON(srs$pxadminputs[i])$commonInputs$values, function(x){return(x[["value"]])})
  for (j in seq(length(commonInputs))) {
    srs[i, (names(commonInputs)[j]) := commonInputs[j]][]
  }
}
srs[, pxadminputs := NULL]

# Expand the embedded decision results
expandedDecisions <- rbindlist(lapply(1:nrow(srs), function(i) {return(srs$pxdecisionresults[i][[1]])}), use.names = T, fill=T)[!is.na(pyPropensity)]
setnames(expandedDecisions, tolower(names(expandedDecisions)))

# Combination of flattened ADM inputs and decision results
inputset <- merge(srs[, !sapply(srs, is_list), with=F], expandedDecisions, allow.cartesian = T, by = "pxinteractionid")
setnames(inputset, tolower(names(inputset))) # not 100% sure why we need to lowercase them but the tests seem to require this

# NB Evidence is not a standard property but in the example flow it is
if ("evidence" %in% names(inputset)) {
  inputset[["Expected.Evidence"]] <- inputset$evidence
}
inputset[["Expected.Propensity"]] <- inputset$pypropensity

# Grab the models from the database

conn <- dbConnect(drv, paste("jdbc:postgresql://", pg_host,  "/", pg_db, sep=""), pg_user, pg_pwd)
dmmodels <- getModelsFromDatamart(conn, appliesTo, modelName)
dmpredictors <- getPredictorsForModelsFromDatamart(conn, dmmodels)
jsonmodels <- getModelsFromJSONTable(conn, appliesTo, modelName, verbose=T)
dbDisconnect(conn)

# Subset the input to only fields that are used in any of the models

inputs <- unique(c(sort(intersect(tolower(names(dmmodels)), names(inputset))), # context keys
                   sort(intersect(unique( tolower(dmpredictors$pypredictorname) ), names(inputset))), # predictors only those listed also in the predictor table
                   sort(names(inputset)[grepl("^Expected.", names(inputset))]))) # expected outputs
write.csv(inputset[, inputs, with=F], inputfileName, row.names = F, quote = F)

# Write the model files

write.csv(dmmodels, modelfileName, row.names = F)
write.csv(dmpredictors, predictorfileName, row.names = F)

jsonfolder <- paste(dest.dir, paste(testname, "json", sep="."), sep="/")
if(!dir.exists(jsonfolder)) {
  dir.create(jsonfolder)
}
for(n in seq(nrow(jsonmodels))) {
  jsonDumpFileName <- paste(jsonfolder,
                            paste(testname, "json", getJSONModelContextAsString(jsonmodels$pymodelpartition[n]), "json", sep="."), sep="/")
  sink(file = jsonDumpFileName, append = F)
  cat(jsonmodels$pyfactory[n])
  sink()
}

cat("Input file", inputfileName, "size", nrow(inputset), "x", ncol(inputset), fill=T)
cat("Model file", modelfileName, "size", nrow(dmmodels), fill=T)
cat("Predictors file", predictorfileName, "size", nrow(dmpredictors), fill=T)
cat("JSON factory folder", jsonfolder, "# models", nrow(jsonmodels), fill=T)
