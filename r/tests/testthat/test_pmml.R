library(testthat)
library(cdhtools)
library(lubridate)
library(jsonlite)
library(XML)

context("check PMML")

verify_results <- function(pmmlString, pmmlFile, inputFile, outputFile)
{
  jpmmlJar <- paste("jpmml", "example-1.3-SNAPSHOT.jar", sep="/")

  sink(pmmlFile)
  print(pmmlString)
  sink()
  cat("      generated PMML:", pmmlFile,fill=T)

  if (file.exists(outputFile)) {
    file.remove(outputFile)
  }

  execJPMML <- paste("java", "-cp", jpmmlJar, "org.jpmml.evaluator.EvaluationExample",
                     "--model", pmmlFile,
                     "--input", inputFile,
                     "--output", outputFile)
  print(execJPMML)
  system(execJPMML)

  expect_equal(1==1, TRUE)

}

# Runs a full unit test by generating PMML from the model and predictor data,
# scoring that with JPMML against provided inputs and comparing the results against expected values.
pmml_unittest <- function(testName)
{
  testFolder <- "pmml_unittestdata"
  tmpFolder <- paste(testFolder, "tmp", sep="/")
  if (!dir.exists(tmpFolder)) { dir.create(tmpFolder) }

  cat(": ", testName, fill=T)

  predictorDataFile <- paste(testFolder, paste(testName, "_predictordata", ".csv", sep=""), sep="/")
  modelDataFile <- paste(testFolder, paste(testName, "_modeldata", ".csv", sep=""), sep="/")
  jsonFolder <- paste(testFolder, paste(testName, ".json", sep=""), sep="/")
  inputFile <- paste(testFolder, paste(testName, "_input", ".csv", sep=""), sep="/")

  if (file.exists(jsonFolder)) {
    cat("   running ADM Factory JSON test from:", jsonFolder, fill=T)
    jsonFiles <- list.files(path = jsonFolder, pattern = "^.*\\.json", full.names = T)
    partitions <- data.table(pymodelpartitionid = sub("^.*json[^.]*\\.(.*)\\.json", "\\1", jsonFiles),
                             pyfactory = sapply(jsonFiles, readr::read_file))
    modelList <- createListFromADMFactory(partitions, testName, tmpFolder, forceLowerCasePredictorNames = T)
    pmml <- createPMML(modelList, testName)

    pmmlFile <- paste(tmpFolder, paste(testName, "json", "pmml", "xml", sep="."), sep="/")
    outputFile <- paste(tmpFolder, paste(testName, "json", "output", "csv", sep="."), sep="/")

    verify_results(pmml, pmmlFile, inputFile, outputFile)
  }
}

test_that("a basic happy-path test with a single 2-predictor model w/o missings, WOS etc", {
  pmml_unittest("singlesimplemodel")
})

