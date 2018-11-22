library(testthat)
library(cdhtools)
library(lubridate)
library(jsonlite)
library(XML)

context("ADM2PMML")

verify_results <- function(pmmlString, pmmlFile, inputFile, outputFile)
{
  jpmmlJar <- paste("jpmml", "example-1.3-SNAPSHOT.jar", sep="/")

  sink(pmmlFile)
  print(pmmlString)
  sink()
  cat("      generated:", pmmlFile,fill=T)

  expect_true(file.exists(pmmlFile), "Generated PMML file generated without issues")

  if (file.exists(outputFile)) {
    file.remove(outputFile)
  }

  if (file.exists(inputFile)) {
    execJPMML <- paste("java", "-cp", jpmmlJar, "org.jpmml.evaluator.EvaluationExample",
                       "--model", pmmlFile,
                       "--input", inputFile,
                       "--output", outputFile)
    system(execJPMML)

    expect_true(file.exists(outputFile), paste("JPMML execution failure:", execJPMML))

    if(file.exists(outputFile)) {
      cat("      generated output:", outputFile,fill=T)

      actualResults <- fread(outputFile)
      expectedResults <- fread(inputFile)

      expect_equal(nrow(actualResults), nrow(expectedResults), info="checking length of generated output files")

      if (nrow(actualResults) == nrow(expectedResults)) {
        expect_equal(actualResults[["Propensity"]],
                     expectedResults[["Expected.Propensity"]], tolerance=1e-6, info="propensity")
        if ("Expected.Evidence" %in% names(expectedResults)) {
          expect_equal(actualResults[["Evidence"]],
                       expectedResults[["Expected.Evidence"]], tolerance=1e-6, info="evidence")
        }
      }
    }
  }
}

compareBinning <- function(predictorName, dmBinning, jsonBinning)
{
  expect_equal(nrow(dmBinning), nrow(jsonBinning), info=paste("comparing nr of bins of", predictorName))

  flds <- setdiff(names(dmBinning), c("modelid", "performance")) # TODO: performance SHOULD match
  for (f in flds) {
    expect_identical(dmBinning[[f]], jsonBinning[[f]], info=paste("comparing", f, "bins of",predictorName))
  }
}

compareCSV <- function(dmCSV, jsonCSV)
{
  dm <- fread(dmCSV)
  json <- fread(jsonCSV)

  expect_identical(names(dm), names(json),
                   info=paste("names not the same", names(dm), "[", dmCSV, "]", "vs", names(json), "[", jsonCSV, "]"))

  if (identical(names(dm), names(json))) {
    dmPreds <- sort(unique(dm$predictorname))
    jsonPreds <- sort(unique(dm$predictorname))

    expect_identical(dmPreds, jsonPreds,
                     info=paste("predictor names not the same", names(dm), "[", dmCSV, "]", "vs", names(json), "[", jsonCSV, "]"))

    # perhaps not test here but will be tested for individual predictors
    expect_identical(nrow(dm), nrow(json),
                     info=paste("binning not same length", nrow(dm), "[", dmCSV, "]", "vs", nrow(json), "[", jsonCSV, "]"))

    if (identical(dmPreds, jsonPreds)) {
      for (p in c(setdiff(dmPreds,"classifier"), intersect(dmPreds, "classifier"))) { # list classifier last
        compareBinning(p, dm[predictorname==p], json[predictorname==p])
      }
    }
  }
}

# Runs a full unit test by generating PMML from the model and predictor data,
# scoring that with JPMML against provided inputs and comparing the results against expected values.
pmml_unittest <- function(testName)
{
  testFolder <- "pmml_unittestdata"
  tmpFolder <- paste(testFolder, "tmp", sep="/")
  if (dir.exists(tmpFolder)) { unlink(tmpFolder, recursive = T) }
  dir.create(tmpFolder)

  context(paste("ADM2PMML", testName))

  predictorDataFile <- paste(testFolder, paste(testName, "_predictordata", ".csv", sep=""), sep="/")
  modelDataFile <- paste(testFolder, paste(testName, "_modeldata", ".csv", sep=""), sep="/")
  jsonFolder <- paste(testFolder, paste(testName, ".json", sep=""), sep="/")
  inputFile <- paste(testFolder, paste(testName, "_input", ".csv", sep=""), sep="/")

  if (file.exists(jsonFolder)) {
    cat("   JSON source folder:", jsonFolder, fill=T)
    jsonFiles <- list.files(path = jsonFolder, pattern = "^.*\\.json", full.names = T)
    partitions <- data.table(pymodelpartitionid = sub("^.*json[^.]*\\.(.*)\\.json", "\\1", jsonFiles),
                             pyfactory = sapply(jsonFiles, readr::read_file))
    modelList <- createListFromADMFactory(partitions, testName, tmpFolder, forceLowerCasePredictorNames = T)
    pmml <- createPMML(modelList, testName)

    pmmlFile <- paste(tmpFolder, paste(testName, "json", "pmml", "xml", sep="."), sep="/")
    outputFile <- paste(tmpFolder, paste(testName, "json", "output", "csv", sep="."), sep="/")

    verify_results(pmml, pmmlFile, inputFile, outputFile)
  }

  if (file.exists(predictorDataFile)) {
    cat("   Datamart predictor data:", predictorDataFile, fill=T)

    predictorData <- fread(predictorDataFile)
    if (file.exists(modelDataFile)) {
      modelData <- read.csv(modelDataFile, stringsAsFactors = F) # fread adds extra double-quotes for JSON strings, thus using simple read.csv
      modelList <- createListFromDatamart(predictorData, testName, tmpFolder, modelData, lowerCasePredictors = T)
    } else {
      modelList <- createListFromDatamart(predictorData, testName, tmpFolder, lowerCasePredictors = T)
    }

    pmml <- createPMML(modelList, testName)

    pmmlFile <- paste(tmpFolder, paste(testName, "dm", "pmml", "xml", sep="."), sep="/")
    outputFile <- paste(tmpFolder, paste(testName, "dm", "output", "csv", sep="."), sep="/")

    verify_results(pmml, pmmlFile, inputFile, outputFile)
  }

  if (file.exists(jsonFolder) & file.exists(predictorDataFile)) {
    cat("   compare binning of all predictors in intermediate CSVs", fill=T)
    dmCSVFiles <- list.files(path = tmpFolder, pattern = paste("^", testName, "\\.dm.*\\.csv", sep=""), full.names = T)
    jsonCSVFiles <- list.files(path = tmpFolder, pattern = paste("^", testName, "\\.json.*\\.csv", sep=""), full.names = T)
    outputFiles <- list.files(path = tmpFolder, pattern = paste("output\\.csv$", sep=""), full.names = T)
    dmBinningFiles <- setdiff(dmCSVFiles, outputFiles)
    jsonBinningFiles <- setdiff(jsonCSVFiles, outputFiles)

    dmBinningFilesModelNames <- sub("^.*\\.dm\\.(.*)", "\\1", dmBinningFiles) # this is only for better error reporting
    jsonBinningFilesModelNames <- sub("^.*\\.json\\.(.*)", "\\1", jsonBinningFiles)

    expect_equal(length(dmBinningFiles), length(jsonBinningFiles),
                 info = paste("different nr of intermediate CSV files: dm:", length(dmBinningFiles),
                              "json:", length(jsonBinningFiles),
                              "mismatches:", paste(c(setdiff(dmBinningFilesModelNames, jsonBinningFilesModelNames),
                                                     setdiff(jsonBinningFilesModelNames, dmBinningFilesModelNames)),
                                                   collapse = "; ")))

    if (length(dmBinningFiles) == length(jsonBinningFiles)) {
      dmBinningFiles <- sort(dmBinningFiles)
      jsonBinningFiles <- sort(jsonBinningFiles)
      for (i in seq(length(dmBinningFiles))) {
        compareCSV(dmBinningFiles[i], jsonBinningFiles[i]) # assume the files are named so their alphabetical order is the same
      }
    }
  }

}

test_that("a basic happy-path test with a single 2-predictor model w/o missings, WOS etc", {
  pmml_unittest("singlesimplemodel")
})
test_that("a basic model consisting of 2 models, also providing new partition key values", {
  pmml_unittest("simplemultimodel")
})
test_that("check symbolic binning with WOS", {
  pmml_unittest("multimodel_wos")
})
test_that("checks flexible context keys (wich require to parse the JSON inside the pyName field), also some invalid models with empty keys", {
  pmml_unittest("flexandinvalidkeys")
})
test_that("highlights an issue with the way ADM defines the smoothing factor in the score calculation - one of the tests in this will be failing if defining the smoothing factor in a naive way", {
  pmml_unittest("smoothfactorissue") # previously called "precisionissue"
})
test_that("At least some of the models use a predictor with a RESIDUAL bin", {
  pmml_unittest("residualtest")
})
test_that("Models with different evidence for predictors (added/removed)", {
  pmml_unittest("unequalevidence")
})
test_that("Issue with creating PMML from internal JSON", {
  pmml_unittest("issue-4-singlebinpredictor")
})
test_that("Issue with a single classifier bin", {
  pmml_unittest("singleclassifierbin")
})
test_that("Test the test generator", {
  pmml_unittest("ExampleModelForADM2PMMLUnitTesting")
})

# TODO add a few more JSON vs DM tests

# TODO add specific Scorecard explanation tests

test_that("PMML generation from DM exports", {
  context("PMML from Datamart Exports")

  allModels <- readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots","dsexports")
  allPredictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_AllPredictorSnapshots","dsexports")

  pmmlFiles <- adm2pmml(allModels, allPredictors)

  expect_equal(length(pmmlFiles), 3)
  expect_equal(names(pmmlFiles)[1], "./BannerModel.pmml")
  expect_equal(names(pmmlFiles)[2], "./SalesModel.pmml")
  expect_equal(names(pmmlFiles)[3], "./VerySimpleSalesModel.pmml")
  expect_equal(length(pmmlFiles[[1]]), 5)
  expect_equal(length(pmmlFiles[[2]]), 5)
  expect_equal(length(pmmlFiles[[3]]), 5)

  expect_equal(length(adm2pmml(allModels, allPredictors, ruleNameFilter="^(?!VerySimple).*$", appliesToFilter="PMDSM")), 2)
  expect_equal(length(adm2pmml(allModels, allPredictors, appliesToFilter="DMSample")), 0)
})
