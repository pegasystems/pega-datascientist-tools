library(testthat)
# library(cdhtools)
# library(lubridate)
# library(jsonlite)
# library(XML)

context("ADM2PMML")

run_jpmml <- function(pmmlFile, inputFile, outputFile)
{
  jpmmlJar <- file.path("jpmml", "example-1.4-SNAPSHOT.jar")

  if (file.exists(outputFile)) {
    file.remove(outputFile)
  }

  execJPMML <- paste("java", "-cp", jpmmlJar, "org.jpmml.evaluator.EvaluationExample",
                     "--model", pmmlFile,
                     "--input", inputFile,
                     "--output", outputFile)
  system(execJPMML)

  expect_true(file.exists(outputFile), paste("JPMML execution failure, no output file:", execJPMML))
  expect_true(0 == file.access(outputFile, mode=4), paste("JPMML execution failure, cannot read output file:", execJPMML))
}

verify_results <- function(pmmlString, pmmlFile, inputFile, outputFile)
{
  sink(pmmlFile)
  print(pmmlString)
  sink()
  cat("      generated:", pmmlFile,fill=T)

  expect_true(file.exists(pmmlFile), "Generated PMML file generated without issues")

  if (file.exists(inputFile)) {
    run_jpmml(pmmlFile, inputFile, outputFile)

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

compareBinning <- function(predictorName, dmBinning, jsonBinning, dmCSV, jsonCSV)
{
  expect_equal(nrow(dmBinning[BinType!="SYMBOL"]), nrow(jsonBinning[BinType!="SYMBOL"]),
               info=paste("nr of bins of", predictorName, ":",
                          nrow(dmBinning[BinType!="SYMBOL"]), "[", dmCSV, "]", "vs",
                          nrow(jsonBinning[BinType!="SYMBOL"]), "[", jsonCSV, "]"))

  # this seems to be model performance, not predictor performance
  expect_equal(unique(dmBinning$Performance), unique(jsonBinning$Performance), tolerance=1e-6,
               info=paste("Performance",
                          "[", dmCSV, "]", "vs", "[", jsonCSV, "]"))

  if (nrow(dmBinning[BinType!="SYMBOL"]) == nrow(jsonBinning[BinType!="SYMBOL"])) {
    flds <- setdiff(names(dmBinning), c("ModelID", "Performance"))
    for (f in flds) {
      if (nrow(dmBinning) == nrow(jsonBinning)) {
        # full comparison
        expect_equal(dmBinning[[f]], jsonBinning[[f]], tolerance=1e-6,
                     info=paste("comparing the", f, "bins of",predictorName,
                                "[", dmCSV, "]", "vs", "[", jsonCSV, "]"))
      } else {
        # exclude the SYMBOL matches from the comparison
        expect_equal(dmBinning[BinType!="SYMBOL"][[f]], jsonBinning[BinType!="SYMBOL"][[f]], tolerance=1e-6,
                     info=paste("comparing (except SYMBOL matches)", f, "bins of",predictorName,
                                "[", dmCSV, "]", "vs", "[", jsonCSV, "]"))
      }
    }
  }
}

compareCSV <- function(dmCSV, jsonCSV)
{
  dm <- fread(dmCSV)
  json <- fread(jsonCSV)

  expect_identical(names(dm), names(json),
                   info=paste("names not the same", names(dm),
                              "[", dmCSV, "]", "vs", names(json), "[", jsonCSV, "]"))

  if (identical(names(dm), names(json))) {
    dmPreds <- sort(unique(dm$PredictorName))
    jsonPreds <- sort(unique(dm$PredictorName))

    expect_identical(dmPreds, jsonPreds,
                     info=paste("predictor names not the same", names(dm), "[", dmCSV, "]", "vs", names(json), "[", jsonCSV, "]"))

    # exclude when PredictorType = "SYMBOLIC" as the factory representation contains all symbols
    # while this is truncated in the DM - this may not be a problem per se, if so then propensities will show this

    expect_identical(nrow(dm[PredictorType!="SYMBOLIC"]), nrow(json[PredictorType!="SYMBOLIC"]),
                     info=paste("total number of bins for all non-symbolic predictors not the same",
                                nrow(dm[PredictorType!="SYMBOLIC"]), "[", dmCSV, "]", "vs",
                                nrow(json[PredictorType!="SYMBOLIC"]), "[", jsonCSV, "]"))

    if (identical(dmPreds, jsonPreds)) {
      for (p in c(setdiff(dmPreds,"Classifier"),
                  intersect(dmPreds, "Classifier"))) { # list classifier last
        compareBinning(p, dm[PredictorName==p], json[PredictorName==p], dmCSV, jsonCSV)
      }
    }
  }
}

# Runs a full unit test by generating PMML from the model and predictor data,
# scoring that with JPMML against provided inputs and comparing the results against expected values.

pmml_unittest <- function(testName)
{
  testFolder <- "d"
  #outFolder <- tempdir()
  outFolder <- paste(testFolder, "tmp", sep="/")
  if (!dir.exists(outFolder)) dir.create(outFolder)

  context(paste("ADM2PMML", testName))

  fileArchive <- file.path(testFolder, paste(testName, ".zip", sep=""))
  if (file.exists(fileArchive)) {
    testFolder <- file.path(testFolder, testName)
    jsonFolder <- file.path(testFolder, paste(testName, ".json", sep=""))
    cat("   Extracted test files from archive:", fileArchive, "to", testFolder, "and", jsonFolder, fill=T)
    flz <- unzip(fileArchive, list=T)$Name
    unzip(fileArchive, files=flz[!grepl(".json$", flz)], exdir=testFolder, junkpaths = T)
    unzip(fileArchive, files=flz[grepl(".json$", flz)], exdir=jsonFolder, junkpaths = T)
  }

  predictorDataFile <- file.path(testFolder, paste(testName, "_predictordata", ".csv", sep=""))
  modelDataFile <- file.path(testFolder, paste(testName, "_modeldata", ".csv", sep=""))
  jsonFolder <- file.path(testFolder, paste(testName, ".json", sep=""))
  inputFile <- file.path(testFolder, paste(testName, "_input", ".csv", sep=""))

  if (file.exists(jsonFolder)) {
    cat("   JSON source folder:", jsonFolder, fill=T)
    jsonFiles <- list.files(path = jsonFolder, pattern = "^.*\\.json", full.names = T)
    partitions <- data.table(pymodelpartitionid = sub("^.*json[^.]*\\.(.*)\\.json", "\\1", jsonFiles),
                             pyfactory = sapply(jsonFiles, readr::read_file))
    modelList <- normalizedBinningFromADMFactory(partitions, testName, outFolder,
                                          forceLowerCasePredictorNames=T)

    pmml <- createPMML(modelList, testName)

    pmmlFile <- file.path(outFolder, paste(testName, "json", "pmml", "xml", sep="."))
    outputFile <- file.path(outFolder, paste(testName, "json", "output", "csv", sep="."))

    verify_results(pmml, pmmlFile, inputFile, outputFile)
  }

  if (file.exists(predictorDataFile)) {
    cat("   Datamart predictor data:", predictorDataFile, fill=T)

    predictorData <- fread(predictorDataFile)
    standardizeFieldCasing(predictorData)

    # forcing lowercase predictornames here
    predictorData[, PredictorName := tolower(PredictorName)]

    if (file.exists(modelDataFile)) {
      modelData <- data.table(read.csv(modelDataFile, stringsAsFactors = F)) # fread adds extra double-quotes for JSON strings, thus using simple read.csv
      standardizeFieldCasing(modelData)
      modelList <- normalizedBinningFromDatamart(predictorData, testName, outFolder, modelData, useLowercaseContextKeys=TRUE)
    } else {
      modelList <- normalizedBinningFromDatamart(predictorData, testName, outFolder, useLowercaseContextKeys=TRUE)
    }

    pmml <- createPMML(modelList, testName)

    pmmlFile <- file.path(outFolder, paste(testName, "dm", "pmml", "xml", sep="."))
    outputFile <- file.path(outFolder, paste(testName, "dm", "output", "csv", sep="."))

    verify_results(pmml, pmmlFile, inputFile, outputFile)
  }

  if (file.exists(jsonFolder) & file.exists(predictorDataFile)) {
    cat("   compare binning of all predictors in intermediate CSVs", fill=T)
    dmCSVFiles <- list.files(path = outFolder, pattern = paste("^", testName, "\\.dm.*\\.csv", sep=""), full.names = T)
    jsonCSVFiles <- list.files(path = outFolder, pattern = paste("^", testName, "\\.json.*\\.csv", sep=""), full.names = T)
    outputFiles <- list.files(path = outFolder, pattern = paste("output\\.csv$", sep=""), full.names = T)
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
        compareCSV(dmBinningFiles[i],
                   jsonBinningFiles[i]) # assume the files are named so their alphabetical order is the same
      }
    }
  }

  if (file.exists(fileArchive)) {
    cat("   Cleaning up extracted test files from archive:", fileArchive, "in", testFolder, fill=T)
    unlink(testFolder, recursive = T)
  }

}

test_that("Test running the JPMML engine with a simple model", {
  context("Test running the JPMML engine with a simple model")

  pmmlFile <- file.path("jpmml", "single_iris_logreg.xml")
  inputFile <- file.path("jpmml", "Iris.csv")
  outFile <- file.path(tempdir(), "iris_out.csv")

  expect_equal(nrow(fread(inputFile)), 150)

  run_jpmml(pmmlFile, inputFile, outFile)
  #cat("", file=outFile, append=T, fill=T)

  expect_true(0 == file.access(outFile, mode=4))

  scores <- fread(outFile)
  expect_equal(nrow(scores), 150)
})

test_that("Public functions from JSON utils", {
  context("Public JSON PMML utility functions")

  expect_equal(getJSONModelContextAsString("{\"partition\":{\"pyIssue\":\"Sales\", \"pyGroup\":\"Cards\"}}"),
               "pygroup_Cards_pyissue_Sales")

  # TODO: perhaps exercise the peeling of JSON from pyName here as well
})

# PMML Tests with ADM Datamart data

test_that("Simple ADM model with 2 actions, also providing new partition key values", {
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

test_that("Use of a predictor with a RESIDUAL bin", {
  pmml_unittest("residualtest")
})

test_that("Missing input values", {
  # NB example-1.4-SNAPSHOT.jar reads the CSV slightly different than previously,
  # now need explicit NA to indicate missing value
  pmml_unittest("missingvalues")
})

test_that("Models with different evidence for predictors (added/removed)", {
  pmml_unittest("unequalevidence")
})

test_that("Issue with creating PMML from internal JSON", {
  pmml_unittest("issue-4-singlebinpredictor")
})

# Tests using JSON factory data

test_that("Test the test generator", {
  pmml_unittest("testfw")
})

test_that("Simple but deeper dive", {
  pmml_unittest("deeperdive")
})

test_that("Issue with a single classifier bin", {
  pmml_unittest("singleclassifierbin")
})


# Check user facing wrapper function

test_that("Wrapper function using DM exports", {
  context("Wrapper function adm2pmml")

  dm <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All",
                    "Data-Decision-ADM-PredictorBinningSnapshot_All",
                    "dsexports")

  pmmlFiles <- adm2pmml(dm)

  expect_equal(length(pmmlFiles), 3)
  expect_equal(names(pmmlFiles)[1], "./BannerModel.pmml")
  expect_equal(names(pmmlFiles)[2], "./SalesModel.pmml")
  expect_equal(names(pmmlFiles)[3], "./VerySimpleSalesModel.pmml")
  expect_equal(length(pmmlFiles[[1]]), 5)
  expect_equal(length(pmmlFiles[[2]]), 5)
  expect_equal(length(pmmlFiles[[3]]), 5)

  dm2 <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All",
                    "Data-Decision-ADM-PredictorBinningSnapshot_All",
                    "dsexports",
                    filterModelData = function(mdls) {return(mdls[grepl("^(?!VerySimple).*$", ConfigurationName, perl=T)])})

  expect_equal(length(adm2pmml(dm2)), 2)

  dm3 <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All",
                     "Data-Decision-ADM-PredictorBinningSnapshot_All",
                     "dsexports",
                     filterModelData = function(mdls) {data.table()})

  expect_equal(length(adm2pmml(dm3)), 0)
})

# Reason codes

scorePMMLWithReasonCodes <- function(explanations)
{
  testFolder <- "d"
  # outFolder <- tempdir()
  outFolder <- paste(testFolder, "tmp2", sep="/")
  if (!dir.exists(outFolder)) dir.create(outFolder)

  # Convert the simplest model to PMML including reason code options
  predData <- fread(file.path(testFolder, "deeperdive_predictordata.csv"))
  predData[, responsecount := 0]
  dummyModelData <- data.table(pymodelid = unique(predData$pymodelid),
                               pyconfigurationname = c("simplemodel"),
                               pyname = "Dummy",
                               pyissue = "Dummy",
                               pychannel = "Dummy",
                               pygroup = "Dummy",
                               pydirection = "Dummy",
                               pytotalpredictors = 0,
                               pyactivepredictors = 0,
                               pysnapshottime = unique(predData$pysnapshottime),
                               pypositives = 0,
                               pynegatives = 0,
                               pyresponsecount = 0,
                               pyperformance = 0)

  pmmlFile <- adm2pmml(ADMDatamart(dummyModelData, predData),
                        destDir = outFolder,
                       explanations = explanations)

  expect_equal(length(pmmlFile), 1)
  expect_equal(basename(names(pmmlFile)), "simplemodel.pmml")

  # Run with inputs. The inputs include the same 3 cases that are detailed in the deeper dive Excel sheet
  run_jpmml(file.path(outFolder, "simplemodel.pmml"),
            file.path(testFolder, "deeperdive_inputs_reasoncodetests.csv"),
            file.path(outFolder, "deeperdive_output.csv"))

  # Check the outputs contain reason codes
  expect_true(0 == file.access(file.path(outFolder, "deeperdive_output.csv"), mode=4))
  output <- fread(file = file.path(outFolder, "deeperdive_output.csv"))

  # Drop duplicate columns. For some reason the JPMML runner repeats many.
  outputNames <- names(output)
  output[, which(sapply(2:length(outputNames), function(i){ outputNames[i] %in% outputNames[1:(i-1)] })) := NULL ]

  # By default 3 reason codes. Note the output repeats some of the fields.
  expect_equal(sum(startsWith(names(output), "Explain-")), explanations$nrExplanations)

  # Flatten all reason codes. Each reason code should have 6 elements:
  # First predictor name and bin label. Then four values for score, min
  # average and max, all four prefixed with a label, e.g. "score-100".
  getValueForReasonKVP <- function(x) { return(gsub(".*=(.*)", "\\1", x)) }
  reasonKVPnames <- c("predictor", "value", "score", "min", "avg", "max")
  for (i in seq_len(explanations$nrExplanations)) {
    output[, paste(paste0("r", i), reasonKVPnames, sep="_") := lapply(tstrsplit(output[[paste0("Explain-", i)]], split="|", fixed=T),
                                                                      getValueForReasonKVP)]
  }
  # output[, paste("r2", reasonKVPnames, sep="_") := lapply(tstrsplit(`Explain-2`, split="|", fixed=T), getValueForReasonKVP)]

  # Generic tests
  if (explanations$nrExplanations > 0) {
    expect_true(all(output[`Explain-1` != "N/A"][, (r1_score >= r1_min) & (r1_score <= r1_max)]), "All scores should be between min and max")
    expect_true(all(output[`Explain-1` != "N/A"][, (r1_avg >= r1_min) & (r1_avg <= r1_max)]), "All average scores should be between min and max")
  }

  return(output)
}

test_that("Scorecard reason codes", {
  # context("Scorecard reason codes")

  scores <- scorePMMLWithReasonCodes(reasonCodesAboveMin(3))
  write.csv(scores, file.path("d", "tmp2", "deeperdive_output_3_above_min.csv"))
  expect_equal(scores$r1_predictor, c(rep("Country",5), "Age")) # Age only shows up if Country is missing when using points above minimum
  expect_equal(scores$r2_predictor, c(rep("Age",5), "Country"))
  expect_equal(scores$r3_predictor, rep("N/A",6))

  scores <- scorePMMLWithReasonCodes(reasonCodesBelowMax(0))
  write.csv(scores, file.path("d", "tmp2", "deeperdive_output_3_no_reasoncodes.csv"))
  expect_false(any(startsWith(names(scores), "Explain-")))

  scores <- scorePMMLWithReasonCodes(reasonCodesBelowMax(5))
  write.csv(scores, file.path("d", "tmp2", "deeperdive_output_3_below_max.csv"))
  expect_equal(scores$r1_predictor, c("Country", "Age", "Country", "Country", "Country", "Country"))
  expect_equal(scores$r2_predictor, c("Age", "Country", "Age", "Age", "Age", "Age"))

  # For points relative to mean we need to evaluate above and below separately
  scores <- scorePMMLWithReasonCodes(reasonCodesAboveMean(2))
  write.csv(scores, file.path("d", "tmp2", "deeperdive_output_3_above_mean.csv"))
  expect_equal(scores$r1_predictor, c("Country", "Country", "N/A", "Country", "Country", "Age"))
  expect_equal(scores$r2_predictor, c("N/A", "Age", "N/A", "N/A", "N/A", "N/A"))

  scores <- scorePMMLWithReasonCodes(reasonCodesBelowMean(2))
  write.csv(scores, file.path("d", "tmp2", "deeperdive_output_3_below_mean.csv"))
  expect_equal(scores$r1_predictor, c("Age", "N/A", "Age", "Age", "Age", "Country"))
  expect_equal(scores$r2_predictor, c("N/A", "N/A", "Country", "N/A", "N/A", "N/A"))
})

