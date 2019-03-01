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

compareBinning <- function(predictorName, dmBinning, jsonBinning, dmCSV, jsonCSV)
{
  expect_equal(nrow(dmBinning[bintype!="SYMBOL"]), nrow(jsonBinning[bintype!="SYMBOL"]),
               info=paste("nr of bins of", predictorName, ":",
                          nrow(dmBinning[bintype!="SYMBOL"]), "[", dmCSV, "]", "vs",
                          nrow(jsonBinning[bintype!="SYMBOL"]), "[", jsonCSV, "]"))

  # this seems to be model performance, not predictor performance
  expect_equal(unique(dmBinning$performance), unique(jsonBinning$performance), tolerance=1e-6,
               info=paste("performance",
                          "[", dmCSV, "]", "vs", "[", jsonCSV, "]"))

  if (nrow(dmBinning[bintype!="SYMBOL"]) == nrow(jsonBinning[bintype!="SYMBOL"])) {
    flds <- setdiff(names(dmBinning), c("modelid", "performance"))
    for (f in flds) {
      if (nrow(dmBinning) == nrow(jsonBinning)) {
        # full comparison
        expect_equal(dmBinning[[f]], jsonBinning[[f]], tolerance=1e-6,
                     info=paste("comparing the", f, "bins of",predictorName,
                                "[", dmCSV, "]", "vs", "[", jsonCSV, "]"))
      } else {
        # exclude the SYMBOL matches from the comparison
        expect_equal(dmBinning[bintype!="SYMBOL"][[f]], jsonBinning[bintype!="SYMBOL"][[f]], tolerance=1e-6,
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
    dmPreds <- sort(unique(dm$predictorname))
    jsonPreds <- sort(unique(dm$predictorname))

    expect_identical(dmPreds, jsonPreds,
                     info=paste("predictor names not the same", names(dm), "[", dmCSV, "]", "vs", names(json), "[", jsonCSV, "]"))

    # exclude when predictortype = "SYMBOLIC" as the factory representation contains all symbols
    # while this is truncated in the DM - this may not be a problem per se, if so then propensities will show this

    expect_identical(nrow(dm[predictortype!="SYMBOLIC"]), nrow(json[predictortype!="SYMBOLIC"]),
                     info=paste("total number of bins for all non-symbolic predictors not the same",
                                nrow(dm[predictortype!="SYMBOLIC"]), "[", dmCSV, "]", "vs",
                                nrow(json[predictortype!="SYMBOLIC"]), "[", jsonCSV, "]"))

    if (identical(dmPreds, jsonPreds)) {
      for (p in c(setdiff(dmPreds,"classifier"), intersect(dmPreds, "classifier"))) { # list classifier last
        compareBinning(p, dm[predictorname==p], json[predictorname==p], dmCSV, jsonCSV)
      }
    }
  }
}

# Runs a full unit test by generating PMML from the model and predictor data,
# scoring that with JPMML against provided inputs and comparing the results against expected values.
pmml_unittest <- function(testName)
{
  testFolder <- "adm2pmml-testdata"
  tmpFolder <- paste(testFolder, "tmp", sep="/")
  if (!dir.exists(tmpFolder)) dir.create(tmpFolder)

  context(paste("ADM2PMML", testName))

  fileArchive <- paste(testFolder, paste(testName, ".zip", sep=""), sep="/")
  if (file.exists(fileArchive)) {
    testFolder <- paste(testFolder, testName, sep="/")
    jsonFolder <- paste(testFolder, paste(testName, ".json", sep=""), sep="/")
    cat("   Extracted test files from archive:", fileArchive, "to", testFolder, "and", jsonFolder, fill=T)
    flz <- unzip(fileArchive, list=T)$Name
    unzip(fileArchive, files=flz[!grepl(".json$", flz)], exdir=testFolder, junkpaths = T)
    unzip(fileArchive, files=flz[grepl(".json$", flz)], exdir=jsonFolder, junkpaths = T)
  }

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

  if (file.exists(fileArchive)) {
    cat("   Cleaning up extracted test files from archive:", fileArchive, "in", testFolder, fill=T)
    unlink(testFolder, recursive = T)
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
test_that("Use of a predictor with a RESIDUAL bin", {
  pmml_unittest("residualtest")
})
test_that("Missing input values", {
  # this started happening after upgrade example-1.3-SNAPSHOT.jar to example-1.4-SNAPSHOT.jar
  pmml_unittest("missingvalues")
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
  pmml_unittest("testfw")
})

# TODO add specific Scorecard explanation tests

test_that("Public functions from JSON utils", {
  context("Public JSON PMML utility functions")

  expect_equal(getJSONModelContextAsString("{\"partition\":{\"pyIssue\":\"Sales\", \"pyGroup\":\"Cards\"}}"),
               "pygroup_Cards_pyissue_Sales")

})

test_that("PMML generation from DM exports", {
  context("PMML from Datamart Exports")

  allModels <- readDSExport("Data-Decision-ADM-ModelSnapshot_All","dsexports")
  allPredictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_All","dsexports")

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

# Helper function to get a list of the reported and actual model performance for a list of adaptive models. The reported
# performance is part of the data passed in. The actual performance is calculated by taking into account the possible range
# of the model scores.
getModelPerformanceOverview <- function(dmModels, dmPredictors)
{
  # Similar steps for initial filtering and processing as in "adm2pmml" function
  setnames(dmModels, tolower(names(dmModels)))
  setnames(dmPredictors, tolower(names(dmPredictors)))
  modelList <- createListFromDatamart(dmPredictors[pymodelid %in% dmModels$pymodelid],
                                      fullName="Sales Models DMSample", tmpFolder=NULL, modelsForPartition=dmModels)

  # Name, response count and reported performance obtained from the data directly. For the score min/max using the utility function
  # that summarizes the predictor bins into a table ("scaling") with the min and max weight per predictor. These weights are the
  # normalized log odds and score min/max is found by adding them up.
  perfOverview <- data.table( pyname = sapply(modelList, function(m) {return(m$context$pyname)}),
                              performance = sapply(modelList, function(m) {return(m$binning$performance[1])}),
                              correct_performance = NA, # placeholder
                              responses = sapply(modelList, function(m) {return(m$binning$totalpos[1] + m$binning$totalneg[1])}),
                              score_min = sapply(modelList, function(m) {binz <- copy(m$binning); scaling <- setBinWeights(binz); return(sum(scaling$minWeight))}),
                              score_max = sapply(modelList, function(m) {binz <- copy(m$binning); scaling <- setBinWeights(binz); return(sum(scaling$maxWeight))}))

  classifiers <- lapply(modelList, function(m) { return (m$binning[predictortype == "CLASSIFIER"])})

  findClassifierBin <- function( classifierBins, score )
  {
    if (nrow(classifierBins) == 1) return(1)

    return (1 + findInterval(score, classifierBins$binupperbound[1 : (nrow(classifierBins) - 1)]))
  }

  perfOverview$nbins <- sapply(classifiers, nrow)
  perfOverview$score_bin_min <- sapply(seq(nrow(perfOverview)), function(n) { return( findClassifierBin( classifiers[[n]], perfOverview$score_min[n]))} )
  perfOverview$score_bin_max <- sapply(seq(nrow(perfOverview)), function(n) { return( findClassifierBin( classifiers[[n]], perfOverview$score_max[n]))} )
  perfOverview$correct_performance <- sapply(seq(nrow(perfOverview)), function(n) { return( auc_from_bincounts(classifiers[[n]]$binpos[perfOverview$score_bin_min[n]:perfOverview$score_bin_max[n]],
                                                                                                              classifiers[[n]]$binneg[perfOverview$score_bin_min[n]:perfOverview$score_bin_max[n]] )) })
  setorder(perfOverview, pyname)

  return(perfOverview)
}

# This test confirms BUG-417860 stating that ADM performance numbers can be overly optimistic in the beginning.
test_that("Score Ranges", {
  context("Verify range of score distribution")

  # The source of the data tested in this test is taken by a dataset export of the two Datamart tables (in Pega 8.3), then
  # subsetting this to just the models we're interested in. The resulting data.table objects are saved. To
  # repeat this exercise, execute the following lines:

      # dmModels <- readDSExport("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20190227T140454_GMT.zip","~/Downloads") # "dsexports"
      # dmPredictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20190227T140628_GMT.zip","~/Downloads")
      #
      # dmModels <- dmModels[pyConfigurationName %in% c("SalesModel", "BundleModelIndependent"), !grepl("^px|pz", names(dmModels)), with=F]
      # dmPredictors <- dmPredictors[pyModelID %in% dmModels$pyModelID, !grepl("^px|pz", names(dmPredictors)), with=F]
      #
      # save(dmModels, dmPredictors, file="tests/testthat/dsexports/test_score_ranges.RData")

  load("dsexports/test_score_ranges.RData")

  # First bunch of models are from the "independent" bundle strategy. There are no customer level predictors in this model and there
  # are also no parameterized predictors for the sequencing. No predictors at all should give all models performance 0.5 but due to
  # BUG-417860 this is not the case. Therefore commented out the first test.
  perfOverviewIndependentBundleNoPredictors <- getModelPerformanceOverview(dmModels[pyConfigurationName=="BundleModelIndependent"], dmPredictors)
  # expect_equal(perfOverviewIndependentBundleNoPredictors$performance, rep(0.5, nrow(perfOverviewIndependentBundleNoPredictors)))
  expect_equal(perfOverviewIndependentBundleNoPredictors$correct_performance, rep(0.5, nrow(perfOverviewIndependentBundleNoPredictors)))

  # The Sales Model is an initialized DMSample system. Some of the models have too optimistic performance numbers early on in their
  # life due to the same bug mentioned above.
  perfOverviewSalesModel <- getModelPerformanceOverview(dmModels[pyConfigurationName=="SalesModel"], dmPredictors)

  expect_equal(nrow(perfOverviewSalesModel), 47)

  # Models with a single active classifier bin should have performance 0.5. Only the ones that have just one
  # classifier bin (active or not) currently do.
  expect_equal(perfOverviewSalesModel[(score_bin_max - score_bin_min) == 0]$correct_performance, rep(0.5, 10))
  expect_equal(perfOverviewSalesModel[nbins==1]$performance, rep(0.5, sum(perfOverviewSalesModel$nbins==1)))

  # All of the models that have multiple classifier bins currently report a performance > 0.5 even if those
  # bins are not active.
  # This test explictly tests that situation and should be flipped once the bug is fixed.
  dummy <- sapply(perfOverviewSalesModel[(nbins != 1) & ((score_bin_max - score_bin_min) == 0)]$performance,
                  function(x) { expect_gt(x, 0.5) })

  # Performance of classifiers that use their full range should be correct
  expect_equal(perfOverviewSalesModel[(nbins > 1) & (score_bin_max - score_bin_min + 1 == nbins)]$performance,
               perfOverviewSalesModel[(nbins > 1) & (score_bin_max - score_bin_min + 1 == nbins)]$correct_performance,
               tolerance = 1e-06)
})
