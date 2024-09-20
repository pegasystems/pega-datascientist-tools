library(testthat)
# library(pdstools)

context("ADM Checks")

# For ad-hoc check

# dmModels <- readDSExport("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots","~/Downloads/auc_bug") # "dsexports"
# dmPredictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots","~/Downloads/auc_bug")
# dmModels <- dmModels[pyConfigurationName %in% c("SalesModel"), !grepl("^px|pz", names(dmModels)), with=F]
# perfOverview <- getModelPerformanceOverview(dmModels, dmPredictors)

test_that("read ADM Datamart", {
  data <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All", "dsexports")

  expect_true(is.null(data$predictordata))
  expect_equal(nrow(data$modeldata), 30)
  expect_equal(ncol(data$modeldata), 16) # omits internal fields

  expect_identical(sort(names(data$modeldata)),
                   sort(c("ConfigurationName", "Direction", "Group", "Issue",
                          "Channel", "ModelID", "Name", "Negatives", "Performance", "Positives",
                          "ResponseCount", "SnapshotTime", "SuccessRate",
                          "ActivePredictors", "TotalPredictors", "AUC" )))

  expect_equal(sum(sapply(data$modeldata, is.factor)), 6)
  expect_equal(sum(sapply(data$modeldata, is.numeric)), 8)
  expect_equal(sum(sapply(data$modeldata, is.POSIXt)), 1)
  expect_equal(sum(sapply(data$modeldata, is.character)), 1)

  data <- ADMDatamart(modeldata = "Data-Decision-ADM-ModelSnapshot_All",
                      predictordata = F,
                      folder = "dsexports",
                      filterModelData = filterLatestSnapshotOnly)

  expect_equal(nrow(data$modeldata), 15) # only latest snapshot
  expect_equal(ncol(data$modeldata), 16)
})

test_that("ADM Datamart positional arguments", {
  d <- ADMDatamart("dsexports")
  expect_equal(nrow(d$modeldata), 20)
  expect_equal(nrow(d$predictordata), 1648)

  d <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots", "dsexports")
  expect_equal(nrow(d$modeldata), 20)
  expect_null(d$predictordata)

  d <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots", folder = "dsexports")
  expect_equal(nrow(d$modeldata), 20)
  expect_equal(nrow(d$predictordata), 1648)

  d <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All", "dsexports")
  expect_equal(nrow(d$modeldata), 30)
  expect_null(d$predictordata)

  d <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All", folder = "dsexports")
  expect_equal(nrow(d$modeldata), 30)
  expect_equal(nrow(d$predictordata), 1755)

  d <- ADMDatamart(modeldata = "Data-Decision-ADM-ModelSnapshot_All", folder = "dsexports")
  expect_equal(nrow(d$modeldata), 30)
  expect_equal(nrow(d$predictordata), 1755)

  d <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_MyModelSnapshots_20211227T115918_GMT.zip", "dsexports")
  expect_equal(nrow(d$modeldata), 1)
  expect_null(d$predictordata)

  d <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_MyModelSnapshots_20211227T115918_GMT.zip", folder = "dsexports")
  expect_equal(nrow(d$modeldata), 1)
  expect_null(d$predictordata) # because the standard predictor data there is not compatible with this model data it will be nulled

  d <- ADMDatamart(modeldata = "Data-Decision-ADM-ModelSnapshot_MyModelSnapshots_20211227T115918_GMT.zip", folder = "dsexports")
  expect_equal(nrow(d$modeldata), 1)
  expect_null(d$predictordata) # because the standard predictor data there is not compatible with this model data it will be nulled

  expect_error(ADMDatamart("Data-Decision-ADM-ModelSnapshot_DoesNotExist", folder = "dsexports"),
               "Dataset JSON file not found")

  expect_error(ADMDatamart("test_score_ranges.RData", folder = "dsexports"),
               "Unsupported file type")

})


test_that("ADMDatamart predictor reading", {

  # Default reads all snapshots, and includes binning
  data <- ADMDatamart(modeldata = F, predictordata = "Data-Decision-ADM-PredictorBinningSnapshot_All", folder = "dsexports")
  expect_equal(nrow(data$predictordata), 1755) # default reads all snapshots, all binning
  expect_equal(ncol(data$predictordata), 25)

  expect_equal(sum(sapply(data$predictordata, is.factor)), 6)
  expect_equal(sum(sapply(data$predictordata, is.numeric)), 16)
  expect_equal(sum(sapply(data$predictordata, is.POSIXt)), 1)
  expect_equal(sum(sapply(data$predictordata, is.character)), 2)

  # Add filter to drop binning
  data <- ADMDatamart(modeldata = F, predictordata = "Data-Decision-ADM-PredictorBinningSnapshot_All", folder = "dsexports",
                      filterPredictorData = filterPredictorBinning)
  expect_equal(nrow(data$predictordata), 425) # binning fields removed
  expect_equal(ncol(data$predictordata), 15)

  expect_equal(sum(sapply(data$predictordata, is.factor)), 6)
  expect_equal(sum(sapply(data$predictordata, is.numeric)), 7)
  expect_equal(sum(sapply(data$predictordata, is.POSIXt)), 1)
  expect_equal(sum(sapply(data$predictordata, is.character)), 1)
})

test_that("add Contents to ADM Datamart", {
  # for segment explore tool the actual range is useful, so lets include
  data <- ADMDatamart(modeldata = F, predictordata = "Data-Decision-ADM-PredictorBinningSnapshot_All", folder = "dsexports")
  expect_true("Contents" %in% names(data$predictordata))
})

test_that("extract model version", {
  m <- ADMDatamart("ADM-ModelSnapshots-withJSON-fromCDHSample.zip", predictordata = F, folder = "dsexports")

  expect_true("ExtractedVersion" %in% names(m$modeldata))
  expect_equal(m$modeldata$ExtractedVersion[1], "8.2.0.7622")
})

test_that("parsing JSON names from pyName", {

  # m1 is the "raw" version
  m1 <- readDSExport(instancename="ADM-ModelSnapshots-withJSON-fromCDHSample.zip",
                     srcFolder="dsexports")
  expect_equal(ncol(m1), 24)
  expect_equal(nrow(m1), 361)
  expect_false("Treatment" %in% names(m1))
  expect_false("Proposition" %in% names(m1))
  expect_setequal(unique(m1$pyName),
                  c("{\"Proposition\":\"P1\"}","{\"pyName\":\"SuperSaver\",\"pyTreatment\":\"Bundles_WebTreatment\"}","BasicChecking","P10"))

  # m2 is the one where the JSON names should be expanded, resulting in a few extra columns
  m2 <- ADMDatamart(m1, predictordata = F)

  expect_equal(ncol(m2$modeldata), 19)
  expect_equal(nrow(m2$modeldata), 361)
  expect_true("Treatment" %in% names(m2$modeldata))
  expect_true("Proposition" %in% names(m2$modeldata))
  expect_setequal(levels(m2$modeldata$Name),
                  c('{"Proposition":"P1"}','BasicChecking','P10','SuperSaver'))
})

test_that("Reading data from models wo context", {
  # This model has no context keys at all, not even Name
  data <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_MyModelSnapshots",
                      "Data-Decision-ADM-PredictorBinningSnapshot_MyPredictorSnapshots", folder = "dsexports")

  expect_equal(ncol(data$modeldata), 11)
  expect_equal(nrow(data$modeldata), 1)
  expect_equal(ncol(data$predictordata), 25)
  expect_equal(nrow(data$predictordata), 55)

})

# This test confirms BUG-417860 stating that ADM performance numbers can be overly
# optimistic in the beginning.

test_that("Active score range DMSample", {
  load("dsexports/test_score_ranges.RData")
  activeRange <- getActiveRanges(ADMDatamart(dmModels, dmPredictors))

  # one model where the actual range is smaller than the full one
  expect_equal(activeRange[["664cc653-279f-54ae-926f-694652d89a54"]]$active_index_min, 7)
  expect_equal(activeRange[["664cc653-279f-54ae-926f-694652d89a54"]]$active_index_max, 7)
  expect_false(activeRange[["664cc653-279f-54ae-926f-694652d89a54"]]$is_full_indexrange)
  expect_equal(activeRange[["664cc653-279f-54ae-926f-694652d89a54"]]$reportedAUC, 0.760333, 1e-5)
  expect_equal(activeRange[["664cc653-279f-54ae-926f-694652d89a54"]]$activeRangeAUC, 0.5, 1e-5)

  # one model with active range equal to full range
  expect_equal(activeRange[["4574f1fd-13a7-5703-bf38-9374641f370f"]]$active_index_min, 1)
  expect_equal(activeRange[["4574f1fd-13a7-5703-bf38-9374641f370f"]]$active_index_max, 2)
  expect_true(activeRange[["4574f1fd-13a7-5703-bf38-9374641f370f"]]$is_full_indexrange)
  expect_equal(activeRange[["4574f1fd-13a7-5703-bf38-9374641f370f"]]$reportedAUC, 0.557292, 1e-5)
  expect_equal(activeRange[["4574f1fd-13a7-5703-bf38-9374641f370f"]]$activeRangeAUC, 0.557292, 1e-5)

  datamart <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All",
                          "Data-Decision-ADM-PredictorBinningSnapshot_All",
                          "dsexports")
  activeRange <- getActiveRanges(datamart)

  expect_equal(activeRange[[1]]$active_index_min, 1)
  expect_equal(activeRange[[1]]$active_index_max, 3)
  expect_false(activeRange[[1]]$is_full_indexrange)
  expect_equal(activeRange[[1]]$reportedAUC, 0.544661, 1e-5)
  expect_equal(activeRange[[1]]$activeRangeAUC, 0.534542, 1e-5)
})

test_that("Active score range CDH Sample Pega8", {
  data <- ADMDatamart("CDHSample-Pega8-ADMModelSnapshots_20211029T133255_GMT.zip",
                      "CDHSample-Pega8-ADMPredictorSnapshots_20211029T134643_GMT.zip", folder = "dsexports")

  activeRangeList <- getActiveRanges(data)
  activeRange <- rbindlist(activeRangeList[-length(activeRangeList)])

  expect_equal(nrow(activeRange), 4)

  # Note that the last model here may have an issue. The reported AUC does not match
  # either the full range AUC or the AUC from the re-calculated active range. For this
  # data (the full one) a few models had this issue.
  expect_equal(activeRange$score_min, c(-0.6007593, -0.6209292, -0.7491203, -1.4255568), 1e-5)
  expect_equal(activeRange$score_max, c(0.6812438, 0.4689029, 1.4541791, -0.4254948), 1e-5)
  expect_equal(activeRange$active_index_min, c(1, 3, 1, 2))
  expect_equal(activeRange$active_index_max, c(10, 10, 2, 4))
  expect_equal(activeRange$nClassifierBins, c(11, 11, 2, 6))
  expect_equal(activeRange$reportedAUC, c(0.562353, 0.533401, 0.559777, 0.628571), 1e-5)
  expect_equal(activeRange$fullRangeAUC, c(0.5652007, 0.5460856, 0.5597765, 0.5781145), 1e-5)
  expect_equal(activeRange$activeRangeAUC, c(0.5623530, 0.5334013, 0.5597765, 0.5476054), 1e-5)
  expect_equal(activeRange$is_full_indexrange, c(F, F, T, F))
  expect_equal(activeRange$is_AUC_fullrange, c(F, F, T, F))
  expect_equal(activeRange$is_AUC_activerange, c(T, T, T, F))
})

# In old versions of Pega, ADM calculated the AUC from all the classifier bins
# this test uses data from those days

test_that("Active score range old datamart", {
  data <- ADMDatamart(modeldata=F, "BigModel_predictordata.csv", folder = "d")
  activeRangeList <- getActiveRanges(data)
  activeRange <- rbindlist(activeRangeList[-length(activeRangeList)])

  expect_equal(nrow(activeRange), 5)

  expect_equal(activeRange$is_full_indexrange, rep(F, 5)) # active range is always smaller
  expect_equal(activeRange$is_AUC_fullrange, rep(T, 5))   # reported AUC always matches full range AUC
  expect_equal(activeRange$is_AUC_activerange, rep(F, 5)) # reported AUC never matches active range AUC
})

test_that("Feature Importance Unit", {
  context("Feature Importance Unit")

  dm <- list(
    modeldata = data.table(
      ModelID = c("a", "b"),
      Configuration = c("c1")
    ),
    predictordata = data.table(
      BinPositives = c(10, 20, 5, 40, 2, 1, 1, 2),
      BinNegatives = c(100, 100, 200, 500, 10, 20, 5, 8),
      BinResponseCount = c(110, 120, 205, 540, 12, 21, 6, 10),
      Performance = c(0.5),
      PredictorName = c("P1", "P1","P1","P1","P2","P2","P2","P2"),
      PredictorCategory = "PC",
      ModelID = c("a", "a", "b", "b", "a", "a", "b", "b")
    )
  )

  # Unscaled
  varimp <- admVarImp(dm, filter=function(x){x}, debug=T, facets="Configuration", scaled=F)

  print(varimp)

  expect_equal(varimp$PredictorName[1], "P1")
  expect_equal(varimp$PredictorName[2], "P2")

  expect_equal(varimp$Importance[1], 0.408486321)
  expect_equal(varimp$Importance[2], 0.379310604)

  # Scaled
  varimp <- admVarImp(dm, filter=function(x){x}, debug=T, facets="Configuration")

  print(varimp)

  expect_equal(varimp$PredictorName[1], "P1")
  expect_equal(varimp$PredictorName[2], "P2")

  expect_equal(varimp$Importance[1], 100)
  expect_equal(varimp$Importance[2], 92.8576025)
})

test_that("Feature Importance", {
  context("Feature Importance")

  datamart <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All",
                          "Data-Decision-ADM-PredictorBinningSnapshot_All",
                          "dsexports")

  # Without filter would give much more results
  varimp <- admVarImp(datamart)

  expect_equal(ncol(varimp), 8)
  expect_equal(nrow(varimp), 410)

  varimp <- admVarImp(datamart, filter = filterActiveOnly)

  expect_equal(ncol(varimp), 8)
  expect_equal(nrow(varimp), 137)

  expect_equal(max(varimp$Importance), 100) # Should be scaled
  expect_equal(varimp$ImportanceRank[1], 1) # Highest should be on top

  # With aggregator, rolls up
  varimp <- admVarImp(datamart, "ConfigurationName", filter = filterActiveOnly)

  expect_equal(ncol(varimp), 8)
  expect_equal(nrow(varimp), 45)

  expect_equal(max(varimp$Importance), 100) # Should be scaled
  expect_equal(varimp$ImportanceRank[1], 1) # Highest should be on top
})

test_that("ADMDatamart from DS exports", {
  dm <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All", "Data-Decision-ADM-PredictorBinningSnapshot_All", "dsexports")

  expect_equal(length(dm), 2)
  expect_equal(nrow(dm$modeldata), 30)
  expect_equal(nrow(dm$predictordata), 1755)
})

test_that("ADMDatamart from tables", {
  mdls <- readDSExport(instancename = "Data-Decision-ADM-ModelSnapshot_All", srcFolder = "dsexports")
  prds <- readDSExport(instancename = "Data-Decision-ADM-PredictorBinningSnapshot_All", srcFolder = "dsexports")
  dm <- ADMDatamart(mdls, prds)

  expect_equal(length(dm), 2)
  expect_equal(nrow(dm$modeldata), 30)
  expect_equal(nrow(dm$predictordata), 1755)

  dm <- ADMDatamart(mdls, prds,
                    filterModelData = function(x) { return(x[!startsWith(Name, "B")])})

  expect_equal(length(dm), 2)
  expect_equal(nrow(dm$modeldata), 20)
  expect_equal(nrow(dm$predictordata), 842)
})

test_that("hasMultipleSnapshots", {
  data <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All_20180316T135038_GMT.zip",
                      "Data-Decision-ADM-PredictorBinningSnapshot_All_20180316T135050_GMT.zip", "dsexports",
                      cleanupHookModelData = function(mdls) { mdls[, c("Channel", "Direction") := "NA"]})

  expect_true(hasMultipleSnapshots(data$modeldata))
  expect_false(hasMultipleSnapshots(data$predictordata))
})

test_that("Filtering", {
  data <- ADMDatamart(modeldata = F,
                      "Data-Decision-ADM-PredictorBinningSnapshot_All_20180316T135050_GMT.zip", "dsexports")

  expect_equal(nrow(data$predictordata), 1755)
  expect_equal(ncol(data$predictordata), 25)
  expect_equal(nrow(filterActiveOnly(data$predictordata)), 643)
  expect_equal(nrow(filterActiveOnly(data$predictordata, reverse = T)), 1112)
  expect_equal(nrow(filterInactiveOnly(data$predictordata)), 853)
  expect_equal(nrow(filterInactiveOnly(data$predictordata, reverse = T)), 902)
  expect_equal(nrow(filterClassifierOnly(data$predictordata)), 259)
  expect_equal(nrow(filterClassifierOnly(data$predictordata, reverse = T)), 1496)
  expect_equal(nrow(filterPredictorBinning(data$predictordata)), 425)
  expect_equal(ncol(filterPredictorBinning(data$predictordata)), 16)
})

test_that("Predictor Categorization", {
  expect_equal(defaultPredictorCategorization("Customer.Age"), "Customer")
  expect_equal(defaultPredictorCategorization(""), "TopLevel")
  expect_equal(defaultPredictorCategorization(c("Customer.Age", "Customer.Income", "Context.Clicks", "ChurnScore"),
                                              topLevelLabel="Primary"),
                                              c("Customer", "Customer", "Context", "Primary"))
}
)
