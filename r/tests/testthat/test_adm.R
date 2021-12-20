library(testthat)
# library(cdhtools)

context("ADM Checks")

# For ad-hoc check

# dmModels <- readDSExport("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots","~/Downloads/auc_bug") # "dsexports"
# dmPredictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots","~/Downloads/auc_bug")
# dmModels <- dmModels[pyConfigurationName %in% c("SalesModel"), !grepl("^px|pz", names(dmModels)), with=F]
# perfOverview <- getModelPerformanceOverview(dmModels, dmPredictors)

test_that("generic ADMDatamart reading", {

  expect_error(ADMDatamart("Data-Decision-ADM-ModelSnapshot_All", folder = "dsexports"),
               "Dataset JSON file not found looking for pattern")

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

test_that("ADMDatamart predictor reading", {

  # Default reads all snapshots, and includes binning
  data <- ADMDatamart(modeldata = F, predictordata = "Data-Decision-ADM-PredictorBinningSnapshot_All", folder = "dsexports")
  expect_equal(nrow(data$predictordata), 1755) # default reads all snapshots, all binning
  expect_equal(ncol(data$predictordata), 24)

  expect_equal(sum(sapply(data$predictordata, is.factor)), 5)
  expect_equal(sum(sapply(data$predictordata, is.numeric)), 16)
  expect_equal(sum(sapply(data$predictordata, is.POSIXt)), 1)
  expect_equal(sum(sapply(data$predictordata, is.character)), 2)

  # Add filter to drop binning
  data <- ADMDatamart(modeldata = F, predictordata = "Data-Decision-ADM-PredictorBinningSnapshot_All", folder = "dsexports",
                      filterPredictorData = filterPredictorBinning)
  expect_equal(nrow(data$predictordata), 425) # binning fields removed
  expect_equal(ncol(data$predictordata), 14)

  expect_equal(sum(sapply(data$predictordata, is.factor)), 5)
  expect_equal(sum(sapply(data$predictordata, is.numeric)), 7)
  expect_equal(sum(sapply(data$predictordata, is.POSIXt)), 1)
  expect_equal(sum(sapply(data$predictordata, is.character)), 1)
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

  expect_equal(ncol(m2$modeldata), 18)
  expect_equal(nrow(m2$modeldata), 361)
  expect_true("Treatment" %in% names(m2$modeldata))
  expect_true("Proposition" %in% names(m2$modeldata))
  expect_setequal(levels(m2$modeldata$Name),
                  c('{"Proposition":"P1"}','BasicChecking','P10','SuperSaver'))
})

# This test confirms BUG-417860 stating that ADM performance numbers can be overly
# optimistic in the beginning.
test_that("Score Ranges No Predictors", {
  context("Verify range of score distribution w/o predictors")

  load("dsexports/test_score_ranges.RData")

  # Models from the "independent bundle" strategy. There are no customer level predictors in this model and there
  # are also no parameterized predictors for the sequencing. No predictors at all should give all models performance
  # 0.5 but due to BUG-417860 this is not the case in older releases. Therefore commented out the first test.
  perfOverviewNoPredictors <-
    getModelPerformanceOverview(dmModels[ConfigurationName=="BundleModelIndependent"], dmPredictors)

  # expect_equal(perfOverviewNoPredictors$reported_performance, rep(0.5, nrow(perfOverviewNoPredictors)))
  expect_equal(perfOverviewNoPredictors$actual_performance, rep(0.5, nrow(perfOverviewNoPredictors)))
})

# More normal performance values
test_that("Score Ranges DMSample", {
  context("Verify range of score distribution")

  load("dsexports/test_score_ranges.RData")

  # The Sales Model is an initialized DMSample system. Some of the models have too optimistic performance numbers early on in their
  # life due to the same bug mentioned above.
  perfOverviewSalesModel <- getModelPerformanceOverview(dmModels[ConfigurationName=="SalesModel"], dmPredictors)

  expect_equal(nrow(perfOverviewSalesModel), 47)

  # Models with a single active classifier bin should have performance 0.5. Only the ones that have just one
  # classifier bin (active or not) currently do.
  expect_equal(perfOverviewSalesModel[(actual_score_bin_max - actual_score_bin_min) == 0]$actual_performance, rep(0.5, 10))
  expect_equal(perfOverviewSalesModel[nbins==1]$reported_performance, rep(0.5, sum(perfOverviewSalesModel$nbins==1)))

  # All of the models that have multiple classifier bins currently report a performance > 0.5 even if those
  # bins are not active.
  # This test explictly tests that situation and should be flipped once the bug is fixed.
  dummy <- sapply(perfOverviewSalesModel[(nbins != 1) & ((actual_score_bin_max - actual_score_bin_min) == 0)]$reported_performance,
                  function(x) { expect_gt(x, 0.5) })

  # Performance of classifiers that use their full range should be correct
  expect_equal(perfOverviewSalesModel[(nbins > 1) & (actual_score_bin_max - actual_score_bin_min + 1 == nbins)]$reported_performance,
               perfOverviewSalesModel[(nbins > 1) & (actual_score_bin_max - actual_score_bin_min + 1 == nbins)]$actual_performance,
               tolerance = 1e-06)
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
