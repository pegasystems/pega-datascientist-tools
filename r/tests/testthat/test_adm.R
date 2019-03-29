library(testthat)
library(cdhtools)

context("ADM Checks")

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

  # For ad-hoc check

      # dmModels <- readDSExport("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots","~/Downloads/auc_bug") # "dsexports"
      # dmPredictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots","~/Downloads/auc_bug")
      # dmModels <- dmModels[pyConfigurationName %in% c("SalesModel"), !grepl("^px|pz", names(dmModels)), with=F]
      # perfOverview <- getModelPerformanceOverview(dmModels, dmPredictors)

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
