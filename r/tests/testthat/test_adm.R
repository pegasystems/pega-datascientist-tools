library(testthat)
library(cdhtools)

context("ADM Checks")

# For ad-hoc check

# dmModels <- readDSExport("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots","~/Downloads/auc_bug") # "dsexports"
# dmPredictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots","~/Downloads/auc_bug")
# dmModels <- dmModels[pyConfigurationName %in% c("SalesModel"), !grepl("^px|pz", names(dmModels)), with=F]
# perfOverview <- getModelPerformanceOverview(dmModels, dmPredictors)


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
