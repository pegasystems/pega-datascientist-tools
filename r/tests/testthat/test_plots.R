library(testthat)
library(cdhtools)
library(ggplot2)
library(scales)

context("Plotting functions")

test_that("plotADMBinning", {
  data(admdatamart_binning)

  p <- plotADMBinning(admdatamart_binning[PredictorName=="Customer.Age"][ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotADMCumulativeGains", {
  data(admdatamart_binning)

  p <- plotADMCumulativeGains(admdatamart_binning[EntryType=="Classifier"][ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotADMCumulativeLift", {
  data(admdatamart_binning)

  p <- plotADMCumulativeLift(admdatamart_binning[EntryType=="Classifier"][ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotADMModelPerformanceOverTime", {
  data(admdatamart_models)

  p <- plotADMModelPerformanceOverTime(admdatamart_models) # ****

  expect_s3_class(p, "ggplot")

  p <- plotADMModelPerformanceOverTime(admdatamart_models, "Name", "Group")

  expect_s3_class(p, "ggplot")
})

test_that("plotADMModelSuccessRateOverTime", {
  data(admdatamart_models)

  p <- plotADMModelSuccessRateOverTime(admdatamart_models) # ****

  expect_s3_class(p, "ggplot")

  p <- plotADMModelSuccessRateOverTime(admdatamart_models, "Name", c("Group", "ConfigurationName"))

  expect_s3_class(p, "ggplot")
})

test_that("plotADMPerformanceSuccessRateBoxPlot", {
  data(admdatamart_models)

  p <- plotADMPerformanceSuccessRateBoxPlot(admdatamart_models)

  expect_s3_class(p, "ggplot")

  p <- plotADMPerformanceSuccessRateBoxPlot(admdatamart_models, "Group", "ConfigurationName")

  expect_s3_class(p, "ggplot")
})

test_that("plotADMPerformanceSuccessRateBubbleChart", {
  data(admdatamart_models)

  p <- plotADMPerformanceSuccessRateBubbleChart(admdatamart_models) # ****

  expect_s3_class(p, "ggplot")

  p <- plotADMPerformanceSuccessRateBubbleChart(admdatamart_models, aggregation = "Name", facets = "ConfigurationName")

  expect_s3_class(p, "ggplot")
})

test_that("plotADMPredictorPerformance", {
  data(admdatamart_models)
  data(admdatamart_binning)

  p <- plotADMPredictorPerformance(admdatamart_binning)

  expect_s3_class(p, "ggplot")

  p <- plotADMPredictorPerformance(admdatamart_binning, admdatamart_models)

  expect_s3_class(p, "ggplot")
})

test_that("plotADMPredictorPerformanceMatrix", {
  data(admdatamart_models)
  data(admdatamart_binning)

  p <- plotADMPredictorPerformanceMatrix(admdatamart_binning, admdatamart_models) # ****

  expect_s3_class(p, "ggplot")

  p <- plotADMPredictorPerformanceMatrix(admdatamart_binning, admdatamart_models, aggregation = "Name", facets = "")

  expect_s3_class(p, "ggplot")
})

test_that("plotADMPropositionSuccessRates", {
  data(admdatamart_models)

  p <- plotADMPropositionSuccessRates(admdatamart_models) # ****

  expect_s3_class(p, "ggplot")
})

test_that("plotADMVarImp", {
  data(admdatamart_models)
  data(admdatamart_binning)

  p <- plotADMVarImp(admVarImp(admdatamart_models, admdatamart_binning))

  expect_s3_class(p, "ggplot")

  p <- plotADMVarImp(admVarImp(admdatamart_models, admdatamart_binning, facets = "Group"))

  expect_s3_class(p, "ggplot")
})

