library(testthat)
library(cdhtools)
library(ggplot2)
library(scales)

context("Plotting functions")

test_that("Abbreviate Names" , {
  expect_equal(plotsAbbreviateName("Hello World", abbreviateLength=32), "Hello World")
  expect_equal(plotsAbbreviateName("IH.Inbound.CallCenter.Accept.CountOf", abbreviateLength=32), "IH...d.CallCenter.Accept.CountOf")
  expect_equal(plotsAbbreviateName("Hello World", abbreviateLength=8), "Hel...ld")
  expect_equal(plotsAbbreviateName("Hello World", abbreviateLength=9), "Hel...rld")
  expect_equal(plotsAbbreviateName("Hello World", abbreviateLength=4), "Hell")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", abbreviateLength=18), "IH...ccept.CountOf")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", abbreviateLength=6), "IH...f")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", abbreviateLength=4), "IH..")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", abbreviateLength=1), "I")
})

test_that("Abbreviate Intervals" , {
  expect_equal(plotsAbbreviateInterval("[-0.2488888, -0.2222222>"), "[-0.248, -0.222>")
  expect_equal(plotsAbbreviateInterval("<-0.2222222"), "<-0.222")
  expect_equal(plotsAbbreviateInterval(">=0.987654321"), ">=0.987")
})

test_that("Default Predictor Categorization", {
  expect_equal(defaultPredictorCategorization("IH.Inbound.CountOf"), "IH")
  expect_equal(defaultPredictorCategorization("NetWealth"), "TopLevel")
  expect_equal(defaultPredictorCategorization(
    c("IH.Inbound.CountOf", "Parameter.HourOfDay", "Age", "Customer.Incomd")),
    c("IH", "Parameter", "TopLevel", "Customer"))
})

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

test_that("plotADMPredictorImportance", {
  data(admdatamart_models)
  data(admdatamart_binning)

  p <- plotADMPredictorImportance(admdatamart_binning)

  expect_s3_class(p, "ggplot")

  p <- plotADMPredictorImportance(admdatamart_binning, admdatamart_models)

  expect_s3_class(p, "ggplot")

  p <- plotADMPredictorImportance(admdatamart_binning, admdatamart_models, categoryAggregateView = T)

  expect_s3_class(p, "ggplot")
})

test_that("plotADMPredictorImportanceHeatmap", {
  data(admdatamart_models)
  data(admdatamart_binning)

  p <- plotADMPredictorImportanceHeatmap(admdatamart_binning, admdatamart_models) # ****

  expect_s3_class(p, "ggplot")

  p <- plotADMPredictorImportanceHeatmap(admdatamart_binning, admdatamart_models, aggregation = "Name", facets = "")

  expect_s3_class(p, "ggplot")
})

test_that("plotADMPropositionSuccessRates", {
  data(admdatamart_models)

  p <- plotADMPropositionSuccessRates(admdatamart_models)

  expect_s3_class(p, "ggplot")
})
