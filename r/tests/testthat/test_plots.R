library(testthat)
library(cdhtools)
library(ggplot2)
library(scales)

context("Plotting functions")

test_that("Abbreviate Names" , {
  expect_equal(plotsAbbreviateName("Hello World"), "Hello World")
  expect_equal(plotsAbbreviateName("IH.Inbound.CallCenter.Accept.CountOf"), "IH...d.CallCenter.Accept.CountOf")
  expect_equal(plotsAbbreviateName("Hello World", len=8), "Hel...ld")
  expect_equal(plotsAbbreviateName("Hello World", len=9), "Hel...rld")
  expect_equal(plotsAbbreviateName("Hello World", len=4), "Hell")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", len=18), "IH...ccept.CountOf")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", len=6), "IH...f")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", len=4), "IH..")
  expect_equal(plotsAbbreviateName("IH.Inbound.Accept.CountOf", len=1), "I")
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

