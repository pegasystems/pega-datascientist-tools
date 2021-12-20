library(testthat)
# library(cdhtools)
# library(ggplot2)
# library(scales)

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

test_that("plotBinning", {
  data(admdatamart)

  p <- plotBinning(admdatamart$predictordata[PredictorName=="Customer.Age"][ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotCumulativeGains", {
  data(admdatamart)

  p <- plotCumulativeGains(filterClassifierOnly(admdatamart$predictordata)[ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotCumulativeLift", {
  data(admdatamart)

  p <- plotCumulativeLift(filterClassifierOnly(admdatamart$predictordata)[ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotPerformanceOverTime", {
  data(admdatamart)

  p <- plotPerformanceOverTime(admdatamart)

  expect_s3_class(p, "ggplot")

  p <- plotPerformanceOverTime(admdatamart, aggregation = "Issue", facets = "Group")

  expect_s3_class(p, "ggplot")
})

test_that("plotSuccessRateOverTime", {
  data(admdatamart)

  p <- plotSuccessRateOverTime(admdatamart)

  expect_s3_class(p, "ggplot")

  p <- plotSuccessRateOverTime(admdatamart, aggregation = "Name", facets = c("Group", "ConfigurationName"))

  expect_s3_class(p, "ggplot")
})

test_that("plotPerformanceSuccessRateBoxPlot", {
  data(admdatamart)

  p <- plotPerformanceSuccessRateBoxPlot(admdatamart)

  expect_s3_class(p, "ggplot")

  p <- plotPerformanceSuccessRateBoxPlot(admdatamart, facets = c("Group", "ConfigurationName"))

  expect_s3_class(p, "ggplot")
})

test_that("plotPerformanceSuccessRateBubbleChart", {
  data(admdatamart)

  p <- plotPerformanceSuccessRateBubbleChart(admdatamart)

  expect_s3_class(p, "ggplot")

  p <- plotPerformanceSuccessRateBubbleChart(admdatamart, aggregation = "Name", facets = "ConfigurationName")

  expect_s3_class(p, "ggplot")
})

test_that("plotPredictorImportance", {
  data(admdatamart)

  p <- plotPredictorImportance(admdatamart)

  expect_s3_class(p, "ggplot")

  p <- plotPredictorImportance(admdatamart, categoryAggregateView = T)

  expect_s3_class(p, "ggplot")
})

test_that("plotPredictorImportanceHeatmap", {
  data(admdatamart)

  p <- plotPredictorImportanceHeatmap(admdatamart)

  expect_s3_class(p, "ggplot")

  p <- plotPredictorImportanceHeatmap(admdatamart, aggregation = "Name", facets = "")

  expect_s3_class(p, "ggplot")
})

test_that("plotPropositionSuccessRates", {
  data(admdatamart)

  p <- plotPropositionSuccessRates(admdatamart)

  expect_s3_class(p, "ggplot")
})
