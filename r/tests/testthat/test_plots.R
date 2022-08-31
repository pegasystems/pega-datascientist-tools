library(testthat)
# library(pdstools)
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
  data(adm_datamart)

  p <- plotBinning(adm_datamart$predictordata[PredictorName=="Customer.Age"][ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotCumulativeGains", {
  data(adm_datamart)

  p <- plotCumulativeGains(filterClassifierOnly(adm_datamart$predictordata)[ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotCumulativeLift", {
  data(adm_datamart)

  p <- plotCumulativeLift(filterClassifierOnly(adm_datamart$predictordata)[ModelID==ModelID[1]])

  expect_s3_class(p, "ggplot")
})

test_that("plotPerformanceOverTime", {
  data(adm_datamart)

  p <- plotPerformanceOverTime(adm_datamart)

  expect_s3_class(p, "ggplot")

  p <- plotPerformanceOverTime(adm_datamart, aggregation = "Issue", facets = "Group")

  expect_s3_class(p, "ggplot")
})

test_that("plotSuccessRateOverTime", {
  data(adm_datamart)

  p <- plotSuccessRateOverTime(adm_datamart)

  expect_s3_class(p, "ggplot")

  p <- plotSuccessRateOverTime(adm_datamart, aggregation = "Name", facets = c("Group", "ConfigurationName"))

  expect_s3_class(p, "ggplot")
})

test_that("plotPerformanceSuccessRateBoxPlot", {
  data(adm_datamart)

  p <- plotPerformanceSuccessRateBoxPlot(adm_datamart)

  expect_s3_class(p, "ggplot")

  p <- plotPerformanceSuccessRateBoxPlot(adm_datamart, facets = c("Group", "ConfigurationName"))

  expect_s3_class(p, "ggplot")
})

test_that("plotPerformanceSuccessRateBubbleChart", {
  data(adm_datamart)

  p <- plotPerformanceSuccessRateBubbleChart(adm_datamart)

  expect_s3_class(p, "ggplot")

  p <- plotPerformanceSuccessRateBubbleChart(adm_datamart, aggregation = "Name", facets = "ConfigurationName")

  expect_s3_class(p, "ggplot")
})

test_that("plotPredictorImportance", {
  data(adm_datamart)

  p <- plotPredictorImportance(adm_datamart)

  expect_s3_class(p, "ggplot")

  p <- plotPredictorImportance(adm_datamart, categoryAggregateView = T)

  expect_s3_class(p, "ggplot")
})

test_that("plotPredictorImportanceHeatmap", {
  data(adm_datamart)

  p <- plotPredictorImportanceHeatmap(adm_datamart)

  expect_s3_class(p, "ggplot")

  p <- plotPredictorImportanceHeatmap(adm_datamart, aggregation = "Name", facets = "")

  expect_s3_class(p, "ggplot")
})

test_that("plotPropositionSuccessRates", {
  data(adm_datamart)

  p <- plotPropositionSuccessRates(adm_datamart)

  expect_s3_class(p, "ggplot")
})

test_that("userFriendlyADMBinning", {
  data <- ADMDatamart("Data-Decision-ADM-ModelSnapshot_All_20180316T134315_GMT.zip",
                      "Data-Decision-ADM-PredictorBinningSnapshot_All_20180316T135050_GMT.zip", "dsexports",
                      cleanupHookModelData = function(mdls) { mdls[, c("Channel", "Direction") := "NA"]},
                      filterModelData = function(mdls) { return(mdls[ModelID == "0e5e83b4-6313-5fb2-8947-5e83e37437e9"])})

  ufb <- userFriendlyADMBinning(filterClassifierOnly(data$predictordata))

  expect_equal(nrow(ufb), 38)
  expect_equal(ncol(ufb), 10)

  ufb <- userFriendlyADMBinning(data$predictordata[PredictorName == "PERSONALFIELD2"])

  expect_equal(nrow(ufb), 10)
  expect_equal(ncol(ufb), 7)

})
