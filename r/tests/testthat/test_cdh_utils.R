library(testthat)
library(cdhtools)

context("check something")

test_that("dataset export can be read", {
  data <- readDSExport("Data-pxStrategyResult_Debug","dsexports")
  expect_equal(nrow(data), 56) # TODO - save examples
})
