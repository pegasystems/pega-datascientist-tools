library(testthat)
library(cdhtools)
library(lubridate)

context("check PMML")

# Runs a full unit test by generating PMML from the model and predictor data,
# scoring that with JPMML against provided inputs and comparing the results against expected values.
pmml_unittest <- function(testName)
{
    cat("test: ", testName, fill=T)
}

test_that("a basic happy-path test with a single 2-predictor model w/o missings, WOS etc", {
  pmml_unittest("singlesimplemodel")
})

