library(testthat)

context("Toplevel Examples")

test_that("example folder exists", {
  exampleFolder <- file.path("../../..", "examples")

  expect_true(dir.exists(exampleFolder))
})

# test_that("expected R notebooks are there", {
#   notebooks <- list.files(exampleFolder, pattern=".Rmd", recursive = T, full.names = T)
#
#   expect_equal(4, length(notebooks), info="Number of R notebooks")
# })
