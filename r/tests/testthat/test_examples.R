library(testthat)

context("Toplevel Examples")

# > test_check("cdhtools")
# [1] "Example folder check"
# [1] "C:/Users/runneradmin/AppData/Local/Temp/RtmpYJIkSb/file442fbea53/cdhtools.Rcheck/tests/testthat"
# [1] "C:\\Users\\runneradmin\\AppData\\Local\\Temp\\RtmpYJIkSb\\file442fbea53\\examples"

# Looks like we only have access to the test folder and down from
# GitHub actions - they're running in total isolation

# test_that("example folder exists", {
#   exampleFolder <- normalizePath(file.path("../../..", "examples"))
#
#   print("Example folder check")
#   print(getwd())
#   print(exampleFolder)
#
#   # p <- normalizePath(exampleFolder, mustWork = T)
#   # print(p)
#
#   expect_true(dir.exists(exampleFolder))
# })

# test_that("expected R notebooks are there", {
#   notebooks <- list.files(exampleFolder, pattern=".Rmd", recursive = T, full.names = T)
#
#   expect_equal(4, length(notebooks), info="Number of R notebooks")
# })
