library(testthat)

library(rmarkdown)

# Below used in the notebooks. Necessary to repeat?
library(lubridate)
library(cdhtools)
library(data.table)
library(ggplot2)
library(colorspace)

context("Toplevel Examples")

# > test_check("cdhtools")
# [1] "Example folder check"
# [1] "C:/Users/runneradmin/AppData/Local/Temp/RtmpYJIkSb/file442fbea53/cdhtools.Rcheck/tests/testthat"
# [1] "C:\\Users\\runneradmin\\AppData\\Local\\Temp\\RtmpYJIkSb\\file442fbea53\\examples"

# Looks like we only have access to the test folder and down from
# GitHub actions - they're running in total isolation. So until we find
# a solution for that, just running it locally.

exampleFolder <- file.path("../../..", "examples")
offlineReportsFolder <- file.path("../../..", "offlinereports")

test_that("example folder exists", {
  print("Example folder check")
  print(getwd())
  print(exampleFolder)

  # p <- normalizePath(exampleFolder, mustWork = T)
  # print(p)

  # expect_true(dir.exists(exampleFolder))
})

test_that("expected R notebooks are there", {
  if (dir.exists(exampleFolder)) {
    # So below will be skipped on Github actions for now

    notebooks <- list.files(exampleFolder, pattern=".Rmd", recursive = T, full.names = T)

    expect_equal(5, length(notebooks), info="Number of R notebooks")
  }
})

verify_notebook <- function(nb)
{
  print(nb)

  outputfile <- basename(gsub("(.*).Rmd$", "\\1.html", nb))
  if (file.exists(outputfile)) {
    file.remove(outputfile)
  }

  rmarkdown::render(nb, output_dir = ".", quiet = T)

  expect_true(file.exists(outputfile))
}

test_that("check example notebooks", {
  if (dir.exists(exampleFolder)) {
    # So below will be skipped on Github actions for now, only running locally

    notebooks <- list.files(exampleFolder, pattern=".Rmd", recursive = T, full.names = T)
    for (notebook in notebooks) {
      verify_notebook(notebook)
    }
  }
})

test_that("check offline model overview", {
  if (dir.exists(offlineReportsFolder)) {
    # So below will be skipped on Github actions for now, only running locally

    verify_notebook(file.path(offlineReportsFolder, "adaptivemodeloverview.Rmd"))
  }
})

test_that("check offline model report", {
  if (dir.exists(offlineReportsFolder)) {
    # So below will be skipped on Github actions for now, only running locally

    verify_notebook(file.path(offlineReportsFolder, "modelreport.Rmd"))
  }
})

