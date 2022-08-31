library(testthat)

# library(rmarkdown)

# Below used in the notebooks. Necessary to repeat?
# library(lubridate)
# library(pdstools)
# library(data.table)
# library(ggplot2)
# library(colorspace)

context("Verify the R notebooks in the examples folder")

# > test_check("pdstools")
# [1] "Example folder check"
# [1] "C:/Users/runneradmin/AppData/Local/Temp/RtmpYJIkSb/file442fbea53/pdstools.Rcheck/tests/testthat"
# [1] "C:\\Users\\runneradmin\\AppData\\Local\\Temp\\RtmpYJIkSb\\file442fbea53\\examples"

# Looks like we only have access to the test folder and down from
# GitHub actions - they're running in total isolation. So until we find
# a solution for that, just running it locally.

packageRootFolder <- file.path("../../..")
exampleFolder <- file.path(packageRootFolder, "examples")

test_that("example folder exists", {
  cat("Working dir:", getwd(), fill=T)
  cat("Example dir:", exampleFolder, fill=T)

  # p <- normalizePath(exampleFolder, mustWork = T)
  # print(p)

  # expect_true(dir.exists(exampleFolder))
})

test_that("expected R notebooks are there", {
  if (dir.exists(exampleFolder)) {
    # Below will be skipped on Github actions for now because it only sees
    # the R project folder, not the example folder sibling.

    notebooks <- list.files(exampleFolder, pattern=".Rmd", recursive = T, full.names = T)

    expect_equal(10, length(notebooks), info="Number of R notebooks")
  }
})

# Perhaps watch out in all files or all Rmd files for a
# "source" statement in a line that is not commented out
# sapply(list.files("~/Documents/pega/cdh-datascientist-tools/r/R", "*.R", full.names = T), source)
# any(grepl("^[ ]*[^#]*source", readLines(f)))

verify_notebook <- function(nb)
{
  print(nb)

  outputfile <- basename(gsub("(.*).Rmd$", "\\1.html", nb))
  if (file.exists(outputfile)) {
    file.remove(outputfile)
  }

  tryCatch( rmarkdown::render(nb, output_dir = ".", quiet = T),
            error = function(e) {
              expect_true(F,
                          info = paste("Processing input file", nb, "to", getwd(), "error:", e))
              } )

  expect_true(file.exists(outputfile),
              info = paste("Expected file", outputfile, "in", getwd()))
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

test_that("check not sourcing pdstools files directly", {
  if (dir.exists(exampleFolder)) {
    rFilez <- list.files(packageRootFolder, pattern=".Rmd", recursive = T, full.names = T)
    for (rFile in rFilez) {
      # skip files with .Rcheck in the name
      if (grepl(".Rcheck", rFile, fixed = T)) next

      scanFileForSourceInclusion <- grepl("^[ ]*[^#]*source[[:space:]]*[(|)]", readLines(rFile))
      expect_false(any(scanFileForSourceInclusion),
                   info = paste("Looks like there is a direct include of pdstools source in", rFile,
                                "at lines",
                                paste(which(scanFileForSourceInclusion), collapse = ", ")))

      if (any(scanFileForSourceInclusion)) {
        grFile <<- rFile
        print(grFile)
        stop("Direct includes found")
      }
    }
  }
})
