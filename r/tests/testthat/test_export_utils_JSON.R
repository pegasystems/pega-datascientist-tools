library(testthat)
library(cdhtools)
library(data.table)
library(jsonlite)

context("check ADM Factory JSON utilities")

test_that("ADM Factory to Binning simple", {
  admFactoryModel <- readLines("d/deeperdive.json/PSEXTCONTR.0089008f-7dc2-5f3f-b44f-6399237fbb85.json")
  asDataMart <- admJSONFactoryToBinning( paste(admFactoryModel, collapse=" ") )

  expect_equal(nrow(asDataMart), 25)
  expect_equal(ncol(asDataMart), 16)

  summary <- asDataMart[, .(nbins = .N), by=pypredictorname]
  expect_equal(summary$pypredictorname, c("AGE", "COUNTRY", "Classifier"), info=paste(summary$pypredictorname, collapse=";"))
  expect_equal(summary$nbins, c(7, 4, 14), info=paste(summary$nbins, collapse=";"))
})

test_that("ADM Factory to Binning more complex", {
  admFactoryModel <- paste(readLines("d/issue-4-singlebinpredictor.json/FS-CDH-Data-Customer_SalesAdaptiveModel.json.973807ad-af2d-5f81-ba85-9edae1963b8d.json"), collapse=" ")
  asDataMart <- admJSONFactoryToBinning(admFactoryModel)

  expect_equal(nrow(asDataMart), 283)
  expect_equal(ncol(asDataMart), 16)

  expect_equal(asDataMart[pypredictorname=="AGE"]$pybinsymbol,
               c("<3.04","[3.04, 14.08>","[14.08, 25.04>","[25.04, 29.04>","[29.04, 32.08>","[32.08, 36.08>","[36.08, 43.04>","\u226543.04"))
  expect_equal(asDataMart[pypredictorname=="CREDITSTATUS"]$pybinsymbol,
               c("Missing","Clear","Due payment"))
})

test_that("ADM Factory to Binning similar to direct DM export", {
  admFactoryModel <- paste(readLines("d/BigModel.json/BigModel.json.pygroup_WebBanners_pyname_BOFFERSUM.json"), collapse=" ")
  asDataMart <- admJSONFactoryToBinning(admFactoryModel)

  dmModels <- fread("d/BigModel_modeldata.csv")
  dmPredictors <- fread("d/BigModel_predictordata.csv")[pymodelid == dmModels$pymodelid[which(dmModels$pyname=="BOFFERSUM")]]

  expect_equal(length(unique(asDataMart$pypredictorname)), 44)

  # special handling for bin type RESIDUAL, this is set in DM but not in the JSON convert as it does not seem to matter
  dmPredictors[pybintype=="RESIDUAL", pybintype:="EQUIBEHAVIOR"]

  for (fld in c("ADDRESS", "AGE", "CITY", "COUNTRY", "CREDITHISTORY", "Classifier")) {
    for (col in c("pypredictorname", "pybinindex", "pytype", "pybintype",
                  "pyentrytype", "pybinpositives", "pybinnegatives",
                  "pybinindex", "pyperformance", "pylift", "pyzratio" )) {
      expect_equal(asDataMart[pypredictorname==fld][[col]], dmPredictors[pypredictorname==fld][[col]], tolerance=1e-6,
                   info=paste("comparing", col, "of", fld,
                              paste(asDataMart[pypredictorname==fld][[col]], collapse=";"), "vs",
                              paste(dmPredictors[pypredictorname==fld][[col]], collapse=";")))
    }
    # special cases
    # pytype vs pypredictortype name confusion
    expect_equal(asDataMart[pypredictorname==fld][["pypredictortype"]], dmPredictors[pypredictorname==fld][["pytype"]],
                 info=paste("comparing", "pypredictortype", "of", fld))
    # first lower bound has a value in the DM snapshots, NA in the other
    expect_equal(asDataMart[pypredictorname==fld][["pybinlowerbound"]][2:length(asDataMart[pypredictorname==fld][["pybinlowerbound"]])],
                 dmPredictors[pypredictorname==fld][["pybinlowerbound"]][2:length(dmPredictors[pypredictorname==fld][["pybinlowerbound"]])],
                 info=paste("comparing", "pybinlowerbound", "of", fld))
    # last upper bound has a value in the DM snapshots, NA in the other
    expect_equal(asDataMart[pypredictorname==fld][["pybinupperbound"]][1:(length(asDataMart[pypredictorname==fld][["pybinupperbound"]])-1)],
                 dmPredictors[pypredictorname==fld][["pybinupperbound"]][1:(length(dmPredictors[pypredictorname==fld][["pybinupperbound"]])-1)],
                 info=paste("comparing", "pybinupperbound", "of", fld))
  }
})

