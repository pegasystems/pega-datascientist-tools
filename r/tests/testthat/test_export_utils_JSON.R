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

  summary <- asDataMart[, .(nbins = .N), by=PredictorName]
  expect_equal(summary$PredictorName, c("AGE", "COUNTRY", "Classifier"),
               info=paste(summary$PredictorName, collapse=";"))
  expect_equal(summary$nbins, c(7, 4, 14), info=paste(summary$nbins, collapse=";"))
})



test_that("ADM Factory to Binning more complex", {
  admFactoryModel <- paste(readLines("d/issue-4-singlebinpredictor.json/FS-CDH-Data-Customer_SalesAdaptiveModel.json.973807ad-af2d-5f81-ba85-9edae1963b8d.json"), collapse=" ")
  asDataMart <- admJSONFactoryToBinning(admFactoryModel)

  expect_equal(nrow(asDataMart), 283)
  expect_equal(ncol(asDataMart), 16)

  expect_equal(asDataMart[PredictorName=="AGE"]$BinSymbol,
               c("<3.04","[3.04, 14.08>","[14.08, 25.04>","[25.04, 29.04>","[29.04, 32.08>","[32.08, 36.08>","[36.08, 43.04>","\u226543.04"))
  expect_equal(asDataMart[PredictorName=="CREDITSTATUS"]$BinSymbol,
               c("Missing","Clear","Due payment"))
})

test_that("ADM Factory to Binning similar to direct DM export", {
  admFactoryModel <- paste(readLines("d/BigModel.json/BigModel.json.pygroup_WebBanners_pyname_BOFFERSUM.json"), collapse=" ")
  asDataMart <- admJSONFactoryToBinning(admFactoryModel)

  dmModels <- fread("d/BigModel_modeldata.csv")
  applyUniformPegaFieldCasing(dmModels)

  dmPredictors <- fread("d/BigModel_predictordata.csv")
  applyUniformPegaFieldCasing(dmPredictors)
  dmPredictors <- dmPredictors[ModelID == dmModels$ModelID[which(dmModels$Name=="BOFFERSUM")]]

  expect_equal(length(unique(asDataMart$PredictorName)), 44)

  # special handling for bin type RESIDUAL, this is set in DM but not in the JSON convert as it does not seem to matter
  dmPredictors[BinType=="RESIDUAL", BinType:="EQUIBEHAVIOR"]

  for (fld in c("ADDRESS", "AGE", "CITY", "COUNTRY", "CREDITHISTORY", "Classifier")) {

    for (col in c("PredictorName", "BinIndex", "Type", "BinType",
                  "EntryType", "BinPositives", "BinNegatives",
                  "Performance", "Lift", "ZRatio" )) {
      expect_true(col %in% names(asDataMart),
                  info=paste("Expected", col, "in converted data from Factory (", sort(names(asDataMart)), ")"))
      expect_true(col %in% names(dmPredictors),
                  info=paste("Expected", col, "in DM export (", sort(names(dmPredictors)), ")"))

      expect_equal(asDataMart[PredictorName==fld][[col]],
                   dmPredictors[PredictorName==fld][[col]], tolerance=1e-6,
                   info=paste("comparing", col, "of", fld,
                              paste(asDataMart[PredictorName==fld][[col]], collapse=";"), "vs",
                              paste(dmPredictors[PredictorName==fld][[col]], collapse=";")))
    }

    # special cases
    # pytype vs pypredictortype name confusion
    expect_equal(asDataMart[PredictorName==fld]$PredictorType,
                 dmPredictors[PredictorName==fld]$Type,
                 info=paste("comparing", "PredictorType", "of", fld))

    # first lower bound has a value in the DM snapshots, NA in the other
    expect_equal(asDataMart[PredictorName==fld]$BinLowerBound[2:length(asDataMart[PredictorName==fld]$BinLowerBound)],
                 dmPredictors[PredictorName==fld]$BinLowerBound[2:length(dmPredictors[PredictorName==fld]$BinLowerBound)],
                 info=paste("comparing", "BinLowerBound", "of", fld))

    # last upper bound has a value in the DM snapshots, NA in the other
    expect_equal(asDataMart[PredictorName==fld]$BinUpperBound[1:(length(asDataMart[PredictorName==fld]$BinUpperBound)-1)],
                 dmPredictors[PredictorName==fld]$BinUpperBound[1:(length(dmPredictors[PredictorName==fld]$BinUpperBound)-1)],
                 info=paste("comparing", "BinUpperBound", "of", fld))
  }
})

