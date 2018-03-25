library(testthat)
library(cdhtools)
library(lubridate)

context("check basic utilities")

test_that("dataset export can be read", {
  data <- readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots","dsexports")
  expect_equal(nrow(data), 30)
})

# to add data:

# admdatamart_models <- readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots", "~/Downloads")
# names(admdatamart_models) <- tolower(names(admdatamart_models))
# for(f in c("pyperformance")) admdatamart_models[[f]] <- as.numeric(admdatamart_models[[f]])
# devtools::use_data(admdatamart_models)
# save(admdatamart_models, file="data/admdatamart_models.rda", compress='xz')
#
# admdatamart_binning <- readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots", "~/Downloads")
# names(admdatamart_binning) <- tolower(names(admdatamart_binning))
# for(f in c("pyperformance")) admdatamart_binning[[f]] <- as.numeric(admdatamart_binning[[f]])
# devtools::use_data(admdatamart_binning)
# save(admdatamart_binning, file="data/admdatamart_binning.rda", compress='xz')
# + describe in R/data.R

# ihsampledata taken from IH in DMSample after initialization
#ihsampledata <- readDSExport("Data-pxStrategyResult_pxInteractionHistory", "~/Downloads")
#ihsampledata <- ihsampledata[ pyApplication=="DMSample"&pySubjectID %in% paste("CE", seq(1:10), sep="-")]
#devtools::use_data(ihsampledata)
#save(ihsampledata, file="data/ihsampledata.rda", compress='xz')
# + describe in R/data.R

test_that("AUC from binning", {
  expect_equal(auc_from_bincounts( c(3,1,0), c(2,0,1)), 0.75)

  # This actually is an example from the COC Mesh article
  positives <- c(50,70,75,80,85,90,110,130,150,160)
  negatives <- c(1440,1350,1170,990,810,765,720,675,630,450)

  expect_equal(auc_from_bincounts(positives, negatives), 0.6871)
  expect_equal(auc_from_bincounts(positives, rep(0, length(positives))), 0.5)
})

test_that("AUC from full arrays", {
  expect_equal(auc_from_probs( c("yes", "yes", "no"), c(0.6, 0.2, 0.2)), 0.75)

  # from https://www.r-bloggers.com/calculating-auc-the-area-under-a-roc-curve/
  category <- c(1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0)
  prediction <- rev(seq_along(category))
  prediction[9:10] <- mean(prediction[9:10])
  expect_equal(auc_from_probs(category, prediction), 0.825)

  # This actually is an example from the COC Mesh article
  # these will be turned into a lengthy format of 10.000 responses
  positives <- c(50,70,75,80,85,90,110,130,150,160)
  negatives <- c(1440,1350,1170,990,810,765,720,675,630,450)

  truth <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(1, positives[i]), rep(0, negatives[i])))}))
  probs <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(positives[i]/(positives[i]+negatives[i]), positives[i]+negatives[i])))}))

  expect_equal(auc_from_probs(truth, probs), 0.6871)
  expect_equal(auc_from_probs(truth, rep(0, length(probs))), 0.5)
})

test_that("GINI conversion", {
  expect_equal(auc2GINI(0.8232), 0.6464)
  expect_equal(auc2GINI(0.5), 0.0)
  expect_equal(auc2GINI(1.0), 1.0)
  expect_equal(auc2GINI(0.6), 0.2)
  expect_equal(auc2GINI(0.4), 0.2)
  expect_equal(auc2GINI(NA), 0.0)
})

test_that("Date conversion", {
  expect_equal(toPRPCDateTime(fromPRPCDateTime("20180316T134127.847 CET")), "20180316T124127.846 GMT")
  expect_equal(toPRPCDateTime(fromPRPCDateTime("20180316T000000.000 EST")), "20180316T050000.000 GMT")
})
