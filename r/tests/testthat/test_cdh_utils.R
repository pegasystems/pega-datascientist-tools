library(testthat)
library(cdhtools)
library(lubridate)

context("check something")

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
# + describe in R/data.R

test_that("AUC from binning", {
  expect_equal(dsm_auc( c(3,1,0), c(2,0,1)), 0.75)

  # This actually is an example from the COC Mesh article
  positives <- c(50,70,75,80,85,90,110,130,150,160)
  negatives <- c(1440,1350,1170,990,810,765,720,675,630,450)

  expect_equal(dsm_auc(positives, negatives), 0.6871)
  expect_equal(dsm_auc(positives, rep(0, length(positives))), 0.5)
})

test_that("AUC from full arrays", {
  expect_equal(safe_auc( c("yes", "yes", "no"), c(0.6, 0.2, 0.2)), 0.75)

  # from https://www.r-bloggers.com/calculating-auc-the-area-under-a-roc-curve/
  category <- c(1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0)
  prediction <- rev(seq_along(category))
  prediction[9:10] <- mean(prediction[9:10])
  expect_equal(safe_auc(category, prediction), 0.825)

  # This actually is an example from the COC Mesh article
  # these will be turned into a lengthy format of 10.000 responses
  positives <- c(50,70,75,80,85,90,110,130,150,160)
  negatives <- c(1440,1350,1170,990,810,765,720,675,630,450)

  truth <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(1, positives[i]), rep(0, negatives[i])))}))
  probs <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(positives[i]/(positives[i]+negatives[i]), positives[i]+negatives[i])))}))

  expect_equal(safe_auc(truth, probs), 0.6871)
  expect_equal(safe_auc(truth, rep(0, length(probs))), 0.5)
})


