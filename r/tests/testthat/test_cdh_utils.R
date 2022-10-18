library(testthat)
# library(pdstools)
# library(lubridate)

context("check basic utilities")

test_that("checking readDSExport", {
  data <- readDSExport("Data-Decision-ADM-ModelSnapshot_All_20180316T134315_GMT.zip","dsexports")
  expect_equal(nrow(data), 15)
  expect_equal(ncol(data), 22)

  data <- readDSExport("dsexports/Data-Decision-ADM-ModelSnapshot_All_20180316T135038_GMT.zip")
  expect_equal(nrow(data), 30)
  expect_equal(ncol(data), 24)

  data <- readDSExport("Data-Decision-ADM-ModelSnapshot_All","dsexports")
  expect_equal(nrow(data), 30)
  expect_equal(ncol(data), 24)

  data <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_All","dsexports")
  expect_equal(nrow(data), 1755)
  expect_equal(ncol(data), 35)
  expect_equal(length(unique(data$pyModelID)), 15)

  expect_error( readDSExport("Non existing non zip file",""))
  expect_error( readDSExport("Data-Decision-ADM-ModelSnapshot_All_20180316T134315_GMT.zip",""))

  data <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_All","dsexports", excludeComplexTypes = F)
  expect_equal(nrow(data), 1755)
  expect_equal(ncol(data), 35)

  data <- readDSExport(file.path("dsexports", "Data-Decision-ADM-ModelSnapshot_All_20180316T134315_GMT.json"))
  expect_equal(nrow(data), 15)
  expect_equal(ncol(data), 22)
})

test_that("write DS export", {
  data <- readDSExport("Data-Decision-ADM-ModelSnapshot_All_20180316T134315_GMT.zip","dsexports")

  writeDSExport(data, filename = file.path("dsexports", "rewritten.zip"))
  data2 <- readDSExport(file.path("dsexports", "rewritten.zip"))

  expect_equal(nrow(data2), nrow(data))
  expect_equal(ncol(data2), ncol(data))
  expect_equal(names(data2), names(data))
  expect_equal(data2, data)
})


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
  expect_equal(auc_from_bincounts(positives, negatives), 0.6871)
  expect_equal(auc_from_probs(truth, rep(0, length(probs))), 0.5)
  expect_equal(auc_from_probs(rep("Accept, 10"), runif(n = 10)), 0.5)

  # A similar but much smaller example that starts with the bin counts,
  # turns that into truths and probabilities and does the calculation.
  positives <- c(2, 4, 7)
  negatives <- c(4, 2, 1)
  truth <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(1, positives[i]), rep(0, negatives[i])))}))
  probs <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(positives[i]/(positives[i]+negatives[i]), positives[i]+negatives[i])))}))
  expect_equal(auc_from_probs(truth, probs), 0.7637363, 1e-6)
  expect_equal(auc_from_bincounts(positives, negatives), 0.7637363, 1e-6)

  # Example from MLmetrics
  # MLmetrics::AUC(y_pred = logreg$fitted.values, y_true = mtcars$vs)
  # prelude to PRAUC from same package
  data(cars)
  logreg <- glm(formula = vs ~ hp + wt,
                family = binomial(link = "logit"), data = mtcars)
  expect_equal(auc_from_probs(mtcars$vs, logreg$fitted.values), 0.9484127)
})

test_that("AUC PR", {
  expect_equal(aucpr_from_probs( c("yes", "yes", "no"), c(0.6, 0.2, 0.2)),
               0.4166667, tolerance=1e-6)

  # Same from bins
  positives <- c(3,1,0)
  negatives <- c(2,0,1)
  truth <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(1, positives[i]), rep(0, negatives[i])))}))
  probs <- unlist(sapply(seq(length(positives)), function(i){ return(c(rep(positives[i]/(positives[i]+negatives[i]), positives[i]+negatives[i])))}))
  expect_equal(aucpr_from_bincounts( positives, negatives), 0.625)

  positives <- c(50,70,75,80,85,90,110,130,150,160)
  negatives <- c(1440,1350,1170,990,810,765,720,675,630,450)
  expect_equal(aucpr_from_bincounts( positives, negatives), 0.1489611, tolerance=1e-6)

  # Example from MLmetrics
  # MLmetrics::AUC(y_pred = logreg$fitted.values, y_true = mtcars$vs)
  # prelude to PRAUC from same package
  data(cars)
  logreg <- glm(formula = vs ~ hp + wt,
                family = binomial(link = "logit"), data = mtcars)
  expect_equal(aucpr_from_probs(mtcars$vs, logreg$fitted.values),
               0.8610687, tolerance=1e-6)
})

test_that("GINI conversion", {
  expect_equal(auc2GINI(0.8232), 0.6464)
  expect_equal(auc2GINI(0.5), 0.0)
  expect_equal(auc2GINI(1.0), 1.0)
  expect_equal(auc2GINI(0.6), 0.2)
  expect_equal(auc2GINI(0.4), 0.2)
  expect_equal(auc2GINI(NA), 0.0)
})

test_that("Lift", {
  p <- c(0,119,59,69,0)
  n <- c(50,387,105,40,37)
  # see example http://techdocs.rpega.com/display/EPZ/2019/06/21/Z-ratio+calculation+in+ADM
  expect_equal( 100*lift(p,n), c(0, 82.456, 126.13, 221.94, 0), tolerance = 1e-3)
})

test_that("Z-Ratio", {
  p <- c(0,119,59,69,0)
  n <- c(50,387,105,40,37)
  # see example http://techdocs.rpega.com/display/EPZ/2019/06/21/Z-ratio+calculation+in+ADM
  expect_equal( zratio(p,n), c(-7.375207, -3.847732,  2.230442,  7.107804, -6.273136), tolerance = 1e-6)
})


test_that("Date conversion", {
  # not safe to test w/o timezone as this is locale dependent
  expect_equal(toPRPCDateTime(fromPRPCDateTime("20180316T134127.847 CET")), "20180316T124127.846 GMT")
  expect_equal(toPRPCDateTime(fromPRPCDateTime("20180316T000000.000 EST")), "20180316T050000.000 GMT")
})
