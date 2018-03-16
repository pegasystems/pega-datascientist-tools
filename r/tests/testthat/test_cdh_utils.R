library(testthat)
library(cdhtools)

context("check something")

test_that("dataset export can be read", {
  data <- readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots","dsexports")
  expect_equal(nrow(data), 30)
})

# admdatamart_models <- readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots", "~/Downloads")
# devtools::use_data(admdatamart_models)
# save(admdatamart_models, file="data/admdatamart_models.rda", compress='xz')
