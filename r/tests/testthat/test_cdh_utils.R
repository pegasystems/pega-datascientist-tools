library(testthat)
library(cdhtools)

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
