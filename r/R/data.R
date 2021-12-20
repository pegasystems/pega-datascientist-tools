#' #' Sample model snapshots from the ADM Data Mart.
#' #'
#' #' A \code{data.table} containing two snapshots of 3 models.
#' #'
#' #' @format A \code{data.table} with 30 rows and 16 variables:
#' #' \describe{
#' #'   \item{ModelID}{Model ID}
#' #'   \item{SnapshotTime}{Snapshot time}
#' #'   ...
#' #' }
#' #' @source A sample Pega application
#' "admdatamart_models"
#'
#' #' Sample predictor binning snapshots from the ADM Data Mart.
#' #'
#' #' A \code{data.table} containing a snapshot of the predictor binning of the ADM models in \code{admdatamart_models}.
#' #'
#' #' @format A \code{data.table} with 1755 rows and 32 variables:
#' #' \describe{
#' #'   \item{ModelID}{Model ID}
#' #'   \item{SnapshotTime}{Snapshot time}
#' #'   \item{PredictorName}{Name of the predictor}
#' #'   ...
#' #' }
#' #' @source A sample Pega application
#' "admdatamart_binning"

#' Sample ADM Data Mart model and predictor data.
#'
#' Standard \code{list} datastructure as produced by \code{ADMDatamart}.
#'
#' @format A \code{list} with two \code{data.table} objects.
#' \describe{
#'   \item{modeldata}
#'   \item{predictordata}
#' }
#' @source A sample Pega application
"admdatamart"

#' A small sample of IH data to illustrate how to import and analyze this.
#'
#' A \code{data.table} containing a sample of IH data.
#'
#' @format A \code{data.table} with 50000 interactions for ca 10.000 different customers.
#'
#' @source DMSample
"ihsampledata"

#' A sample export of a Historical Dataset to illustrate how to import and analyze this.
#'
#' A \code{data.table} containing a sample of HDS data.
#'
#' @format A \code{data.table} with 12062 decisions with associated outcomes.
#'
#' @source CDHSample
"hds"

# Run use_data_raw() if you have code to generate data (R data):
# usethis::use_data_raw("mydata_set")
# This will create a data-raw folder.
# Let’s say you want to make a data set mydata_set:
# Create data-raw/mydata_set.R and at the end, run
# usethis::use_data(mydata_set, compress = "xz")
# And this will make a data/mydata_set.rda file.

