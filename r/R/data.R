#' Sample ADM Data Mart model and predictor data.
#'
#' Standard \code{list} datastructure as produced by \code{ADMDatamart}.
#'
#' @format A \code{list} with two \code{data.table} objects.
#' \describe{
#'   \item{modeldata}
#'   \item{predictordata}
#' }
#'
#' @source A sample Pega application
"adm_datamart"

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

