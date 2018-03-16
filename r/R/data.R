#' Sample model snapshots from the ADM Data Mart.
#'
#' A \code{data.table} containing two snapshots of 3 models.
#'
#' @format A \code{data.table} with 30 rows and 22 variables:
#' \describe{
#'   \item{pyModelID}{Model ID}
#'   \item{pySnapshotTime}{Snapshot time}
#'   ...
#' }
#' @source A sample Pega application
"admdatamart_models"

#' Sample predictor binning snapshots from the ADM Data Mart.
#'
#' A \code{data.table} containing a snapshot of the predictor binning of the ADM models in \code{admdatamart_models}.
#'
#' @format A \code{data.table} with 1755 rows and 35 variables:
#' \describe{
#'   \item{pyModelID}{Model ID}
#'   \item{pySnapshotTime}{Snapshot time}
#'   \item{pyPredictorName}{Name of the predictor}
#'   ...
#' }
#' @source A sample Pega application
"admdatamart_binning"
