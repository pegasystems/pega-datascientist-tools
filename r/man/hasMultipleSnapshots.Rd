% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adm.R
\name{hasMultipleSnapshots}
\alias{hasMultipleSnapshots}
\title{Check if data has multiple snapshots}
\usage{
hasMultipleSnapshots(dt)
}
\arguments{
\item{dt}{Model or predictor table to check.}
}
\value{
True if there is more than one snapshot
}
\description{
Check if data has multiple snapshots
}
\examples{
data(adm_datamart)
hasMultipleSnapshots(adm_datamart$modeldata)
}
